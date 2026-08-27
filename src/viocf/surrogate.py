from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DEFAULT_TARGETS = (
    "rms_dbfs",
    "attack_time_s",
    "spectral_centroid_hz",
    "spectral_rolloff_hz",
    "hnr_db",
    "f0_cents_error",
    "onset_time_s",
    "active_duration_s",
)
CATEGORICAL_PREDICTORS = ("technique", "pattern", "register", "timing_variant")
NUMERIC_PREDICTORS = ("cc1_final", "w_tech", "w_cc", "sampling_steps")


def _load_feature_tables(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("At least one feature CSV is required")
    data = pd.concat(frames, ignore_index=True, sort=False)
    if "source" in data:
        data = data.loc[data["source"].eq("model")].copy()
    if data.empty:
        raise ValueError("No model feature rows were found")
    return data


def _prepare_predictors(data: pd.DataFrame) -> pd.DataFrame:
    required = {"prompt_id", "technique", "pattern", "register", "cc1_final"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Surrogate predictors are missing: {missing}")
    prepared = data.copy()
    if "timing_variant" not in prepared:
        prompt_ids = prepared["prompt_id"].astype(str)
        prepared["timing_variant"] = np.select(
            [prompt_ids.str.endswith("_short"), prompt_ids.str.endswith("_long")],
            ["short", "long"],
            default="base",
        )
    for column in ("w_tech", "w_cc"):
        if column not in prepared:
            prepared[column] = 1.0
    if "sampling_steps" not in prepared:
        prepared["sampling_steps"] = 30
    for column in CATEGORICAL_PREDICTORS:
        prepared[column] = prepared[column].fillna("unknown").astype(str)
    for column in NUMERIC_PREDICTORS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def _make_pipeline(*, random_seed: int, n_estimators: int, forest_n_jobs: int) -> Pipeline:
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    preprocess = ColumnTransformer(
        [
            ("categorical", categorical, list(CATEGORICAL_PREDICTORS)),
            ("numeric", numeric, list(NUMERIC_PREDICTORS)),
        ],
        remainder="drop",
    )
    regressor = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=random_seed,
        n_jobs=forest_n_jobs,
    )
    return Pipeline([("preprocess", preprocess), ("regressor", regressor)])


def _normalise_targets(targets: Sequence[str] | str | None, columns: Sequence[str]) -> list[str]:
    if targets is None:
        selected = [target for target in DEFAULT_TARGETS if target in columns]
    elif isinstance(targets, str):
        selected = [targets]
    else:
        selected = list(dict.fromkeys(str(target) for target in targets))
    if not selected:
        raise ValueError("No surrogate targets were selected")
    missing = sorted(set(selected) - set(columns))
    if missing:
        raise ValueError(f"Surrogate targets are missing: {missing}")
    return selected


def train_response_surrogate(
    feature_paths: Iterable[str | Path],
    output_dir: str | Path,
    targets: Sequence[str] | str | None = None,
    *,
    random_seed: int = 20260826,
    n_estimators: int = 400,
    cv_splits: int = 5,
    n_jobs: int = -1,
) -> dict[str, Path]:
    """Fit interpretable ExtraTrees response surfaces with prompt-grouped CV.

    Each target is fitted independently so a feature that is undefined for
    some prompts (for example single-pitch F0 error) does not discard valid
    training rows for every other target. The serialized artifact contains a
    mapping from target name to its fitted sklearn pipeline.
    """
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")

    data = _prepare_predictors(_load_feature_tables(feature_paths))
    selected_targets = _normalise_targets(targets, data.columns)
    predictors = list(CATEGORICAL_PREDICTORS + NUMERIC_PREDICTORS)
    models: dict[str, Pipeline] = {}
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}

    for target_index, target in enumerate(selected_targets):
        target_values = pd.to_numeric(data[target], errors="coerce")
        valid = target_values.notna()
        subset = data.loc[valid].copy()
        y = target_values.loc[valid].astype(float)
        groups = subset["prompt_id"].fillna("unknown").astype(str)
        group_count = int(groups.nunique())
        if len(subset) < 4 or group_count < 2:
            skipped[target] = (
                f"requires at least 4 finite rows and 2 prompt groups; "
                f"found {len(subset)} rows/{group_count} groups"
            )
            continue

        splits = min(cv_splits, group_count)
        seed = int(random_seed) + target_index * 1009
        cv_pipeline = _make_pipeline(
            random_seed=seed,
            n_estimators=n_estimators,
            forest_n_jobs=1,
        )
        predictions = np.asarray(
            cross_val_predict(
                cv_pipeline,
                subset[predictors],
                y,
                groups=groups,
                cv=GroupKFold(n_splits=splits),
                n_jobs=n_jobs,
                method="predict",
            ),
            dtype=float,
        ).reshape(-1)

        mae = float(mean_absolute_error(y, predictions))
        rmse = float(math.sqrt(mean_squared_error(y, predictions)))
        target_std = float(np.std(y.to_numpy(dtype=float), ddof=0))
        r2 = float(r2_score(y, predictions)) if len(y) >= 2 else float("nan")
        metric_rows.append(
            {
                "target": target,
                "rows": len(subset),
                "prompt_groups": group_count,
                "cv_splits": splits,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "target_std": target_std,
                "normalised_mae": mae / target_std if target_std > 0 else float("nan"),
            }
        )
        for row_index, observed, predicted in zip(subset.index, y, predictions):
            prediction_rows.append(
                {
                    "row_index": int(row_index),
                    "clip_id": (
                        str(subset.at[row_index, "clip_id"])
                        if "clip_id" in subset
                        else str(row_index)
                    ),
                    "prompt_id": str(subset.at[row_index, "prompt_id"]),
                    "target": target,
                    "observed": float(observed),
                    "predicted_cv": float(predicted),
                    "residual": float(observed - predicted),
                }
            )

        fitted = _make_pipeline(
            random_seed=seed,
            n_estimators=n_estimators,
            forest_n_jobs=n_jobs,
        ).fit(subset[predictors], y)
        models[target] = fitted
        transformed_names = fitted.named_steps["preprocess"].get_feature_names_out()
        importances = fitted.named_steps["regressor"].feature_importances_
        for feature_name, importance in zip(transformed_names, importances):
            importance_rows.append(
                {
                    "target": target,
                    "transformed_feature": str(feature_name),
                    "importance": float(importance),
                }
            )

    if not models:
        detail = "; ".join(f"{target}: {reason}" for target, reason in skipped.items())
        raise ValueError(f"No target had enough data for grouped cross-validation. {detail}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "response_surrogate.joblib"
    metrics_path = output / "surrogate_grouped_cv_metrics.csv"
    predictions_path = output / "surrogate_grouped_cv_predictions.csv"
    importance_path = output / "surrogate_feature_importance.csv"
    metadata_path = output / "surrogate_metadata.json"

    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    pd.DataFrame(importance_rows).to_csv(importance_path, index=False)
    metadata = {
        "artifact_format": "viocf-extra-trees-response-surrogate-v1",
        "training_rows_available": len(data),
        "grouping_variable": "prompt_id",
        "categorical_predictors": list(CATEGORICAL_PREDICTORS),
        "numeric_predictors": list(NUMERIC_PREDICTORS),
        "targets_requested": selected_targets,
        "targets_trained": list(models),
        "targets_skipped": skipped,
        "n_estimators": n_estimators,
        "random_seed": random_seed,
        "interpretation_warning": (
            "Feature importance is predictive, not causal; confirm interventions with generated audio."
        ),
    }
    joblib.dump(
        {
            "models": models,
            "predictors": predictors,
            "metadata": metadata,
        },
        model_path,
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "model": model_path,
        "metrics": metrics_path,
        "predictions": predictions_path,
        "importance": importance_path,
        "metadata": metadata_path,
    }
