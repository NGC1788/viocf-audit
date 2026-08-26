from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mido
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def _relative_response(frame: pd.DataFrame, feature: str, baseline_cc: int = 64) -> pd.DataFrame:
    data = frame.loc[frame["profile"].eq("constant")].copy()
    if "unit_id" not in data:
        if "noise_group" in data:
            model_unit = data["noise_group"].fillna(data.get("replicate", 0)).astype(str)
        else:
            model_unit = data.get("replicate", pd.Series(0, index=data.index)).astype(str)
        performer = data.get("performer_id", pd.Series("P1", index=data.index)).fillna("P1").astype(str)
        violin = data.get("violin_id", pd.Series("V0", index=data.index)).fillna("V0").astype(str)
        take = data.get("take", data.get("replicate", pd.Series(1, index=data.index))).astype(str)
        real_unit = performer + ":" + violin + ":" + take
        data["unit_id"] = np.where(data["source"].eq("model"), model_unit, real_unit)
    group_keys = ["source", "prompt_id", "technique", "unit_id"]
    records: list[dict[str, Any]] = []
    for key, group in data.groupby(group_keys, dropna=False):
        cells = group.groupby("cc1_final")[feature].mean()
        if baseline_cc not in cells.index:
            continue
        baseline = float(cells.loc[baseline_cc])
        for cc1, value in cells.items():
            records.append(
                {
                    "source": key[0],
                    "prompt_id": key[1],
                    "technique": key[2],
                    "unit_id": key[3],
                    "cc1": int(cc1),
                    "relative_response": float(value) - baseline,
                }
            )
    return pd.DataFrame(records)


def _fit_curve(x: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    order = np.argsort(x)
    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[order], y[order])


def fit_technique_calibration(
    feature_paths: Iterable[str | Path],
    output_dir: str | Path,
    feature: str = "rms_dbfs",
    fit_violins: tuple[str, ...] = ("V1", "V2"),
    heldout_violin: str = "V3",
    baseline_cc: int = 64,
) -> dict[str, Path]:
    frames = [pd.read_csv(path) for path in feature_paths]
    if not frames:
        raise ValueError("At least one feature CSV is required")
    data = pd.concat(frames, ignore_index=True, sort=False)
    if feature not in data:
        raise ValueError(f"Calibration feature is missing: {feature}")
    response = _relative_response(data, feature, baseline_cc=baseline_cc)
    if response.empty:
        raise ValueError("No complete CC1 response blocks were found")

    model = response.loc[response["source"].eq("model")]
    real_source = data.loc[data["source"].eq("real")]
    if "violin_id" in real_source and fit_violins:
        permitted_units = set(
            (
                real_source.loc[real_source["violin_id"].astype(str).isin(fit_violins), "performer_id"].fillna("P1").astype(str)
                + ":"
                + real_source.loc[real_source["violin_id"].astype(str).isin(fit_violins), "violin_id"].astype(str)
                + ":"
                + real_source.loc[real_source["violin_id"].astype(str).isin(fit_violins), "take"].astype(str)
            ).tolist()
        )
        real_fit = response.loc[response["source"].eq("real") & response["unit_id"].isin(permitted_units)]
        if real_fit.empty:
            real_fit = response.loc[response["source"].eq("real")]
    else:
        real_fit = response.loc[response["source"].eq("real")]
    if model.empty or real_fit.empty:
        raise ValueError("Both model and real feature rows are required for calibration")

    desired_grid = np.arange(0, 128, dtype=float)
    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    techniques = sorted(set(model["technique"]) & set(real_fit["technique"]))
    for technique in techniques:
        model_points = (
            model.loc[model["technique"].eq(technique)]
            .groupby("cc1")["relative_response"]
            .mean()
            .sort_index()
        )
        real_points = (
            real_fit.loc[real_fit["technique"].eq(technique)]
            .groupby("cc1")["relative_response"]
            .mean()
            .sort_index()
        )
        if len(model_points) < 3 or len(real_points) < 3:
            continue
        model_curve = _fit_curve(
            model_points.index.to_numpy(dtype=float), model_points.to_numpy(dtype=float)
        )
        real_curve = _fit_curve(
            real_points.index.to_numpy(dtype=float), real_points.to_numpy(dtype=float)
        )
        model_dense = model_curve.predict(desired_grid)
        real_dense = real_curve.predict(desired_grid)
        corrected = np.array(
            [desired_grid[int(np.argmin(np.abs(model_dense - target)))] for target in real_dense],
            dtype=float,
        )
        # Enforce an interpretable monotonic mapping even if inverse ambiguity occurs.
        mapping_curve = IsotonicRegression(
            increasing=True, y_min=0.0, y_max=127.0, out_of_bounds="clip"
        ).fit(desired_grid, corrected)
        corrected = np.clip(np.rint(mapping_curve.predict(desired_grid)), 0, 127).astype(int)
        for desired_cc, corrected_cc, target, predicted in zip(
            desired_grid.astype(int), corrected, real_dense, model_curve.predict(corrected)
        ):
            curve_rows.append(
                {
                    "technique": technique,
                    "desired_cc1": int(desired_cc),
                    "corrected_cc1": int(corrected_cc),
                    "target_real_relative_response": float(target),
                    "predicted_model_relative_response": float(predicted),
                    "calibration_feature": feature,
                }
            )
        for desired_cc in sorted(set(real_points.index.astype(int))):
            index = int(np.clip(desired_cc, 0, 127))
            rows.append(
                {
                    "technique": technique,
                    "desired_cc1": index,
                    "corrected_cc1": int(corrected[index]),
                    "before_error": float(abs(model_curve.predict([index])[0] - real_curve.predict([index])[0])),
                    "surrogate_after_error": float(
                        abs(model_curve.predict([corrected[index]])[0] - real_curve.predict([index])[0])
                    ),
                    "calibration_feature": feature,
                    "fit_violins": ";".join(fit_violins),
                    "heldout_violin": heldout_violin,
                }
            )

    mapping = pd.DataFrame(curve_rows)
    summary = pd.DataFrame(rows)
    if mapping.empty:
        raise ValueError("No technique had enough response levels for calibration")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mapping_path = output / "cc1_calibration_curve.csv"
    summary_path = output / "cc1_calibration_summary.csv"
    mapping.to_csv(mapping_path, index=False)
    summary.to_csv(summary_path, index=False)

    figure_path = output / "cc1_calibration_mapping.png"
    figure, axis = plt.subplots(figsize=(8, 6))
    for technique, group in mapping.groupby("technique"):
        axis.plot(group["desired_cc1"], group["corrected_cc1"], label=technique)
    axis.plot([0, 127], [0, 127], linestyle="--", color="black", linewidth=1, label="identity")
    axis.set(xlabel="Desired CC1", ylabel="Corrected CC1", title=f"Technique-aware calibration ({feature})")
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    metadata_path = output / "calibration_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "feature": feature,
                "fit_violins": list(fit_violins),
                "heldout_violin": heldout_violin,
                "baseline_cc": baseline_cc,
                "warning": "After-errors are surrogate predictions; generate calibrated audio and re-audit for the actual result.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "mapping": mapping_path,
        "summary": summary_path,
        "figure": figure_path,
        "metadata": metadata_path,
    }


def _lookup_corrected(mapping: pd.DataFrame, technique: str, desired_cc: int) -> int:
    subset = mapping.loc[mapping["technique"].eq(technique)].sort_values("desired_cc1")
    if subset.empty:
        return int(desired_cc)
    corrected = np.interp(
        float(desired_cc),
        subset["desired_cc1"].to_numpy(dtype=float),
        subset["corrected_cc1"].to_numpy(dtype=float),
    )
    return int(np.clip(round(corrected), 0, 127))


def rewrite_cc1(
    source_midi: str | Path,
    destination_midi: str | Path,
    corrected_initial: int,
    corrected_final: int,
    delayed: bool,
) -> Path:
    midi = mido.MidiFile(source_midi)
    cc_messages = [
        message
        for track in midi.tracks
        for message in track
        if message.type == "control_change" and message.control == 1
    ]
    if not cc_messages:
        raise ValueError(f"MIDI has no CC1 events: {source_midi}")
    for index, message in enumerate(cc_messages):
        if delayed and index == len(cc_messages) - 1:
            message.value = int(corrected_final)
        else:
            message.value = int(corrected_initial)
    destination = Path(destination_midi)
    destination.parent.mkdir(parents=True, exist_ok=True)
    midi.save(destination)
    return destination


def apply_calibration_to_manifest(
    manifest_path: str | Path,
    mapping_path: str | Path,
    project_root: str | Path,
    output_manifest: str | Path,
    calibration_id: str = "isotonic-v1",
) -> pd.DataFrame:
    root = Path(project_root).resolve()
    manifest = pd.read_csv(manifest_path)
    mapping = pd.read_csv(mapping_path)
    rows: list[dict[str, Any]] = []
    for record in manifest.to_dict(orient="records"):
        technique = str(record["technique"])
        initial = _lookup_corrected(mapping, technique, int(record["cc1_initial"]))
        final = _lookup_corrected(mapping, technique, int(record["cc1_final"]))
        source_midi = Path(str(record["midi_path"]))
        if not source_midi.is_absolute():
            source_midi = root / source_midi
        clip_id = f"{record['clip_id']}__cal-{calibration_id}"
        destination_midi = root / "data" / "midi" / "calibrated" / f"{clip_id}.mid"
        rewrite_cc1(
            source_midi,
            destination_midi,
            initial,
            final,
            delayed=str(record.get("profile", "constant")) == "delayed",
        )
        updated = dict(record)
        updated.update(
            {
                "clip_id": clip_id,
                "cc1_original_initial": record["cc1_initial"],
                "cc1_original_final": record["cc1_final"],
                "cc1_initial": initial,
                "cc1_final": final,
                "calibration_id": calibration_id,
                "midi_path": str(destination_midi.relative_to(root)),
                "audio_path": f"data/model_audio/{clip_id}.wav",
                "status": "planned_calibrated",
            }
        )
        rows.append(updated)
    output = pd.DataFrame(rows)
    path = Path(output_manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output

