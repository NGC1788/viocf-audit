from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    """빈 CSV 를 만나도 죽지 않는다.

    실연주 녹음이 아직 없으면 실악기 기준선이 필요한 지표 파일이 헤더도 없이
    0 바이트로 남는다. pandas 는 거기서 EmptyDataError 를 던지고, 그 탓에
    **모델만으로 나오는 그림까지 통째로 못 만든다**(실제로 겪음).
    """
    try:
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _relative_rms(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.loc[frame["profile"].eq("constant")].copy()
    if "unit_id" not in data:
        model_unit = data.get("noise_group", data.get("replicate", pd.Series(0, index=data.index))).astype(str)
        real_unit = (
            data.get("performer_id", pd.Series("P1", index=data.index)).fillna("P1").astype(str)
            + ":"
            + data.get("violin_id", pd.Series("V0", index=data.index)).fillna("V0").astype(str)
            + ":"
            + data.get("replicate", data.get("take", pd.Series(1, index=data.index))).astype(str)
        )
        data["unit_id"] = np.where(data["source"].eq("model"), model_unit, real_unit)
    keys = ["source", "prompt_id", "technique", "unit_id"]
    data["relative_rms_db"] = np.nan
    for index in data.groupby(keys, dropna=False).groups.values():
        block = data.loc[index]
        baseline = block.loc[block["dynamic_label"].eq("mf"), "rms_dbfs"].mean()
        data.loc[index, "relative_rms_db"] = block["rms_dbfs"] - baseline
    return data


def make_figures(
    feature_paths: Iterable[str | Path],
    metrics_dir: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    frames = [_read_csv_or_empty(path) for path in feature_paths]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("At least one feature CSV is required")
    features = pd.concat(frames, ignore_index=True, sort=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    paths: list[Path] = []

    if {"rms_dbfs", "cc1_final", "source", "technique"}.issubset(features.columns):
        response = _relative_rms(features)
        figure, axis = plt.subplots(figsize=(11, 7))
        sns.lineplot(
            data=response,
            x="cc1_final",
            y="relative_rms_db",
            hue="source",
            style="technique",
            markers=True,
            dashes=False,
            errorbar=("ci", 95),
            ax=axis,
        )
        axis.axhline(0, color="black", linewidth=1)
        axis.set(
            xlabel="CC1",
            ylabel="Relative RMS (dB; within-block mf = 0)",
            title="Real vs model dynamics response",
        )
        figure.tight_layout()
        path = output / "fig1_dynamics_response.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        paths.append(path)

    metric_root = Path(metrics_dir)
    contrasts_path = metric_root / "contrasts.csv"
    if contrasts_path.exists():
        contrasts = _read_csv_or_empty(contrasts_path)
        metadata = {
            "source",
            "prompt_id",
            "unit_id",
            "control_type",
            "technique",
            "dynamic_label",
            "contrast",
        }
        numeric = [
            column
            for column in contrasts.columns
            if column not in metadata and pd.api.types.is_numeric_dtype(contrasts[column])
        ]
        if numeric:
            matrix = contrasts.groupby(["source", "contrast"])[numeric].mean()
            sources = [source for source in ("real", "model") if source in matrix.index]
            figure, axes = plt.subplots(
                1, len(sources), figsize=(max(9, 7 * len(sources)), max(6, 0.4 * len(matrix)))
            )
            if len(sources) == 1:
                axes = [axes]
            for axis, source in zip(axes, sources):
                table = matrix.loc[source]
                sns.heatmap(table, center=0, cmap="vlag", ax=axis, cbar_kws={"label": "Robust-scaled effect"})
                axis.set_title(f"{source}: control-response matrix")
                axis.set(xlabel="Output feature", ylabel="Control intervention")
            figure.tight_layout()
            path = output / "fig2_response_matrices.png"
            figure.savefig(path, dpi=200)
            plt.close(figure)
            paths.append(path)

    delayed_path = metric_root / "delayed_branch.csv"
    if delayed_path.exists():
        delayed = _read_csv_or_empty(delayed_path)
        delayed = delayed.loc[delayed["technique"].isin(["sustain", "pizzicato"])]
        if not delayed.empty:
            figure, axes = plt.subplots(1, 2, figsize=(13, 5))
            sns.barplot(data=delayed, x="technique", y="future_leak", hue="source", ax=axes[0])
            axes[0].set_title("Future-control leakage before branch")
            sns.barplot(data=delayed, x="technique", y="post_effect", hue="source", ax=axes[1])
            axes[1].set_title("Post-branch p vs f effect")
            figure.tight_layout()
            path = output / "fig3_delayed_branch.png"
            figure.savefig(path, dpi=200)
            plt.close(figure)
            paths.append(path)

    alignment_path = metric_root / "effect_alignment.csv"
    leakage_path = metric_root / "excess_leakage.csv"
    if alignment_path.exists() and leakage_path.exists():
        alignment = _read_csv_or_empty(alignment_path)
        leakage = _read_csv_or_empty(leakage_path)
        if not alignment.empty and not leakage.empty:
            figure, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.barplot(
                data=alignment,
                x="control_type",
                y="effect_alignment_cosine",
                errorbar=("ci", 95),
                ax=axes[0],
            )
            axes[0].set_ylim(-1, 1)
            axes[0].set_title("Counterfactual effect alignment")
            sns.barplot(
                data=leakage,
                x="control_type",
                y="excess_leakage",
                errorbar=("ci", 95),
                ax=axes[1],
            )
            axes[1].set_title("Human-calibrated excess leakage")
            figure.tight_layout()
            path = output / "fig4_headline_metrics.png"
            figure.savefig(path, dpi=200)
            plt.close(figure)
            paths.append(path)
    return paths
