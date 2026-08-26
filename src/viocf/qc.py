from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audio import amplitude_to_db, detect_active_region, read_audio, rms


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def qc_audio(path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    audio = read_audio(path, mono=True)
    samples = audio.samples
    analysis = config["analysis"]
    region = detect_active_region(
        samples,
        audio.sample_rate,
        threshold_db_above_noise=float(analysis["active_threshold_db_above_noise"]),
    )
    active = (
        samples[int(region["start_sample"]) : int(region["end_sample"])]
        if region["active"]
        else np.array([], dtype=np.float32)
    )
    active_rms = rms(active)
    noise_rms = float(region["noise_rms"])
    snr_db = float(amplitude_to_db(active_rms / max(noise_rms, 1e-12)))
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    clip_threshold = float(analysis["clip_threshold"])
    clipped = int(np.count_nonzero(np.abs(samples) >= clip_threshold))
    reasons: list[str] = []
    if audio.sample_rate != int(config["sample_rate"]):
        reasons.append(f"sample_rate_{audio.sample_rate}")
    if audio.channels != 1:
        reasons.append(f"channels_{audio.channels}")
    if clipped > 0:
        reasons.append("clipping")
    if not region["active"]:
        reasons.append("near_silence")
    if np.isfinite(snr_db) and snr_db < float(analysis["min_snr_db"]):
        reasons.append("low_snr")
    if not np.all(np.isfinite(samples)):
        reasons.append("non_finite_samples")
    return {
        "resolved_audio_path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "sample_rate": audio.sample_rate,
        "channels_original": audio.channels,
        "subtype": audio.subtype,
        "duration_s": len(samples) / audio.sample_rate,
        "peak_dbfs": float(amplitude_to_db(peak)),
        "active_rms_dbfs": float(amplitude_to_db(active_rms)),
        "noise_dbfs": float(amplitude_to_db(noise_rms)),
        "snr_db": snr_db,
        "clipped_samples": clipped,
        "clipped_fraction": clipped / max(1, len(samples)),
        "active_start_s": float(region["start_s"]),
        "active_end_s": float(region["end_s"]),
        "qc_pass": not reasons,
        "qc_reasons": ";".join(reasons),
    }


def qc_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    project_root: str | Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    root = Path(project_root).resolve()
    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, Any]] = []
    for record in manifest.to_dict(orient="records"):
        audio_path = Path(str(record["audio_path"]))
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        row = dict(record)
        if not audio_path.exists():
            row.update(
                {
                    "resolved_audio_path": str(audio_path),
                    "qc_pass": False,
                    "qc_reasons": "missing_audio",
                }
            )
        else:
            try:
                row.update(qc_audio(audio_path, config))
            except Exception as exc:  # noqa: BLE001 - preserve per-file QC failures
                row.update(
                    {
                        "resolved_audio_path": str(audio_path),
                        "qc_pass": False,
                        "qc_reasons": f"{type(exc).__name__}: {exc}",
                    }
                )
        rows.append(row)
    frame = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "rows": len(frame),
                "passed": int(frame["qc_pass"].fillna(False).sum()),
                "failed": int((~frame["qc_pass"].fillna(False)).sum()),
                "failure_counts": (
                    frame.loc[~frame["qc_pass"].fillna(False), "qc_reasons"]
                    .value_counts(dropna=False)
                    .to_dict()
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return frame
