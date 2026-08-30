from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
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


def _qc_one(job: tuple[int, dict[str, Any], str, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    """워커 하나가 클립 하나를 검사한다. 실패도 행으로 보존한다."""
    index, record, audio_path_text, config = job
    audio_path = Path(audio_path_text)
    row = dict(record)
    if not audio_path.exists():
        row.update({
            "resolved_audio_path": str(audio_path),
            "qc_pass": False,
            "qc_reasons": "missing_audio",
        })
        return index, row
    try:
        row.update(qc_audio(audio_path, config))
    except Exception as exc:  # noqa: BLE001 - preserve per-file QC failures
        row.update({
            "resolved_audio_path": str(audio_path),
            "qc_pass": False,
            "qc_reasons": f"{type(exc).__name__}: {exc}",
        })
    return index, row


def _default_workers() -> int:
    # 코어를 다 쓰면 서버가 먹통이 된다. 2개는 남긴다.
    return max(1, (os.cpu_count() or 2) - 2)


def qc_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    project_root: str | Path,
    config: dict[str, Any],
    workers: int | None = None,
    progress_every: int = 1000,
) -> pd.DataFrame:
    """manifest 의 오디오를 전수 검사한다.

    ⚠ 병렬이다. 클립마다 파일 전체를 읽고 **SHA-256 까지 계산**하므로 직렬로는
    1만 8천 클립에 수십 분이 든다. 그동안 나머지 코어가 논다(실제로 겪음).

    결정성은 유지된다. 검사에 난수가 없고, 결과를 manifest 순서로 되돌린다.
    워커 수를 바꿔도 출력 파일은 같다.
    """
    root = Path(project_root).resolve()
    manifest = pd.read_csv(manifest_path)

    jobs: list[tuple[int, dict[str, Any], str, dict[str, Any]]] = []
    for index, record in enumerate(manifest.to_dict(orient="records")):
        audio_path = Path(str(record["audio_path"]))
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        jobs.append((index, record, str(audio_path), config))

    worker_count = workers if workers is not None else _default_workers()
    collected: dict[int, dict[str, Any]] = {}

    if worker_count <= 1 or len(jobs) <= 1:
        for job in jobs:
            index, row = _qc_one(job)
            collected[index] = row
    else:
        environment = {
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
        # spawn 방식에서 자식이 viocf 를 다시 import 해야 한다 (features.py 와 같은 이유).
        package_root = str(Path(__file__).resolve().parents[1])
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        if package_root not in existing_pythonpath.split(os.pathsep):
            environment["PYTHONPATH"] = (
                f"{package_root}{os.pathsep}{existing_pythonpath}"
                if existing_pythonpath
                else package_root
            )
        previous = {name: os.environ.get(name) for name in environment}
        os.environ.update(environment)
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as pool:
                for done, (index, row) in enumerate(
                    pool.map(_qc_one, jobs, chunksize=8), start=1
                ):
                    collected[index] = row
                    if progress_every and done % progress_every == 0:
                        print(
                            f"  QC {done:,}/{len(jobs):,} (워커 {worker_count}개)",
                            file=sys.stderr,
                            flush=True,
                        )
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    # manifest 순서로 되돌린다 — 워커 수와 무관하게 같은 파일이 나와야 한다.
    rows = [collected[index] for index in sorted(collected)]
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
