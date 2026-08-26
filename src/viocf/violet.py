from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .qc import sha256_file


def _load_jsonl_files(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL {path}:{line_number}: {exc}") from exc
                record["debug_manifest_path"] = str(path)
                records.append(record)
    return records


def collect_violet_run(
    run_dir: str | Path,
    manifest_path: str | Path,
    project_root: str | Path,
    output_report: str | Path,
    copy_audio: bool = True,
) -> pd.DataFrame:
    run = Path(run_dir).resolve()
    root = Path(project_root).resolve()
    manifest = pd.read_csv(manifest_path)
    debug_files = sorted(run.rglob("conditioning_debug.jsonl"))
    if not debug_files:
        raise FileNotFoundError(f"No conditioning_debug.jsonl below {run}")
    debug = pd.DataFrame(_load_jsonl_files(debug_files))
    if debug.empty:
        raise ValueError("VIOLET debug manifests contain no records")
    if debug["filename"].duplicated().any():
        duplicates = debug.loc[debug["filename"].duplicated(), "filename"].tolist()
        raise ValueError(f"Duplicate VIOLET output stems: {duplicates[:5]}")

    merged = manifest.merge(debug, left_on="clip_id", right_on="filename", how="left", suffixes=("", "_violet"))
    merged["violet_found"] = merged["saved_audio"].notna()
    expected_seed = pd.to_numeric(merged.get("seed"), errors="coerce")
    actual_seed = pd.to_numeric(merged.get("render_seed"), errors="coerce")
    merged["render_seed_matches_expected"] = expected_seed.eq(actual_seed)
    for weight in ("tech", "cc"):
        expected = pd.to_numeric(merged.get(f"w_{weight}"), errors="coerce")
        actual = pd.to_numeric(merged.get(f"effective_w_{weight}"), errors="coerce")
        merged[f"w_{weight}_matches_expected"] = np.isclose(
            expected,
            actual,
            rtol=0.0,
            atol=1e-8,
            equal_nan=False,
        )
    copied_paths: list[str | None] = []
    hashes: list[str | None] = []
    for record in merged.to_dict(orient="records"):
        if not bool(record.get("violet_found")):
            copied_paths.append(None)
            hashes.append(None)
            continue
        debug_manifest = Path(str(record["debug_manifest_path"]))
        source_audio = debug_manifest.parent / str(record["saved_audio"])
        if not source_audio.exists():
            copied_paths.append(None)
            hashes.append(None)
            continue
        destination = root / str(record["audio_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha256_file(source_audio)
        if copy_audio:
            if destination.exists() and sha256_file(destination) != source_hash:
                raise FileExistsError(
                    f"Refusing to overwrite different audio: {destination}. Move it explicitly first."
                )
            if not destination.exists():
                shutil.copy2(source_audio, destination)
        copied_paths.append(str(destination))
        hashes.append(source_hash)
    merged["collected_audio_path"] = copied_paths
    merged["audio_sha256"] = hashes

    # Pairing QA: every counterfactual cell in a noise group must have one seed,
    # one attempt, and active dynamics/technique weights.
    group_qa = (
        merged.loc[merged["violet_found"]]
        .groupby("noise_group", dropna=False)
        .agg(
            output_count=("clip_id", "size"),
            render_seed_unique=("render_seed", "nunique"),
            render_seed_matches_expected=("render_seed_matches_expected", "all"),
            render_attempt_max=("render_attempt", "max"),
            w_tech_min=("effective_w_tech", "min"),
            w_tech_max=("effective_w_tech", "max"),
            w_cc_min=("effective_w_cc", "min"),
            w_cc_max=("effective_w_cc", "max"),
            w_tech_matches_expected=("w_tech_matches_expected", "all"),
            w_cc_matches_expected=("w_cc_matches_expected", "all"),
        )
        .reset_index()
    )
    group_qa["pairing_pass"] = (
        (group_qa["render_seed_unique"] == 1)
        & group_qa["render_seed_matches_expected"]
        & (group_qa["render_attempt_max"] == 1)
        & group_qa["w_tech_matches_expected"]
        & group_qa["w_cc_matches_expected"]
    )

    report = Path(output_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(report, index=False)
    group_qa.to_csv(report.with_name(report.stem + "_groups.csv"), index=False)
    summary = {
        "expected_outputs": len(manifest),
        "found_outputs": int(merged["violet_found"].sum()),
        "missing_outputs": merged.loc[~merged["violet_found"], "clip_id"].tolist(),
        "pairing_groups": len(group_qa),
        "pairing_failed_groups": group_qa.loc[~group_qa["pairing_pass"], "noise_group"].tolist(),
        "all_pass": bool(merged["violet_found"].all() and group_qa["pairing_pass"].all()),
    }
    report.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return merged
