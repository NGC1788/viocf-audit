from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from viocf.violet import collect_violet_run


def _write_fake_run(tmp_path: Path, *, second_seed: int = 1234) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    run = tmp_path / "run" / "test_samples"
    run.mkdir(parents=True)
    clips = ["group-rep01__t-sustain__d-p", "group-rep01__t-sustain__d-f"]
    manifest = pd.DataFrame(
        [
            {
                "clip_id": clip_id,
                "noise_group": "group-rep01",
                "seed": 1234,
                "w_tech": 0.5,
                "w_cc": 2.0,
                "audio_path": f"data/model_audio/{clip_id}.wav",
            }
            for clip_id in clips
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    records = []
    for index, clip_id in enumerate(clips):
        wav_name = f"{clip_id}.wav"
        sf.write(run / wav_name, np.zeros(480, dtype=np.float32), 48000)
        records.append(
            {
                "filename": clip_id,
                "saved_audio": wav_name,
                "render_seed": 1234 if index == 0 else second_seed,
                "render_attempt": 1,
                "effective_w_tech": 0.5,
                "effective_w_cc": 2.0,
            }
        )
    with (run / "conditioning_debug.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return project, run.parent, manifest_path


def test_collector_accepts_manifest_expected_guidance_weights(tmp_path: Path) -> None:
    project, run, manifest = _write_fake_run(tmp_path)
    report = tmp_path / "collect.csv"
    frame = collect_violet_run(run, manifest, project, report, copy_audio=False)
    groups = pd.read_csv(report.with_name("collect_groups.csv"))
    summary = json.loads(report.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert frame["render_seed_matches_expected"].all()
    assert frame["w_tech_matches_expected"].all()
    assert frame["w_cc_matches_expected"].all()
    assert groups["pairing_pass"].all()
    assert summary["all_pass"] is True


def test_collector_rejects_a_condition_specific_seed(tmp_path: Path) -> None:
    project, run, manifest = _write_fake_run(tmp_path, second_seed=9999)
    report = tmp_path / "collect.csv"
    collect_violet_run(run, manifest, project, report, copy_audio=False)
    groups = pd.read_csv(report.with_name("collect_groups.csv"))
    summary = json.loads(report.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert not groups["pairing_pass"].all()
    assert summary["all_pass"] is False
