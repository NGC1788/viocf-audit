from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from viocf.config import load_config
from viocf.midi import inspect_violet_midi
from viocf.surrogate import train_response_surrogate
from viocf.sweep import create_compute_sweep, planned_sweep_counts


def _make_config(root: Path) -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    config = {
        "sample_rate": 48000,
        "clip_seconds": 10.0,
        "note_onset_seconds": 0.75,
        "tempo_bpm": 96,
        "techniques": {
            "sustain": 36,
            "staccato": 40,
            "pizzicato": 43,
            "legato_slur": 49,
        },
        "dynamics": {"p": 32, "mf": 64, "f": 96},
        "model": {"base_seed": 20260826, "replicates": 5, "w_tech": 1.0, "w_cc": 1.0},
        "real": {"performer_id": "P1", "violins": ["V1", "V2", "V3"], "takes_per_cell": 2},
        "analysis": {"random_seed": 20260826},
    }
    path = config_dir / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_default_sweep_counts() -> None:
    assert planned_sweep_counts() == {
        "dense_clips": 6912,
        "guidance_clips": 4608,
        "total_clips": 11520,
    }


def test_small_sweep_writes_paired_manifests_and_valid_midi(tmp_path: Path) -> None:
    config = load_config(_make_config(tmp_path))
    outputs = create_compute_sweep(config, dense_replicates=1, guidance_replicates=1)
    dense = pd.read_csv(outputs["dense"])
    guidance = pd.read_csv(outputs["guidance_all"])
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))

    assert len(dense) == 24 * 9 * 4
    assert len(guidance) == 6 * 4 * 3 * 16
    assert summary["total_clips"] == len(dense) + len(guidance)
    assert summary["guidance_pair_manifests"] == 16
    assert dense["clip_id"].is_unique
    assert guidance["clip_id"].is_unique
    assert set(dense.groupby("noise_group").size()) == {36}
    assert set(guidance.groupby("noise_group").size()) == {192}
    assert dense.groupby("noise_group")["seed"].nunique().max() == 1
    assert guidance.groupby("noise_group")["seed"].nunique().max() == 1
    assert all(row.clip_id.split("__", maxsplit=1)[0] == row.noise_group for row in dense.itertuples())

    pair_paths = [path for key, path in outputs.items() if key.startswith("guidance_wt")]
    assert len(pair_paths) == 16
    assert all(len(pd.read_csv(path)) == 72 for path in pair_paths)
    for frame in (dense, guidance):
        midi_path = tmp_path / str(frame.iloc[0]["midi_path"])
        assert inspect_violet_midi(midi_path)["valid"] is True


def _synthetic_surrogate_features() -> pd.DataFrame:
    rows = []
    prompts = [
        ("long_low_short", "long", "low", "short"),
        ("long_high_long", "long", "high", "long"),
        ("scale_low_long", "scale", "low", "long"),
        ("scale_high_short", "scale", "high", "short"),
        ("repeat_low_short", "repeat", "low", "short"),
        ("repeat_high_long", "repeat", "high", "long"),
    ]
    technique_effect = {"sustain": 0.0, "pizzicato": -2.0}
    for prompt_index, (prompt_id, pattern, register, timing_variant) in enumerate(prompts):
        for technique, effect in technique_effect.items():
            for cc1 in (24, 64, 104):
                for replicate in (1, 2):
                    rms = -42.0 + 0.11 * cc1 + effect + 0.1 * prompt_index
                    centroid = 1300.0 + 4.0 * cc1 + 180.0 * (technique == "pizzicato")
                    rows.append(
                        {
                            "clip_id": f"{prompt_id}-{technique}-{cc1}-{replicate}",
                            "source": "model",
                            "prompt_id": prompt_id,
                            "pattern": pattern,
                            "register": register,
                            "timing_variant": timing_variant,
                            "technique": technique,
                            "cc1_final": cc1,
                            "w_tech": 1.0,
                            "w_cc": 1.0,
                            "rms_dbfs": rms,
                            "spectral_centroid_hz": (
                                np.nan if prompt_index == 0 else centroid + replicate
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def test_surrogate_trains_each_target_with_grouped_cv(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    _synthetic_surrogate_features().to_csv(features, index=False)
    outputs = train_response_surrogate(
        [features],
        tmp_path / "surrogate_multi",
        targets=["rms_dbfs", "spectral_centroid_hz"],
        n_estimators=24,
        cv_splits=3,
        n_jobs=1,
    )
    metrics = pd.read_csv(outputs["metrics"])
    predictions = pd.read_csv(outputs["predictions"])
    artifact = joblib.load(outputs["model"])

    assert set(metrics["target"]) == {"rms_dbfs", "spectral_centroid_hz"}
    assert set(artifact["models"]) == {"rms_dbfs", "spectral_centroid_hz"}
    assert metrics["prompt_groups"].min() >= 5
    assert predictions.loc[predictions["target"].eq("rms_dbfs")].shape[0] == 72
    assert predictions.loc[predictions["target"].eq("spectral_centroid_hz")].shape[0] == 60
    assert outputs["importance"].exists()


def test_surrogate_accepts_a_single_target_string(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    _synthetic_surrogate_features().to_csv(features, index=False)
    outputs = train_response_surrogate(
        [features],
        tmp_path / "surrogate_single",
        targets="rms_dbfs",
        n_estimators=12,
        cv_splits=2,
        n_jobs=1,
    )
    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["targets_trained"] == ["rms_dbfs"]
