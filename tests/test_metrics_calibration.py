from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from viocf.calibration import fit_technique_calibration
from viocf.metrics import run_metric_suite


def synthetic_feature_table() -> pd.DataFrame:
    rows = []
    techniques = ["sustain", "staccato", "pizzicato", "legato_slur"]
    dynamics = {"p": (32, -6.0), "mf": (64, 0.0), "f": (96, 6.0)}
    technique_rms = {"sustain": 0.0, "staccato": -1.5, "pizzicato": -2.0, "legato_slur": 0.5}
    for source in ["real", "model"]:
        for prompt_index in range(4):
            prompt = f"p{prompt_index}"
            units = ["V1:1", "V2:1", "V3:1"] if source == "real" else ["seed1", "seed2"]
            for unit_index, unit in enumerate(units):
                for technique in techniques:
                    for dynamic_label, (cc1, real_delta) in dynamics.items():
                        dyn_delta = real_delta if source == "real" else 0.75 * real_delta
                        interaction = 0.4 if technique == "pizzicato" and dynamic_label == "f" else 0.0
                        rows.append(
                            {
                                "clip_id": f"{source}-{prompt}-{unit}-{technique}-{dynamic_label}",
                                "source": source,
                                "profile": "constant",
                                "prompt_id": prompt,
                                "technique": technique,
                                "dynamic_label": dynamic_label,
                                "cc1_final": cc1,
                                "replicate": unit_index + 1,
                                "noise_group": unit if source == "model" else np.nan,
                                "performer_id": "P1" if source == "real" else np.nan,
                                "violin_id": unit.split(":")[0] if source == "real" else np.nan,
                                "take": 1 if source == "real" else np.nan,
                                "rms_dbfs": -30 + technique_rms[technique] + dyn_delta + interaction,
                                "attack_time_s": 0.12 - 0.02 * (technique in {"staccato", "pizzicato"}),
                                "spectral_centroid_hz": 1800 + 80 * real_delta + 100 * (technique == "pizzicato"),
                                "spectral_rolloff_hz": 4200 + 120 * real_delta,
                                "hnr_db": 12 - 0.1 * real_delta,
                                "f0_cents_error": (1.0 if source == "real" else 1.5) * (cc1 - 64) / 64,
                                "onset_time_s": 0.5 + 0.005 * (technique == "staccato"),
                                "onset_count": 1,
                                "active_duration_s": 2.0 - 1.2 * (technique in {"staccato", "pizzicato"}),
                            }
                        )
    return pd.DataFrame(rows)


CONFIG = {
    "analysis": {
        "baseline_technique": "sustain",
        "baseline_dynamic": "mf",
        "bootstrap_iterations": 30,
        "random_seed": 123,
        "response_features": [
            "rms_dbfs",
            "attack_time_s",
            "spectral_centroid_hz",
            "spectral_rolloff_hz",
            "hnr_db",
            "f0_cents_error",
            "onset_time_s",
            "active_duration_s",
        ],
        "pitch_timing_features": [
            "f0_cents_error",
            "onset_time_s",
            "onset_count",
            "active_duration_s",
        ],
    }
}


def test_metric_suite_outputs(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    synthetic_feature_table().to_csv(features, index=False)
    outputs = run_metric_suite([features], tmp_path / "metrics", CONFIG)
    alignment = pd.read_csv(outputs["alignment"])
    composition = pd.read_csv(outputs["composition"])
    monotonicity = pd.read_csv(outputs["monotonicity"])
    assert not alignment.empty
    assert alignment["effect_alignment_cosine"].mean() > 0.5
    assert not composition.empty
    assert monotonicity["monotonic"].all()
    assert outputs["summary"].exists()


def test_metric_suite_excludes_generator_only_rows_from_headlines(tmp_path: Path) -> None:
    frame = synthetic_feature_table()
    frame["analysis_tier"] = "real_counterfactual_primary"
    exploratory = frame.loc[frame["source"].eq("model")].copy()
    exploratory["clip_id"] = "extra-" + exploratory["clip_id"]
    exploratory["technique"] = "trill_major"
    exploratory["analysis_tier"] = "generator_only_exploratory"
    combined = pd.concat([frame, exploratory], ignore_index=True)
    features = tmp_path / "features.csv"
    combined.to_csv(features, index=False)
    outputs = run_metric_suite([features], tmp_path / "metrics", CONFIG)
    monotonicity = pd.read_csv(outputs["monotonicity"])
    assert "trill_major" not in set(monotonicity["technique"])


def test_calibration_mapping_is_monotonic(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    synthetic_feature_table().to_csv(features, index=False)
    outputs = fit_technique_calibration([features], tmp_path / "calibration")
    mapping = pd.read_csv(outputs["mapping"])
    assert set(mapping["technique"]) == {"sustain", "staccato", "pizzicato", "legato_slur"}
    for _, group in mapping.groupby("technique"):
        assert np.all(np.diff(group.sort_values("desired_cc1")["corrected_cc1"]) >= 0)
    assert outputs["figure"].exists()
