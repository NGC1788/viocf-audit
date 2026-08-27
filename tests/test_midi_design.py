from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from viocf.config import load_config
from viocf.design import create_design, create_smoke_design
from viocf.midi import inspect_violet_midi


def make_config(root: Path) -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    config = {
        "project_name": "test",
        "sample_rate": 48000,
        "bit_depth": 24,
        "clip_seconds": 10.0,
        "note_onset_seconds": 0.75,
        "tempo_bpm": 96,
        "techniques": {"sustain": 36, "staccato": 40, "pizzicato": 43, "legato_slur": 49},
        "dynamics": {"p": 32, "mf": 64, "f": 96},
        "model": {"base_seed": 20260826, "replicates": 5, "w_tech": 1.0, "w_cc": 1.0},
        "real": {
            "performer_id": "P1",
            "violins": ["V1", "V2", "V3"],
            "takes_per_cell": 2,
            "microphone_distance_cm": 70,
            "microphone_height_offset_cm": 15,
            "target_peak_dbfs": -12,
        },
        "analysis": {
            "baseline_technique": "sustain",
            "baseline_dynamic": "mf",
            "active_threshold_db_above_noise": 12,
            "min_snr_db": 30,
            "clip_threshold": 0.999,
            "bootstrap_iterations": 20,
            "random_seed": 20260826,
            "response_features": ["rms_dbfs", "onset_time_s"],
            "pitch_timing_features": ["onset_time_s"],
        },
    }
    path = config_dir / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_pilot_design_counts_and_shared_noise(tmp_path: Path) -> None:
    config = load_config(make_config(tmp_path))
    outputs = create_design(config, profile="pilot")
    model = pd.read_csv(outputs["model"])
    real = pd.read_csv(outputs["real"])
    delayed_model = pd.read_csv(outputs["delayed_model"])
    delayed_real = pd.read_csv(outputs["delayed_real"])
    assert len(model) == 48
    assert len(real) == 48
    assert len(delayed_model) == 12
    assert len(delayed_real) == 12
    assert model.groupby("noise_group")["seed"].nunique().max() == 1
    assert set(model["analysis_tier"]) == {"real_counterfactual_primary"}
    assert set(model["sampling_steps"]) == {30}
    assert set(model.groupby("noise_group").size()) == {12}
    assert set(delayed_model.groupby("noise_group").size()) == {6}

    first = tmp_path / model.iloc[0]["midi_path"]
    report = inspect_violet_midi(first)
    assert report["valid"] is True
    assert report["playable_note_onsets"] == report["keyswitch_onsets"]
    assert report["cc1_values"]


def test_delayed_midi_has_common_prefix_and_branch(tmp_path: Path) -> None:
    config = load_config(make_config(tmp_path))
    outputs = create_design(config, profile="pilot")
    delayed = pd.read_csv(outputs["delayed_model"])
    for row in delayed.itertuples():
        report = inspect_violet_midi(tmp_path / row.midi_path)
        values = report["cc1_values"]
        assert values[:2] == [64, 64]
        assert values[-1] == row.cc1_final


def test_smoke_design_has_two_same_noise_cells(tmp_path: Path) -> None:
    config = load_config(make_config(tmp_path))
    manifest_path = create_smoke_design(config)
    smoke = pd.read_csv(manifest_path)
    assert len(smoke) == 2
    assert set(smoke["dynamic_label"]) == {"p", "f"}
    assert smoke["noise_group"].nunique() == 1
    assert smoke["seed"].nunique() == 1
    midi_dir = tmp_path / "data" / "midi" / "smoke" / "model"
    assert sorted(path.stem for path in midi_dir.glob("*.mid")) == sorted(smoke["clip_id"])


def test_extra_model_techniques_are_expanded_only_and_labeled(tmp_path: Path) -> None:
    config_path = make_config(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["techniques"].update(
        {"spiccato": 41, "tremolo": 37, "trill_major": 38, "harmonic": 44}
    )
    raw["real"]["techniques"] = ["sustain", "staccato", "pizzicato", "legato_slur"]
    raw["model"]["replicates"] = 1
    raw["real"]["full_violin_prompts"] = 12
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)

    pilot = pd.read_csv(create_design(config, profile="pilot")["model"])
    assert set(pilot["technique"]) == set(raw["real"]["techniques"])

    expanded = pd.read_csv(create_design(config, profile="expanded")["model"])
    extra = expanded.loc[expanded["technique"].isin({"spiccato", "tremolo", "trill_major", "harmonic"})]
    assert not extra.empty
    assert set(extra["analysis_tier"]) == {"generator_only_exploratory"}
    trill = extra.loc[extra["technique"].eq("trill_major")]
    assert not trill["single_pitch"].any()
    assert trill["reference_midi"].isna().all()
