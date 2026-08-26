from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from viocf.features import extract_features
from viocf.qc import qc_audio

CONFIG = {
    "sample_rate": 48000,
    "analysis": {
        "f0_backend": "pyin",
        "f0_confidence_threshold": 0.75,
        "active_threshold_db_above_noise": 12,
        "min_snr_db": 25,
        "clip_threshold": 0.999,
    },
}


def synth_violin_like(path: Path, frequency: float = 440.0, amplitude: float = 0.2) -> None:
    sample_rate = 48000
    silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    time = np.arange(int(2.0 * sample_rate), dtype=np.float32) / sample_rate
    attack = np.minimum(1.0, time / 0.08)
    tone = amplitude * attack * (
        np.sin(2 * np.pi * frequency * time)
        + 0.25 * np.sin(2 * np.pi * 2 * frequency * time)
        + 0.1 * np.sin(2 * np.pi * 3 * frequency * time)
    )
    samples = np.concatenate([silence, tone.astype(np.float32), np.zeros(int(0.3 * sample_rate))])
    sf.write(path, samples, sample_rate, subtype="PCM_24")


def test_qc_and_feature_extraction(tmp_path: Path) -> None:
    path = tmp_path / "a4.wav"
    synth_violin_like(path)
    qc = qc_audio(path, CONFIG)
    assert qc["qc_pass"] is True
    assert qc["clipped_samples"] == 0
    assert qc["snr_db"] > 25

    metadata = {
        "clip_id": "test",
        "source": "real",
        "profile": "constant",
        "single_pitch": True,
        "reference_midi": 69,
        "branch_offset_s": np.nan,
    }
    features = extract_features(path, metadata, CONFIG)
    assert abs(features["f0_cents_error"]) < 15
    assert features["f0_backend"] == "librosa-pyin"
    assert 0.02 < features["attack_time_s"] < 0.2
    assert features["spectral_centroid_hz"] > 400
    assert np.isfinite(features["hnr_db"])


def test_delayed_segment_features(tmp_path: Path) -> None:
    path = tmp_path / "branch.wav"
    synth_violin_like(path)
    metadata = {
        "clip_id": "branch",
        "source": "real",
        "profile": "delayed",
        "single_pitch": True,
        "reference_midi": 69,
        "branch_offset_s": 0.25,
    }
    features = extract_features(path, metadata, CONFIG)
    assert np.isfinite(features["prebranch_rms_dbfs"])
    assert np.isfinite(features["postbranch_rms_dbfs"])
