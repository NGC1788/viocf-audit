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


def test_parallel_feature_extraction_is_deterministic(tmp_path):
    """워커 수를 바꿔도 특징표가 **비트 단위로** 같아야 한다.

    병렬화는 속도를 위해서지 결과를 바꾸려는 게 아니다. 워커 수에 따라 값이
    흔들리면 재현성이 깨지고, 그러면 이 저장소의 모든 수치를 믿을 수 없다.
    (완료 순서가 섞여도 manifest 순서로 되돌리는지를 함께 검증한다)
    """
    import numpy as np
    import pandas as pd
    import soundfile as sf

    from viocf.features import extract_manifest_features

    rate = 22050
    rng = np.random.default_rng(4242)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    records = []
    for index in range(12):
        samples = rng.normal(0, 10 ** (-85 / 20), int(rate * 2.5))
        start, end = int(rate * 0.75), int(rate * 2.2)
        times = np.arange(end - start) / rate
        frequency = 220.0 * 2 ** ((index % 7) / 12)
        samples[start:end] += np.sin(2 * np.pi * frequency * times) * 10 ** (
            (-14 - (index % 3) * 5) / 20
        )
        path = audio_dir / f"clip{index:02d}.wav"
        sf.write(path, samples.astype(np.float32), rate)
        records.append({
            "clip_id": f"clip{index:02d}",
            "audio_path": str(path),
            "prompt_id": f"p{index % 3}",
            "pattern": "long",
            "technique": ["sustain", "tremolo", "staccato"][index % 3],
            "dynamic_label": ["p", "mf", "f"][index % 3],
            "replicate": index % 2,
            "source": "model",
            "note_onset_s": 0.75,
            "midi_pitch": 57,
        })
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)

    outputs = {}
    for workers in (1, 2, 5):
        target = tmp_path / f"features_w{workers}.csv"
        extract_manifest_features(
            manifest, target, tmp_path, CONFIG, workers=workers
        )
        outputs[workers] = target.read_bytes()

    assert outputs[1] == outputs[2] == outputs[5], (
        "워커 수에 따라 특징표가 달라진다 — 재현성이 깨졌다"
    )
    frame = pd.read_csv(tmp_path / "features_w5.csv")
    assert len(frame) == 12
    assert list(frame["clip_id"]) == [f"clip{i:02d}" for i in range(12)], (
        "행 순서가 manifest 순서로 복원되지 않았다"
    )
