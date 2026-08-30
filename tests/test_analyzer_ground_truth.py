"""분석기 정확도 검증 — 정답을 아는 신호를 넣고 되찾아오는지 본다.

기존 test_audio_features.py 는 "돌아가는가 + 값이 상식 범위인가"를 본다(예:
|f0 오차| < 15 cents). 그 관용도로는 5~10 cents 계통오차나 비브라토 과소측정이
전부 통과한다. 벤치마크 연구에서 그건 치명적이다 — 분석기 오차가 곧 "모델의 결함"
으로 보고되기 때문이다.

여기서는 주입한 값을 실제로 복원하는지 검사한다. 특히 **논문 backend(SwiftF0)**를
검사한다. 기존 테스트는 pyin 으로만 돌아서 실제 쓸 backend 는 검증된 적이 없었다.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from viocf.features import extract_features

SAMPLE_RATE = 48000

CONFIG = {
    "sample_rate": SAMPLE_RATE,
    "analysis": {
        "f0_backend": "swiftf0",  # 논문에서 실제로 쓰는 backend
        "f0_confidence_threshold": 0.75,
        "active_threshold_db_above_noise": 12,
        "min_snr_db": 25,
        "clip_threshold": 0.999,
    },
}

METADATA = {
    "clip_id": "gt",
    "source": "real",
    "profile": "constant",
    "single_pitch": True,
    "reference_midi": 69,
    "branch_offset_s": math.nan,
}


def _tone(
    midi: float = 69.0,
    duration: float = 2.5,
    amplitude_db: float = -14.0,
    cents_offset: float = 0.0,
    vibrato_rate_hz: float = 0.0,
    vibrato_extent_cents: float = 0.0,
    decay_db_s: float = 0.0,
    attack_s: float = 0.06,
    brightness: float = 1.0,
) -> np.ndarray:
    frequency = 440.0 * 2 ** ((midi - 69) / 12) * 2 ** (cents_offset / 1200)
    time = np.arange(int(duration * SAMPLE_RATE)) / SAMPLE_RATE
    if vibrato_rate_hz > 0:
        cents = vibrato_extent_cents / 2 * np.sin(2 * np.pi * vibrato_rate_hz * time)
        instantaneous = frequency * 2 ** (cents / 1200)
    else:
        instantaneous = np.full_like(time, frequency)
    phase = 2 * np.pi * np.cumsum(instantaneous) / SAMPLE_RATE
    wave = np.zeros_like(time)
    for harmonic in range(1, 17):
        if frequency * harmonic > SAMPLE_RATE * 0.45:
            break
        wave += (harmonic ** (-2.0 / brightness)) * np.sin(harmonic * phase)
    wave /= np.max(np.abs(wave))
    envelope = np.clip(time / max(attack_s, 1e-4), 0, 1) * 10 ** (decay_db_s * time / 20)
    wave = wave * envelope * 10 ** (amplitude_db / 20)
    silence = np.zeros(int(0.4 * SAMPLE_RATE))
    noise = np.random.default_rng(0).normal(0, 10 ** (-70 / 20), wave.size + 2 * silence.size)
    return (np.concatenate([silence, wave, silence]) + noise).astype(np.float32)


def _features(tmp_path: Path, samples: np.ndarray, name: str = "gt.wav") -> dict:
    path = tmp_path / name
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_24")
    return extract_features(path, METADATA, CONFIG)


# ---------------------------------------------------------------- 음정
@pytest.mark.parametrize("offset", [-30.0, 0.0, 25.0])
def test_pitch_offset_is_recovered_within_5_cents(tmp_path: Path, offset: float) -> None:
    """음정 누출을 cents 로 주장하려면 분석기가 cents 를 맞혀야 한다."""
    baseline = _features(tmp_path, _tone(cents_offset=0.0), "base.wav")["f0_cents_error"]
    shifted = _features(tmp_path, _tone(cents_offset=offset), "shift.wav")["f0_cents_error"]
    assert abs((shifted - baseline) - offset) < 5.0


@pytest.mark.parametrize("midi", [55, 69, 79])
def test_pitch_error_is_stable_across_registers(tmp_path: Path, midi: int) -> None:
    """음역이 바뀌어도 편향이 튀면 '음역별 음정 누출'이 가짜로 만들어진다."""
    metadata = METADATA | {"reference_midi": midi}
    path = tmp_path / f"reg{midi}.wav"
    sf.write(path, _tone(midi=midi), SAMPLE_RATE, subtype="PCM_24")
    assert abs(extract_features(path, metadata, CONFIG)["f0_cents_error"]) < 12.0


def test_low_register_vibrato_does_not_fabricate_pitch_error(tmp_path: Path) -> None:
    """G3(=악보 최저음)에서 비브라토가 음정 오차를 만들어내면 안 된다.

    회귀 방지: fmin 을 최저음에 딱 붙여 두면 비브라토 아래쪽 절반이 버려져
    중앙값이 위로 끌려간다. 수정 전 실측 +29.6 cents (존재하지 않는 누출).
    """
    metadata = METADATA | {"reference_midi": 55}
    flat = tmp_path / "g3_flat.wav"
    wide = tmp_path / "g3_wide.wav"
    sf.write(flat, _tone(midi=55), SAMPLE_RATE, subtype="PCM_24")
    sf.write(wide, _tone(midi=55, vibrato_rate_hz=5.5, vibrato_extent_cents=80), SAMPLE_RATE, subtype="PCM_24")
    without = extract_features(flat, metadata, CONFIG)["f0_cents_error"]
    with_vibrato = extract_features(wide, metadata, CONFIG)["f0_cents_error"]
    assert abs(with_vibrato - without) < 8.0


# ---------------------------------------------------------------- 비브라토
def test_vibrato_rate_and_extent_are_recovered(tmp_path: Path) -> None:
    row = _features(tmp_path, _tone(vibrato_rate_hz=5.5, vibrato_extent_cents=80))
    assert abs(row["vibrato_rate_hz"] - 5.5) < 1.0
    assert 55.0 < row["vibrato_extent_cents"] < 105.0


def test_vibrato_extent_is_monotonic_in_depth(tmp_path: Path) -> None:
    """넓힐수록 커져야 한다. SwiftF0 로 재면 이게 깨진다(그래서 yin 을 쓴다)."""
    extents = [
        _features(tmp_path, _tone(vibrato_rate_hz=5.5, vibrato_extent_cents=depth), f"v{depth}.wav")[
            "vibrato_extent_cents"
        ]
        for depth in (0, 40, 80, 120)
    ]
    assert all(later > earlier for earlier, later in itertools.pairwise(extents))


def test_no_vibrato_reads_near_zero(tmp_path: Path) -> None:
    assert _features(tmp_path, _tone())["vibrato_extent_cents"] < 15.0


# ---------------------------------------------------------------- 다이내믹
def test_level_difference_is_recovered_and_does_not_move_pitch(tmp_path: Path) -> None:
    """+12 dB 를 주면 +12 dB 로 읽혀야 하고, 음정은 그대로여야 한다.

    회귀 방지: 어딘가에서 진폭을 정규화하면 다이내믹 측정 자체가 무의미해진다.
    """
    quiet = _features(tmp_path, _tone(amplitude_db=-26.0), "quiet.wav")
    loud = _features(tmp_path, _tone(amplitude_db=-14.0), "loud.wav")
    assert abs((loud["rms_dbfs"] - quiet["rms_dbfs"]) - 12.0) < 1.5
    assert abs(loud["f0_cents_error"] - quiet["f0_cents_error"]) < 5.0


# ---------------------------------------------------------------- 아티큘레이션
def test_decay_separates_plucked_from_bowed(tmp_path: Path) -> None:
    plucked = _features(tmp_path, _tone(decay_db_s=-18.0, attack_s=0.005), "pizz.wav")
    bowed = _features(tmp_path, _tone(decay_db_s=0.0), "arco.wav")
    assert plucked["decay_t20_s"] < bowed["decay_t20_s"] or math.isnan(bowed["decay_t20_s"])


def test_attack_time_orders_correctly(tmp_path: Path) -> None:
    fast = _features(tmp_path, _tone(attack_s=0.01), "fast.wav")["attack_time_s"]
    slow = _features(tmp_path, _tone(attack_s=0.20), "slow.wav")["attack_time_s"]
    assert fast < slow


def test_brightness_moves_centroid(tmp_path: Path) -> None:
    dark = _features(tmp_path, _tone(brightness=1.0), "dark.wav")["spectral_centroid_hz"]
    bright = _features(tmp_path, _tone(brightness=3.0), "bright.wav")["spectral_centroid_hz"]
    assert bright > dark * 1.2


# ---------------------------------------------------------------- 활바꿈 / 재발음
def _passage(dip_db: float, duration_fraction: float = 1.0, decay_db_s: float = 0.0) -> np.ndarray:
    """8음 악구. dip_db 만큼 음 경계에서 포락선이 파인다(=활바꿈 깊이)."""
    total = np.zeros(int(4.5 * SAMPLE_RATE))
    inter_onset = 0.469
    for index in range(8):
        frequency = 440.0 * 2 ** ((index % 5) / 12)
        length = inter_onset * duration_fraction
        time = np.arange(int(length * SAMPLE_RATE)) / SAMPLE_RATE
        wave = sum(
            harmonic ** -2.0 * np.sin(2 * np.pi * frequency * harmonic * time)
            for harmonic in range(1, 13)
        )
        wave = wave / np.max(np.abs(wave)) * 10 ** (decay_db_s * time / 20)
        notch = int(0.035 * SAMPLE_RATE)
        if dip_db < 0 and wave.size > notch:
            wave[:notch] *= np.linspace(10 ** (dip_db / 20), 1, notch)
        start = int((0.75 + index * inter_onset) * SAMPLE_RATE)
        total[start : start + wave.size] += wave[: max(0, total.size - start)]
    noise = np.random.default_rng(1).normal(0, 10 ** (-70 / 20), total.size)
    return (total * 0.2 + noise).astype(np.float32)


def test_envelope_dip_separates_slur_from_detache(tmp_path: Path) -> None:
    """sustain(데타셰)과 legato_slur 를 가르는 유일한 특징이다.

    active_duration_s 는 음이 이어지면 포화돼 두 주법이 같은 값으로 나온다.
    실측: legato 3.79 s vs sustain 3.79 s (구분 불가) / dip 0.0 dB vs 5.0 dB (구분됨).
    """
    metadata = METADATA | {"single_pitch": False, "reference_midi": math.nan}
    results = {}
    for name, dip in (("slur", 0.0), ("detache", -12.0)):
        path = tmp_path / f"{name}.wav"
        sf.write(path, _passage(dip), SAMPLE_RATE, subtype="PCM_24")
        results[name] = extract_features(path, metadata, CONFIG)
    assert results["detache"]["env_dip_depth_db"] > results["slur"]["env_dip_depth_db"] + 2.0


def test_envelope_dip_orders_across_all_four_techniques(tmp_path: Path) -> None:
    metadata = METADATA | {"single_pitch": False, "reference_midi": math.nan}
    depths = []
    for name, dip, fraction, decay in (
        ("slur", 0.0, 1.0, 0.0),
        ("detache", -12.0, 1.0, 0.0),
        ("staccato", -40.0, 0.5, -3.0),
    ):
        path = tmp_path / f"order_{name}.wav"
        sf.write(path, _passage(dip, fraction, decay), SAMPLE_RATE, subtype="PCM_24")
        depths.append(extract_features(path, metadata, CONFIG)["env_dip_depth_db"])
    assert depths[0] < depths[1] < depths[2]


def test_dip_threshold_has_a_valid_plateau(tmp_path: Path) -> None:
    """DIP_PROMINENCE_DB 가 임의값이 아니라 유효 구간 안에 있음을 고정한다.

    이 상수는 sustain(데타셰) vs legato_slur 구분을 좌우한다. 누가 값을 바꾸면
    두 주법이 구분 불가가 되고 technique contrast 가 통째로 무의미해지므로,
    현재 값이 분리가 유지되는 구간 안에 있는지를 회귀 테스트로 못 박는다.
    """
    from viocf import features as feature_module

    metadata = METADATA | {"single_pitch": False, "reference_midi": math.nan}
    paths = {}
    for name, dip in (("slur", 0.0), ("detache", -12.0)):
        path = tmp_path / f"thr_{name}.wav"
        sf.write(path, _passage(dip), SAMPLE_RATE, subtype="PCM_24")
        paths[name] = path

    original = feature_module.DIP_PROMINENCE_DB
    try:
        separated = []
        for threshold in (1.0, 2.0, 3.0, 4.5, 6.0):
            feature_module.DIP_PROMINENCE_DB = threshold
            depths = {
                name: extract_features(path, metadata, CONFIG)["env_dip_depth_db"]
                for name, path in paths.items()
            }
            if depths["detache"] > depths["slur"] + 2.0:
                separated.append(threshold)
        # 현재 채택값이 분리가 유지되는 구간 안에 있어야 한다.
        assert original in separated, (
            f"DIP_PROMINENCE_DB={original} 가 유효 구간 {separated} 밖이다"
        )
        # 구간이 한 점뿐이면 우연히 맞은 것이므로 근거로 쓸 수 없다.
        assert len(separated) >= 3
    finally:
        feature_module.DIP_PROMINENCE_DB = original


def test_prebranch_window_is_anchored_to_notated_time(tmp_path):
    """분기 전 구간은 **악보 시각**에 걸려야 한다.

    검출된 onset 에 창을 걸면, onset 검출이 조건마다 흔들릴 때 창 자체가 움직인다.
    그러면 '분기 전이 다르다'가 진짜 소리 차이인지 다른 구간을 비교한 탓인지
    구분되지 않는다. 이 저장소의 핵심 주장(비인과적 누출)이 그 지표에 걸려 있으므로
    그 모호함을 남길 수 없다.

    분기 전 오디오를 **비트 단위로 동일**하게 만들고 일부 조건에만 짧은 선행
    신호를 넣어 onset 검출을 흔든다. 참값은 spread = 0 이다.
    """
    import numpy as np
    import pandas as pd
    import soundfile as sf

    from viocf.features import extract_manifest_features

    rate = CONFIG["sample_rate"]
    onset, branch_offset = 0.75, 0.25
    length = int(rate * 3.0)
    rng = np.random.default_rng(2024)
    shared_noise = rng.normal(0, 1, length)  # 모든 조건이 같은 잡음을 쓴다

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    records = []
    for dynamic, post_gain, add_blip in (("p", 0.3, False), ("mf", 1.0, True), ("f", 3.0, True)):
        samples = shared_noise * 10 ** (-88 / 20)
        start, branch = int(rate * onset), int(rate * (onset + branch_offset))
        pre_times = np.arange(branch - start) / rate
        # 분기 전: 세 조건이 완전히 동일
        samples[start:branch] += np.sin(2 * np.pi * 440 * pre_times) * 10 ** (-18 / 20)
        post_times = np.arange(length - branch) / rate
        # 분기 후: 조건마다 다르다 (실제 실험과 같은 구조)
        samples[branch:] += (
            np.sin(2 * np.pi * 440 * post_times) * 10 ** (-18 / 20) * post_gain
        )
        if add_blip:
            # onset 검출을 앞으로 끌어당기는 아주 짧은 선행 신호
            blip = int(rate * 0.60)
            samples[blip : blip + int(rate * 0.01)] += 0.02
        path = audio_dir / f"{dynamic}.wav"
        sf.write(path, samples.astype(np.float32), rate)
        records.append({
            "clip_id": dynamic,
            "audio_path": str(path),
            "profile": "delayed",
            "source": "model",
            "prompt_id": "delayed_A4",
            "technique": "sustain",
            "dynamic_label": dynamic,
            "noise_group": "g0",
            "note_onset_s": onset,
            "branch_offset_s": branch_offset,
            "midi_pitch": 69,
        })
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)

    frame = extract_manifest_features(
        manifest, tmp_path / "features.csv", tmp_path, CONFIG, workers=1
    )

    # blip 이 실제로 onset 검출을 흔들었는지 먼저 확인한다.
    # 안 흔들렸다면 이 테스트는 아무것도 검증하지 못한다.
    assert frame["onset_time_s"].nunique() > 1, (
        "onset 검출이 흔들리지 않아 대조가 성립하지 않는다"
    )

    absolute_spread = float(np.ptp(frame["prebranch_abs_rms_dbfs"].dropna().to_numpy()))
    detected_spread = float(np.ptp(frame["prebranch_rms_dbfs"].dropna().to_numpy()))

    assert absolute_spread < 0.01, (
        f"분기 전이 동일한데 악보 시각 기준 spread 가 {absolute_spread:.4f} dB 다"
    )
    # 검출 기준이 가짜 누출을 만든다는 것 자체를 고정해 둔다 — 이게 이 수정의 이유다.
    assert detected_spread > 1.0, (
        "검출 onset 기준이 가짜 누출을 만들지 않는다면 이 테스트의 전제가 바뀐 것이다"
    )
