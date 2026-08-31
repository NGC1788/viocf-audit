from __future__ import annotations

import math
import os
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import scipy.signal

from .audio import amplitude_to_db, detect_active_region, frame_rms, read_audio, rms
from .pitch import estimate_monophonic_pitch, modulation_track

EPS = 1e-12

VIBRATO_BAND_HZ = (3.0, 9.0)      # 바이올린 비브라토 통상 범위
# 이만큼 파여야 '활바꿈/재발음' 1회로 센다.
# ⚠ 이 값은 sustain(데타셰)과 legato_slur 의 구분을 좌우하므로 임의로 정하면 안 된다.
# 민감도 실측(합성 8음 악구, 12 dB 활바꿈 노치 주입):
#     임계값 1.0 / 2.0 / 3.0 / 4.5 dB -> 두 주법이 분리됨 (slur 0.0 vs detache 5.0 dB)
#     임계값 6.0 / 9.0 dB             -> 둘 다 0.0 으로 붕괴, 구분 불가
# 유효 구간이 1.0~4.5 dB 이고 3.0 은 그 안에 있다. 다만 위쪽 여유가 1.5 dB 뿐이고
# 이건 합성음 기준이다. 12 dB 노치를 넣어도 25 ms RMS 프레임이 뭉개서 측정 prominence
# 는 5 dB 밖에 안 나온다. **실제 활바꿈은 더 얕을 수 있으므로 파일럿 녹음으로
# 반드시 재검증할 것** (scripts/run_pilot_analysis.sh 결과에서 두 주법의 분리를 확인).
DIP_PROMINENCE_DB = 3.0


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if np.isfinite(output) else default


def _segment(samples: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    start = max(0, round(start_s * sample_rate))
    end = min(len(samples), round(end_s * sample_rate))
    return samples[start:end] if end > start else np.array([], dtype=np.float32)


def _attack_time(samples: np.ndarray, sample_rate: int) -> tuple[float, float]:
    if samples.size < 16:
        return math.nan, math.nan
    envelope = np.abs(samples.astype(np.float64))
    smooth = max(1, round(0.005 * sample_rate))
    kernel = np.ones(smooth, dtype=np.float64) / smooth
    envelope = np.convolve(envelope, kernel, mode="same")
    peak_search = min(len(envelope), round(0.75 * sample_rate))
    if peak_search <= 1:
        return math.nan, math.nan
    peak_index = int(np.argmax(envelope[:peak_search]))
    peak = float(envelope[peak_index])
    if peak <= EPS:
        return math.nan, math.nan
    before = envelope[: peak_index + 1]
    idx10 = np.flatnonzero(before >= 0.1 * peak)
    idx90 = np.flatnonzero(before >= 0.9 * peak)
    if idx10.size == 0 or idx90.size == 0:
        return math.nan, math.nan
    attack = max(0.0, (int(idx90[0]) - int(idx10[0])) / sample_rate)
    slope = (0.8 * peak) / max(attack, 1.0 / sample_rate)
    return attack, float(slope)


def _decay_t20(samples: np.ndarray, sample_rate: int) -> float:
    if samples.size < int(0.1 * sample_rate):
        return math.nan
    frame = max(32, round(0.025 * sample_rate))
    hop = max(1, round(0.01 * sample_rate))
    curve = librosa.feature.rms(y=samples, frame_length=frame, hop_length=hop, center=False)[0]
    if curve.size < 3 or np.max(curve) <= EPS:
        return math.nan
    peak_idx = int(np.argmax(curve))
    db = 20.0 * np.log10(np.maximum(curve, EPS) / np.max(curve))
    after = np.flatnonzero(db[peak_idx:] <= -20.0)
    if after.size == 0:
        return math.nan
    return float(after[0] * hop / sample_rate)


def _spectral_features(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    if samples.size < 2048 or rms(samples) <= EPS:
        return {
            "spectral_centroid_hz": math.nan,
            "spectral_rolloff_hz": math.nan,
            "spectral_flatness": math.nan,
            "spectral_slope": math.nan,
            "bow_noise_ratio_db": math.nan,
        }
    n_fft = 2048
    hop = 512
    magnitude = np.abs(librosa.stft(samples, n_fft=n_fft, hop_length=hop, center=False))
    power = np.square(magnitude)
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    mean_power = np.mean(power, axis=1) + EPS
    valid = (frequencies >= 100.0) & (frequencies <= min(12000.0, sample_rate / 2.0))
    if np.count_nonzero(valid) >= 2:
        slope = np.polyfit(np.log2(frequencies[valid]), 10.0 * np.log10(mean_power[valid]), 1)[0]
    else:
        slope = math.nan
    bow_band = (frequencies >= 2000.0) & (frequencies <= min(8000.0, sample_rate / 2.0))
    total_band = (frequencies >= 80.0) & (frequencies <= min(12000.0, sample_rate / 2.0))
    bow_energy = float(np.sum(mean_power[bow_band]))
    total_energy = float(np.sum(mean_power[total_band]))
    return {
        "spectral_centroid_hz": float(np.nanmedian(centroid)),
        "spectral_rolloff_hz": float(np.nanmedian(rolloff)),
        "spectral_flatness": float(np.nanmedian(flatness)),
        "spectral_slope": float(slope),
        "bow_noise_ratio_db": float(10.0 * np.log10(max(bow_energy, EPS) / max(total_energy, EPS))),
    }


def _hnr_proxy(samples: np.ndarray) -> float:
    if samples.size < 2048 or rms(samples) <= EPS:
        return math.nan
    harmonic = librosa.effects.harmonic(samples, margin=3.0)
    residual = samples - harmonic
    return float(10.0 * np.log10((np.mean(harmonic**2) + EPS) / (np.mean(residual**2) + EPS)))


def _vibrato_features(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """비브라토 주기·폭.

    F0 는 SwiftF0 가 아니라 yin(연속값)으로 따로 뽑는다 — 이유는 pitch.py 의
    modulation_track 주석 참조(SwiftF0 는 변조 깊이에 대해 비단조적이다).
    폭은 3~9 Hz 대역만 남긴 뒤 준정현파 가정으로 peak-to-peak 로 환산한다.
    """
    blank = {
        "vibrato_rate_hz": math.nan,
        "vibrato_extent_cents": math.nan,
        "f0_mod_std_cents": math.nan,
        "f0_mod_backend": "librosa-yin",
        "f0_mod_voiced_frames": 0,
    }
    if samples.size < int(0.25 * sample_rate):
        return blank
    try:
        track = modulation_track(samples, sample_rate)
    except Exception as exc:  # noqa: BLE001 - 클립을 버리지 않고 실패를 명시한다
        return blank | {"f0_mod_backend": f"ERROR:{type(exc).__name__}"}
    values = track.frequency_hz[np.isfinite(track.frequency_hz)]
    if values.size < 32 or not np.isfinite(track.frame_rate_hz):
        return blank | {"f0_mod_voiced_frames": int(values.size)}

    cents = 1200.0 * np.log2(np.maximum(values, EPS) / np.median(values))
    cents = cents - np.mean(cents)
    fs = float(track.frame_rate_hz)
    nyquist = fs / 2.0
    high = min(VIBRATO_BAND_HZ[1], nyquist * 0.95)
    if high <= VIBRATO_BAND_HZ[0]:
        return blank | {"f0_mod_voiced_frames": int(values.size)}
    b, a = scipy.signal.butter(2, [VIBRATO_BAND_HZ[0] / nyquist, high / nyquist], btype="band")
    band = scipy.signal.filtfilt(b, a, cents)
    freqs, psd = scipy.signal.welch(band, fs=fs, nperseg=int(min(256, band.size)))
    inside = (freqs >= VIBRATO_BAND_HZ[0]) & (freqs <= high)
    rate = float(freqs[inside][np.argmax(psd[inside])]) if np.any(inside) else math.nan
    extent = float(2.0 * math.sqrt(2.0) * np.sqrt(np.mean(np.square(band))))
    return {
        "vibrato_rate_hz": rate,
        "vibrato_extent_cents": extent,
        "f0_mod_std_cents": float(np.std(cents)),
        "f0_mod_backend": track.backend,
        "f0_mod_voiced_frames": int(values.size),
    }


def _envelope_modulation(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    """활바꿈/재발음(re-articulation) 깊이.

    이게 없으면 **sustain(데타셰)과 legato_slur 를 구분할 수 없다.** 이어진 악구에서
    둘을 가르는 건 음 사이 포락선이 얼마나 파이느냐인데, active_duration_s 는
    음이 이어지면 포화돼서 두 주법이 같은 값으로 나온다.
    악보 정보 없이 RMS 포락선의 골 깊이만으로 재므로 어떤 프롬프트에도 쓸 수 있다.
    """
    blank = {"env_dip_depth_db": math.nan, "env_dip_rate_hz": math.nan}
    if samples.size < int(0.2 * sample_rate):
        return blank
    frame = max(64, round(0.025 * sample_rate))
    hop = max(1, round(0.005 * sample_rate))
    curve = librosa.feature.rms(y=samples, frame_length=frame, hop_length=hop, center=False)[0]
    if curve.size < 8 or np.max(curve) <= EPS:
        return blank
    db = 20.0 * np.log10(np.maximum(curve, EPS) / np.max(curve))
    # 골 = 뒤집은 신호의 봉우리. prominence 가 곧 '얼마나 깊이 파였나'(dB).
    valleys, props = scipy.signal.find_peaks(-db, prominence=DIP_PROMINENCE_DB)
    duration = curve.size * hop / sample_rate
    if valleys.size == 0 or duration <= 0:
        return {"env_dip_depth_db": 0.0, "env_dip_rate_hz": 0.0}
    return {
        "env_dip_depth_db": float(np.median(props["prominences"])),
        "env_dip_rate_hz": float(valleys.size / duration),
    }


def _pitch_features(
    samples: np.ndarray,
    sample_rate: int,
    reference_midi: float,
    analysis_config: dict[str, Any],
) -> dict[str, Any]:
    if samples.size < int(0.1 * sample_rate):
        return {
            "f0_median_hz": math.nan,
            "f0_std_cents": math.nan,
            "f0_cents_error": math.nan,
            "f0_cents_error_configured": math.nan,
            "f0_value_backend": "unavailable",
            "f0_backend": str(analysis_config.get("f0_backend", "swiftf0")),
            "f0_voiced_frames": 0,
        }
    try:
        track = estimate_monophonic_pitch(
            samples,
            sample_rate,
            backend=str(analysis_config.get("f0_backend", "swiftf0")),
            confidence_threshold=float(analysis_config.get("f0_confidence_threshold", 0.75)),
        )
    except Exception as exc:  # noqa: BLE001 - retain the clip while making failure explicit
        return {
            "f0_median_hz": math.nan,
            "f0_std_cents": math.nan,
            "f0_cents_error": math.nan,
            "f0_cents_error_configured": math.nan,
            "f0_value_backend": "unavailable",
            "f0_backend": f"ERROR:{type(exc).__name__}",
            "f0_voiced_frames": 0,
        }
    voiced = track.frequency_hz[np.isfinite(track.frequency_hz)]
    if voiced.size == 0:
        return {
            "f0_median_hz": math.nan,
            "f0_std_cents": math.nan,
            "f0_cents_error": math.nan,
            "f0_cents_error_configured": math.nan,
            "f0_value_backend": "unavailable",
            "f0_backend": track.backend,
            "f0_voiced_frames": 0,
        }
    median = float(np.median(voiced))
    cents_series = 1200.0 * np.log2(np.maximum(voiced, EPS) / median)
    reference_hz = float(librosa.midi_to_hz(reference_midi)) if np.isfinite(reference_midi) else math.nan
    configured_error = 1200.0 * math.log2(median / reference_hz) if np.isfinite(reference_hz) else math.nan

    # ------------------------------------------------------------------
    # ⚠ 주 음정값은 설정된 backend 가 아니라 yin 으로 잰다.
    #
    # 이 연구의 headline 지표 하나가 "강약을 바꿨더니 음정이 N cents 움직였다" 이다.
    # 즉 필요한 건 절대 음정이 아니라 **같은 음 안에서 조건 간 차이**의 정확도다.
    # 그런데 SwiftF0 는 그 차이를 사실상 못 잰다(합성 신호 실측, 2.5 s):
    #
    #     주입 차이     SwiftF0        yin
    #     G3  10c        0.0c        9.8c
    #     A4  50c       66.1c       50.1c
    #     G5  25c       -0.3c       25.7c   <-- 완전히 못 봄
    #     G5  50c       21.9c       49.5c
    #
    # SwiftF0 는 거친 음정 격자에 붙어서 10 cents 이동을 0 으로 읽는다. 이대로 두면
    # 조건 간 음정 차이가 항상 0 근처로 나와 **"음정 누출 없음"이라는 거짓 음성**이 된다.
    # yin 은 연속값이라 같은 조건에서 오차가 0.2~0.7 cents 다.
    #
    # 그래서 역할을 나눈다. SwiftF0 는 voiced 판정과 사전고정 비교값으로 남기고
    # (f0_cents_error_configured), 지표가 쓰는 f0_cents_error 는 yin 으로 낸다.
    # 두 값을 같은 열에 섞지 않으며 backend 이름을 각각 기록한다.
    # ------------------------------------------------------------------
    value_error = math.nan
    value_backend = "unavailable"
    try:
        value_track = modulation_track(samples, sample_rate)
        values = value_track.frequency_hz[np.isfinite(value_track.frequency_hz)]
        if values.size >= 16 and np.isfinite(reference_hz):
            value_error = float(1200.0 * math.log2(float(np.median(values)) / reference_hz))
            value_backend = value_track.backend
    except Exception as exc:  # noqa: BLE001
        value_backend = f"ERROR:{type(exc).__name__}"

    return {
        "f0_median_hz": median,
        "f0_std_cents": float(np.std(cents_series)),
        "f0_cents_error": float(value_error if np.isfinite(value_error) else configured_error),
        "f0_cents_error_configured": float(configured_error),
        "f0_value_backend": value_backend if np.isfinite(value_error) else track.backend,
        "f0_backend": track.backend,
        "f0_voiced_frames": int(voiced.size),
    }


# 무음 판정(절대 기준). 앞구간을 보지 않으므로 노이즈 추정 오염에 영향받지 않는다.
# 프레이밍은 audio.detect_active_region 과 같게 둔다. 개정 11·13 참조.
# 지연 실험의 분기 오프셋(design.py 의 branch_offset_s). 지연이 아닌 클립에서도
# **같은 위치의 창**을 재서 비교 분모로 쓰기 위해 여기에 둔다. 둘이 어긋나면
# 누출비가 무의미해지므로 값이 바뀌면 함께 바꿔야 한다.
REFERENCE_BRANCH_OFFSET_S = 0.25

SILENCE_FLOOR_DBFS = -60.0
SILENCE_FRAME_SECONDS = 0.046
SILENCE_HOP_SECONDS = 0.010


def extract_features(path: str | Path, metadata: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    audio = read_audio(path, mono=True)
    samples = audio.samples
    sample_rate = audio.sample_rate
    analysis_cfg = config["analysis"]
    region = detect_active_region(
        samples,
        sample_rate,
        threshold_db_above_noise=float(analysis_cfg["active_threshold_db_above_noise"]),
    )
    if region["active"]:
        active = samples[int(region["start_sample"]) : int(region["end_sample"])]
    else:
        active = np.array([], dtype=np.float32)

    # ⚠ 무음 판정을 여기서 함께 낸다. 이미 오디오를 읽었으니 공짜다.
    #
    # 왜 필요한가: 지표 계산 때 특징표만 넘어가고 QC 표는 안 넘어간다. 그래서
    # metrics 가 폐기된 peak 경계로 물러나 1,678 클립(9.0 %)을 잘못된 기준으로
    # 걸러냈다(실제로 겪음). 절대 기준을 특징표 안에 넣어두면 그 일이 없다.
    #
    # 기준: 어떤 46 ms 프레임 RMS 도 -60 dBFS 를 못 넘으면 소리가 없는 것이다.
    # 앞구간(노이즈 추정)을 보지 않으므로 오염되지 않는다 — 개정 11 참조.
    silence_frame_length = max(32, round(SILENCE_FRAME_SECONDS * sample_rate))
    silence_hop_length = max(1, round(SILENCE_HOP_SECONDS * sample_rate))
    silence_curve = frame_rms(samples, silence_frame_length, silence_hop_length)
    frame_rms_max_dbfs = (
        float(amplitude_to_db(float(silence_curve.max()))) if silence_curve.size else -math.inf
    )

    overall_rms = rms(samples)
    active_rms = rms(active)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    snr_db = float(amplitude_to_db(active_rms / max(float(region["noise_rms"]), EPS)))
    attack_time, attack_slope = _attack_time(active, sample_rate)
    onset_env = librosa.onset.onset_strength(y=samples, sr=sample_rate)
    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        units="time",
        backtrack=False,
    )
    onset_time = float(region["start_s"]) if region["active"] else math.nan

    row: dict[str, Any] = dict(metadata)
    row["frame_rms_max_dbfs"] = frame_rms_max_dbfs
    row["silent_absolute"] = bool(frame_rms_max_dbfs < SILENCE_FLOOR_DBFS)
    row.update(
        {
            "resolved_audio_path": str(Path(path).resolve()),
            "sample_rate": sample_rate,
            "channels_original": audio.channels,
            "audio_subtype": audio.subtype,
            "duration_s": len(samples) / sample_rate,
            "peak_dbfs": float(amplitude_to_db(peak)),
            "rms_dbfs": float(amplitude_to_db(active_rms)),
            "overall_rms_dbfs": float(amplitude_to_db(overall_rms)),
            "noise_dbfs": float(region["noise_dbfs"]),
            "snr_db": snr_db,
            "crest_factor_db": float(amplitude_to_db(peak / max(active_rms, EPS))),
            "onset_time_s": onset_time,
            "onset_count": len(onset_times),
            "active_duration_s": (
                float(region["end_s"] - region["start_s"]) if region["active"] else 0.0
            ),
            "attack_time_s": attack_time,
            "attack_slope": attack_slope,
            "decay_t20_s": _decay_t20(active, sample_rate),
            "hnr_db": _hnr_proxy(active),
        }
    )
    row.update(_envelope_modulation(active, sample_rate))
    row.update(_vibrato_features(active, sample_rate))
    row.update(_spectral_features(active, sample_rate))
    reference_midi = _safe_float(metadata.get("reference_midi"))
    if bool(metadata.get("single_pitch", False)) and np.isfinite(reference_midi):
        row.update(_pitch_features(active, sample_rate, reference_midi, analysis_cfg))
    else:
        row.update(
            {
                "f0_median_hz": math.nan,
                "f0_std_cents": math.nan,
                "f0_cents_error": math.nan,
                "f0_cents_error_configured": math.nan,
                "f0_value_backend": "unavailable",
                "f0_backend": str(analysis_cfg.get("f0_backend", "swiftf0")),
                "f0_voiced_frames": 0,
            }
        )

    branch_offset = _safe_float(metadata.get("branch_offset_s"))

    # ⚠ 인과성 검정용으로는 **악보상 시각**에 창을 걸어야 한다.
    #
    # 아래 detected 창은 `onset_time`(검출된 활성 구간 시작)에 걸린다. onset 검출이
    # 조건마다 흔들리면 창 자체가 움직여서, 분기 전 차이가 '진짜 다른 소리'인지
    # '다른 구간을 비교한 것'인지 구분되지 않는다. 헤드라인 주장이 걸린 지표라
    # 그 모호함을 남길 수 없다.
    #
    # note_onset_s 와 branch_offset_s 는 manifest 값이라 한 noise_group 안의
    # p/mf/f 가 **정확히 같다**. 여기에 창을 걸면 모든 조건이 같은 절대 시각을
    # 본다. 그러면 차이는 오직 오디오의 차이다.
    #
    # ⚠ 지연 클립이 아니어도 같은 창을 잰다.
    #
    # 왜: "피치카토가 가장 많이 샌다"는 주장에 명백한 반론이 있다 — 피치카토는
    # 감쇠가 급해서 고정 창의 RMS 가 원래 더 민감하지 않냐는 것이다. 그걸 막으려면
    # **같은 주법에서 '정당하게 달라도 되는' 경우의 크기**가 필요하다.
    #
    # 주 요인설계 클립은 CC1 이 처음부터 끝까지 상수라, 같은 창에서 p 와 f 가
    # 다른 게 **정상**이다(조건이 애초에 다르므로). 그 크기를 분모로 쓰면
    # 주법별 민감도가 약분된다.
    #
    #     누출비 = 지연 조건의 분기 전 차이 / 상수 조건의 같은 창 차이
    #
    # 지연 클립이 아니면 branch_offset_s 가 NaN 이므로, 창 정의에 쓸 기준 오프셋을
    # 설계값(0.25 s)으로 둔다. 창 위치는 두 경우가 정확히 같아야 비교가 된다.
    notated_onset = _safe_float(metadata.get("note_onset_s"))
    window_offset = branch_offset if np.isfinite(branch_offset) else REFERENCE_BRANCH_OFFSET_S
    if np.isfinite(notated_onset):
        notated_branch = notated_onset + window_offset
        pre_abs = _segment(
            samples, sample_rate,
            notated_onset + 0.04,
            max(notated_onset + 0.05, notated_branch - 0.02),
        )
        pre_abs_spectral = _spectral_features(pre_abs, sample_rate)
        row.update({
            "prebranch_abs_rms_dbfs": float(amplitude_to_db(rms(pre_abs))),
            "prebranch_abs_centroid_hz": pre_abs_spectral["spectral_centroid_hz"],
            "notated_branch_time_s": notated_branch,
            # 이 창이 실제 분기점을 가리키는가, 아니면 비교용 기준 창인가.
            "prebranch_window_is_branch": bool(np.isfinite(branch_offset)),
        })
    else:
        row.update({
            "prebranch_abs_rms_dbfs": math.nan,
            "prebranch_abs_centroid_hz": math.nan,
            "notated_branch_time_s": math.nan,
            "prebranch_window_is_branch": False,
        })

    if np.isfinite(branch_offset) and np.isfinite(onset_time):
        branch_time = onset_time + branch_offset
        pre = _segment(samples, sample_rate, onset_time + 0.04, max(onset_time + 0.05, branch_time - 0.02))
        post = _segment(samples, sample_rate, branch_time + 0.04, branch_time + 1.04)
        pre_spectral = _spectral_features(pre, sample_rate)
        post_spectral = _spectral_features(post, sample_rate)
        row.update(
            {
                "detected_branch_time_s": branch_time,
                "prebranch_rms_dbfs": float(amplitude_to_db(rms(pre))),
                "postbranch_rms_dbfs": float(amplitude_to_db(rms(post))),
                "prebranch_centroid_hz": pre_spectral["spectral_centroid_hz"],
                "postbranch_centroid_hz": post_spectral["spectral_centroid_hz"],
                "post_minus_pre_rms_db": float(amplitude_to_db(rms(post) / max(rms(pre), EPS))),
            }
        )
    else:
        row.update(
            {
                "detected_branch_time_s": math.nan,
                "prebranch_rms_dbfs": math.nan,
                "postbranch_rms_dbfs": math.nan,
                "prebranch_centroid_hz": math.nan,
                "postbranch_centroid_hz": math.nan,
                "post_minus_pre_rms_db": math.nan,
            }
        )
    return row


def _extract_one(job: tuple[int, dict[str, Any], str, dict[str, Any], bool]) -> tuple[int, dict[str, Any] | None]:
    """워커 하나가 클립 하나를 처리한다. 예외는 행으로 보존한다(감사 가능해야 한다)."""
    index, metadata, audio_path_text, config, include_missing = job
    audio_path = Path(audio_path_text)
    if not audio_path.exists():
        if include_missing:
            missing = dict(metadata)
            missing.update(
                {"resolved_audio_path": str(audio_path), "feature_error": "missing_audio"}
            )
            return index, missing
        return index, None
    try:
        return index, extract_features(audio_path, metadata, config)
    except Exception as exc:  # noqa: BLE001 - preserve each failed row for audit
        failed = dict(metadata)
        failed.update(
            {
                "resolved_audio_path": str(audio_path),
                "feature_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return index, failed


def _default_workers() -> int:
    # 코어를 다 쓰면 서버가 먹통이 된다. 2개는 남긴다.
    return max(1, (os.cpu_count() or 2) - 2)


def extract_manifest_features(
    manifest_path: str | Path,
    output_path: str | Path,
    project_root: str | Path,
    config: dict[str, Any],
    include_missing: bool = False,
    workers: int | None = None,
    progress_every: int = 500,
) -> pd.DataFrame:
    """manifest 의 클립에서 해석 가능한 특징을 뽑는다.

    ⚠ 병렬이다. 18,624 클립을 직렬로 돌리면 yin F0 때문에 10 시간 가까이 걸리는데,
    그동안 나머지 코어가 논다(실제로 겪음 — 26코어 서버에서 1코어만 돌았다).

    결정성은 유지된다. 특징 추출에는 난수가 없고, 결과를 manifest 순서로 다시
    정렬해서 쓴다. 워커 수를 바꿔도 출력 파일은 같다.

    각 워커 안에서는 BLAS/OpenMP 스레드를 1로 묶는다. 안 그러면 프로세스마다
    스레드를 또 띄워서 과구독으로 오히려 느려진다.
    """
    root = Path(project_root).resolve()
    manifest = pd.read_csv(manifest_path)

    jobs: list[tuple[int, dict[str, Any], str, dict[str, Any], bool]] = []
    for index, metadata in enumerate(manifest.to_dict(orient="records")):
        audio_path = Path(str(metadata["audio_path"]))
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        jobs.append((index, metadata, str(audio_path), config, include_missing))

    worker_count = workers if workers is not None else _default_workers()
    collected: dict[int, dict[str, Any]] = {}

    if worker_count <= 1 or len(jobs) <= 1:
        for job in jobs:
            index, row = _extract_one(job)
            if row is not None:
                collected[index] = row
    else:
        # 워커마다 스레드를 1개로 묶는다. 과구독 방지.
        environment = {
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
        # ⚠ spawn 방식(맥 기본, 파이썬 3.14+ 리눅스 기본)에서는 자식이 새 인터프리터로
        # 시작해 viocf 를 **다시 import** 해야 한다. editable 설치의 .pth 가 어떤
        # 이유로든 안 먹으면 자식만 ModuleNotFoundError 로 죽는다(실제로 겪음).
        # PYTHONPATH 로 직접 넘기면 시작 방식과 무관하게 확실하다.
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
                    pool.map(_extract_one, jobs, chunksize=8), start=1
                ):
                    if row is not None:
                        collected[index] = row
                    if progress_every and done % progress_every == 0:
                        print(
                            f"  특징 추출 {done:,}/{len(jobs):,} "
                            f"(워커 {worker_count}개)",
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
    return frame


def iter_numeric_features(frame: pd.DataFrame, requested: Iterable[str]) -> list[str]:
    return [name for name in requested if name in frame.columns and pd.api.types.is_numeric_dtype(frame[name])]
