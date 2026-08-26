from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import scipy.signal

from .audio import amplitude_to_db, detect_active_region, read_audio, rms
from .pitch import estimate_monophonic_pitch, modulation_track

EPS = 1e-12

VIBRATO_BAND_HZ = (3.0, 9.0)      # 바이올린 비브라토 통상 범위
DIP_PROMINENCE_DB = 3.0           # 이만큼 파여야 '활바꿈/재발음' 1회로 센다


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


def extract_manifest_features(
    manifest_path: str | Path,
    output_path: str | Path,
    project_root: str | Path,
    config: dict[str, Any],
    include_missing: bool = False,
) -> pd.DataFrame:
    root = Path(project_root).resolve()
    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, Any]] = []
    for metadata in manifest.to_dict(orient="records"):
        audio_path = Path(str(metadata["audio_path"]))
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        if not audio_path.exists():
            if include_missing:
                missing = dict(metadata)
                missing.update({"resolved_audio_path": str(audio_path), "feature_error": "missing_audio"})
                rows.append(missing)
            continue
        try:
            rows.append(extract_features(audio_path, metadata, config))
        except Exception as exc:  # noqa: BLE001 - preserve each failed row for audit
            failed = dict(metadata)
            failed.update(
                {
                    "resolved_audio_path": str(audio_path),
                    "feature_error": f"{type(exc).__name__}: {exc}",
                }
            )
            rows.append(failed)
    frame = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def iter_numeric_features(frame: pd.DataFrame, requested: Iterable[str]) -> list[str]:
    return [name for name in requested if name in frame.columns and pd.api.types.is_numeric_dtype(frame[name])]
