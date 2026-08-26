from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd

from .audio import amplitude_to_db, detect_active_region, read_audio, rms
from .pitch import estimate_monophonic_pitch

EPS = 1e-12


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
            "f0_backend": f"ERROR:{type(exc).__name__}",
            "f0_voiced_frames": 0,
        }
    voiced = track.frequency_hz[np.isfinite(track.frequency_hz)]
    if voiced.size == 0:
        return {
            "f0_median_hz": math.nan,
            "f0_std_cents": math.nan,
            "f0_cents_error": math.nan,
            "f0_backend": track.backend,
            "f0_voiced_frames": 0,
        }
    median = float(np.median(voiced))
    cents_series = 1200.0 * np.log2(np.maximum(voiced, EPS) / median)
    reference_hz = float(librosa.midi_to_hz(reference_midi)) if np.isfinite(reference_midi) else math.nan
    error = 1200.0 * math.log2(median / reference_hz) if np.isfinite(reference_hz) else math.nan
    return {
        "f0_median_hz": median,
        "f0_std_cents": float(np.std(cents_series)),
        "f0_cents_error": float(error),
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
