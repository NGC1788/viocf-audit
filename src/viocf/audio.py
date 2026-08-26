from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

EPS = 1e-12


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    channels: int
    subtype: str


def amplitude_to_db(value: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(value), EPS))


def power_to_db(value: float | np.ndarray) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(value), EPS))


def read_audio(path: str | Path, mono: bool = True) -> AudioData:
    source = Path(path)
    info = sf.info(source)
    samples, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    channels = int(samples.shape[1])
    if mono:
        samples = samples.mean(axis=1)
    return AudioData(
        samples=np.asarray(samples, dtype=np.float32),
        sample_rate=int(sample_rate),
        channels=channels,
        subtype=str(info.subtype),
    )


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32).copy()
    ratio = Fraction(int(target_rate), int(source_rate)).limit_denominator(1000)
    output = resample_poly(samples, ratio.numerator, ratio.denominator, axis=0)
    return np.asarray(output, dtype=np.float32)


def convert_audio_file(
    source: str | Path,
    destination: str | Path,
    target_rate: int = 48000,
    mono: bool = True,
    subtype: str = "PCM_24",
) -> Path:
    audio = read_audio(source, mono=mono)
    converted = resample_audio(audio.samples, audio.sample_rate, target_rate)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, converted, target_rate, subtype=subtype)
    return destination


def frame_signal(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return np.zeros((0, frame_length), dtype=np.float32)
    if samples.size < frame_length:
        samples = np.pad(samples, (0, frame_length - samples.size))
    count = 1 + math.ceil((samples.size - frame_length) / hop_length)
    padded_length = (count - 1) * hop_length + frame_length
    if padded_length > samples.size:
        samples = np.pad(samples, (0, padded_length - samples.size))
    shape = (count, frame_length)
    strides = (samples.strides[0] * hop_length, samples.strides[0])
    return np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides).copy()


def rms(samples: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def frame_rms(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    frames = frame_signal(samples, frame_length, hop_length)
    if frames.size == 0:
        return np.array([], dtype=np.float64)
    return np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))


def detect_active_region(
    samples: np.ndarray,
    sample_rate: int,
    noise_seconds: float = 0.35,
    threshold_db_above_noise: float = 12.0,
    frame_seconds: float = 0.046,
    hop_seconds: float = 0.01,
) -> dict[str, Any]:
    frame_length = max(32, round(frame_seconds * sample_rate))
    hop_length = max(1, round(hop_seconds * sample_rate))
    curve = frame_rms(samples, frame_length, hop_length)
    noise_samples = samples[: max(1, round(noise_seconds * sample_rate))]
    noise_rms = rms(noise_samples)
    threshold = noise_rms * (10.0 ** (threshold_db_above_noise / 20.0))
    # Guard against a perfectly silent pre-roll and against classifying low-level
    # room noise as the note.
    threshold = max(threshold, 10.0 ** (-60.0 / 20.0))
    active_frames = np.flatnonzero(curve >= threshold)
    if active_frames.size == 0:
        return {
            "active": False,
            "start_sample": 0,
            "end_sample": 0,
            "start_s": math.nan,
            "end_s": math.nan,
            "noise_rms": noise_rms,
            "noise_dbfs": float(amplitude_to_db(noise_rms)),
            "threshold": threshold,
            "rms_curve": curve,
            "hop_length": hop_length,
            "frame_length": frame_length,
        }
    first = int(active_frames[0])
    last = int(active_frames[-1])
    start_sample = first * hop_length
    end_sample = min(len(samples), last * hop_length + frame_length)
    return {
        "active": True,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "start_s": start_sample / sample_rate,
        "end_s": end_sample / sample_rate,
        "noise_rms": noise_rms,
        "noise_dbfs": float(amplitude_to_db(noise_rms)),
        "threshold": threshold,
        "rms_curve": curve,
        "hop_length": hop_length,
        "frame_length": frame_length,
    }


def segment_continuous_recording(
    source_wav: str | Path,
    start_s: float,
    end_s: float,
    destination: str | Path,
    target_rate: int = 48000,
) -> Path:
    audio = read_audio(source_wav, mono=True)
    start = max(0, round(float(start_s) * audio.sample_rate))
    end = min(len(audio.samples), round(float(end_s) * audio.sample_rate))
    if end <= start:
        raise ValueError(f"Invalid segment [{start_s}, {end_s}] for {source_wav}")
    excerpt = resample_audio(audio.samples[start:end], audio.sample_rate, target_rate)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, excerpt, target_rate, subtype="PCM_24")
    return destination
