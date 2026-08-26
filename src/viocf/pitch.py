from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import librosa
import numpy as np


@dataclass(frozen=True)
class PitchTrack:
    frequency_hz: np.ndarray
    confidence: np.ndarray
    backend: str


@lru_cache(maxsize=8)
def _swiftf0_detector(fmin: float, fmax: float, confidence_threshold: float):
    try:
        from swift_f0 import SwiftF0
    except ImportError as exc:
        raise RuntimeError(
            "SwiftF0 is not installed. Install the project with `pip install -e '.[f0]'`."
        ) from exc
    return SwiftF0(
        fmin=fmin,
        fmax=fmax,
        confidence_threshold=confidence_threshold,
    )


def _swiftf0_track(
    samples: np.ndarray,
    sample_rate: int,
    fmin: float,
    fmax: float,
    confidence_threshold: float,
) -> PitchTrack:
    detector = _swiftf0_detector(fmin, fmax, confidence_threshold)
    result = detector.detect_from_array(samples.astype(np.float32, copy=False), sample_rate)
    frequency = np.asarray(result.pitch_hz, dtype=float).reshape(-1)
    confidence = np.asarray(result.confidence, dtype=float).reshape(-1)
    voiced = np.asarray(result.voicing, dtype=bool).reshape(-1)
    valid = voiced & np.isfinite(frequency) & (frequency >= fmin) & (frequency <= fmax)
    frequency = np.where(valid, frequency, np.nan)
    confidence = np.where(valid, confidence, 0.0)
    return PitchTrack(frequency_hz=frequency, confidence=confidence, backend="swiftf0-0.1.2")


def _pyin_track(
    samples: np.ndarray,
    sample_rate: int,
    fmin: float,
    fmax: float,
    _confidence_threshold: float,
) -> PitchTrack:
    frequency, voiced, probability = librosa.pyin(
        samples,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=2048,
        hop_length=256,
    )
    valid = np.asarray(voiced, dtype=bool) & np.isfinite(frequency)
    frequency = np.where(valid, frequency, np.nan)
    confidence = np.where(valid, np.asarray(probability, dtype=float), 0.0)
    return PitchTrack(frequency_hz=frequency, confidence=confidence, backend="librosa-pyin")


def estimate_monophonic_pitch(
    samples: np.ndarray,
    sample_rate: int,
    *,
    backend: str = "swiftf0",
    confidence_threshold: float = 0.75,
    fmin: float | None = None,
    fmax: float | None = None,
) -> PitchTrack:
    """Estimate a monophonic pitch track with an explicit, auditable backend.

    `swiftf0` is the preregistered study backend. `pyin` is kept only as a
    dependency-light smoke-test fallback; results from different backends must
    never be silently mixed in one analysis.
    """
    minimum = float(fmin if fmin is not None else librosa.note_to_hz("G3"))
    maximum = float(fmax if fmax is not None else librosa.note_to_hz("C7"))
    selected = backend.strip().lower().replace("-", "")
    if selected in {"swiftf0", "swift"}:
        return _swiftf0_track(samples, sample_rate, minimum, maximum, confidence_threshold)
    if selected in {"pyin", "librosapyin"}:
        return _pyin_track(samples, sample_rate, minimum, maximum, confidence_threshold)
    raise ValueError(f"Unsupported F0 backend: {backend}. Choose swiftf0 or pyin.")
