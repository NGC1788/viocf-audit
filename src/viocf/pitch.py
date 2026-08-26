from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import librosa
import numpy as np

# 악보 최저음(G3=196.0 Hz) 아래로 확보하는 여유. 비브라토는 기준음 **아래로도** 흔들리므로
# fmin 을 최저음에 딱 붙이면 비브라토 아래쪽 절반이 통째로 버려진다.
# 실측(SwiftF0, G3 + 80 cents 비브라토): fmin=G3 이면 유효프레임 57%, 중앙값이 **+29.6 cents**
# 로 부풀고 폭은 35c 로 축소됐다. fmin=150 Hz 로 내리면 유효 100%, 중앙 +2.3c, 폭 65.8c.
# 즉 이 여유가 없으면 저음역에서 **존재하지 않는 음정 누출**이 관측된다.
# 상한은 여유를 두지 않는다. 악보 최고음은 F#6(1480 Hz)이라 C7(2093 Hz)만으로 5반음 여유가
# 있고, SwiftF0 모델 자체의 상한이 2093.75 Hz 라 그 위로는 올릴 수도 없다.
FMIN_MARGIN_SEMITONES = 4.0
SWIFTF0_MODEL_FMIN_HZ = 46.875
SWIFTF0_MODEL_FMAX_HZ = 2093.75


@dataclass(frozen=True)
class PitchTrack:
    frequency_hz: np.ndarray
    confidence: np.ndarray
    backend: str
    frame_rate_hz: float = float("nan")
    timestamps_s: np.ndarray | None = None


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
    stamps = np.asarray(getattr(result, "timestamps", []), dtype=float).reshape(-1)
    rate = float(1.0 / np.median(np.diff(stamps))) if stamps.size > 1 else float("nan")
    return PitchTrack(
        frequency_hz=frequency,
        confidence=confidence,
        backend="swiftf0-0.1.2",
        frame_rate_hz=rate,
        timestamps_s=stamps if stamps.size else None,
    )


def _pyin_track(
    samples: np.ndarray,
    sample_rate: int,
    fmin: float,
    fmax: float,
    _confidence_threshold: float,
) -> PitchTrack:
    hop_length = 256
    frequency, voiced, probability = librosa.pyin(
        samples,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=2048,
        hop_length=hop_length,
        # librosa 기본 resolution=0.1 은 후보를 0.1반음 = **10 cents 격자**에 올린다.
        # 음정 누출을 cents 로 재는 연구에서 그 계단은 그대로 계통오차가 된다.
        resolution=0.05,
    )
    valid = np.asarray(voiced, dtype=bool) & np.isfinite(frequency)
    frequency = np.where(valid, frequency, np.nan)
    confidence = np.where(valid, np.asarray(probability, dtype=float), 0.0)
    return PitchTrack(
        frequency_hz=frequency,
        confidence=confidence,
        backend="librosa-pyin",
        frame_rate_hz=float(sample_rate) / hop_length,
        timestamps_s=np.arange(frequency.size, dtype=float) * hop_length / sample_rate,
    )


def selected_backend_is_swiftf0(backend: str) -> bool:
    return backend.strip().lower().replace("-", "") in {"swiftf0", "swift"}


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
    # ⚠ 기본 범위는 악보 음역에 **여유를 두고** 잡는다 (FMIN_MARGIN_SEMITONES 주석 참조).
    selected = backend.strip().lower().replace("-", "")
    minimum = float(
        fmin if fmin is not None
        else librosa.note_to_hz("G3") * 2 ** (-FMIN_MARGIN_SEMITONES / 12.0)
    )
    maximum = float(fmax if fmax is not None else librosa.note_to_hz("C7"))
    if selected_backend_is_swiftf0(backend):
        minimum = max(minimum, SWIFTF0_MODEL_FMIN_HZ)
        maximum = min(maximum, SWIFTF0_MODEL_FMAX_HZ)
    if selected in {"swiftf0", "swift"}:
        return _swiftf0_track(samples, sample_rate, minimum, maximum, confidence_threshold)
    if selected in {"pyin", "librosapyin"}:
        return _pyin_track(samples, sample_rate, minimum, maximum, confidence_threshold)
    raise ValueError(f"Unsupported F0 backend: {backend}. Choose swiftf0 or pyin.")


# ---------------------------------------------------------------------------
# 변조(비브라토) 전용 트랙
# ---------------------------------------------------------------------------
# ⚠ SwiftF0 는 **변조 깊이 측정에 쓸 수 없다.** 실측(합성 정현파 비브라토, 2.5 s):
#
#     주입 폭      SwiftF0 측정        yin 측정
#     G3  40c        17.2c             44.2c
#     A4  40c        20.0c             40.5c
#     E5  80c        15.3c   <-- 붕괴   77.2c
#     E5 120c       134.3c   <-- 역전  115.2c
#
# SwiftF0 컨투어는 극값에서 평평하게 눌러붙고(예: A4에서 -9.6 c 로 고정), 깊이에 대해
# **비단조적**이다. 즉 "비브라토를 넓혔더니 측정값이 줄었다" 가 나올 수 있다.
# 반면 중앙값(f0_cents_error)은 SwiftF0 가 안정적이다(음마다 +2~3 c 상수 편향).
#
# 그래서 역할을 나눈다.
#   - SwiftF0 : voiced 판정 + 중앙 음정(f0_cents_error)   <- 사전고정 backend, 유지
#   - yin     : 변조(비브라토 폭/주기, f0_mod_std_cents)  <- 연속값, 격자 없음
# 두 backend 의 값을 **같은 물리량에 섞지 않는다.** 각 특징에 backend 를 함께 기록한다.

MOD_SR = 16_000          # 변조 분석용 다운샘플 (yin 은 이 정도면 충분하고 훨씬 빠르다)
MOD_FRAME = 512          # 32 ms — 더 길면 비브라토가 뭉개진다
MOD_HOP = 128            # 8 ms -> 125 Hz 프레임, 비브라토 3~9 Hz 에 충분
MOD_EDGE_FRAMES = 8      # 양 끝 경계 프레임은 버린다


def modulation_track(
    samples: np.ndarray,
    sample_rate: int,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
) -> PitchTrack:
    """비브라토/음정 변조 측정 전용 연속 F0 트랙 (yin).

    격자 양자화가 없어 **차분 정확도**가 높다. 절대 음정에는 음고마다 상수 편향이
    있으므로(측정 +2~6 cents) 절대값이 아니라 변조 성분에만 쓴다.
    """
    minimum = float(fmin if fmin is not None else librosa.note_to_hz("G3") * 2 ** (-FMIN_MARGIN_SEMITONES / 12.0))
    maximum = float(fmax if fmax is not None else librosa.note_to_hz("C7"))
    if samples.size < MOD_FRAME * 2:
        return PitchTrack(
            frequency_hz=np.array([], dtype=float),
            confidence=np.array([], dtype=float),
            backend="librosa-yin",
            frame_rate_hz=float(MOD_SR) / MOD_HOP,
        )
    resampled = (
        librosa.resample(samples.astype(np.float32, copy=False), orig_sr=sample_rate, target_sr=MOD_SR)
        if sample_rate != MOD_SR
        else samples.astype(np.float32, copy=False)
    )
    frequency = librosa.yin(
        resampled, fmin=minimum, fmax=maximum, sr=MOD_SR,
        frame_length=MOD_FRAME, hop_length=MOD_HOP,
    )
    if frequency.size > 2 * MOD_EDGE_FRAMES:
        frequency = frequency[MOD_EDGE_FRAMES:-MOD_EDGE_FRAMES]
    inside = np.isfinite(frequency) & (frequency > minimum) & (frequency < maximum)
    frequency = np.where(inside, frequency, np.nan)
    return PitchTrack(
        frequency_hz=frequency,
        confidence=inside.astype(float),
        backend="librosa-yin",
        frame_rate_hz=float(MOD_SR) / MOD_HOP,
        timestamps_s=np.arange(frequency.size, dtype=float) * MOD_HOP / MOD_SR,
    )
