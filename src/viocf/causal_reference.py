"""인과적 기준 렌더러 — 우리 측정 파이프라인의 양성/음성 대조.

## 왜 필요한가

지연 분기 결과의 핵심 주장은 "분기 전 spread 중앙값 2.03 dB, 인과적이면 0.00"이다.
여기에 당연한 반문이 붙는다.

    **2.03 dB 가 측정 잡음이 아니라는 걸 어떻게 아는가?**

단위 테스트는 합성 파형 몇 개를 볼 뿐이다. 진짜 답은 **인과적임이 보장된 렌더러로
같은 MIDI 를 렌더해 같은 파이프라인에 통과시켜 0.00 이 나오는 것**이다.
그러면 파이프라인 전체(창 정렬, 그룹 짝짓기, 특징 추출, spread 계산)가 한꺼번에
검증된다.

## 인과성이 어떻게 보장되는가

이 렌더러는 표본 n 을 만들 때 **시각 t_n 이하의 MIDI 사건만** 참조한다.

  - CC1 은 sample-and-hold 로 시간축에 깔린다 (VIOLET 과 같은 방식)
  - 그 위에 **단측 IIR** 만 건다. 미래 표본을 보는 필터가 없다
  - 음 시작은 그 시각 이후에만 에너지를 만든다
  - 감쇠 주법은 시작 시점의 CC1 로 감쇠를 정하고, 이후 CC1 변화에 반응하지 않는다
    (튕긴 줄은 손잡이를 돌려도 다시 커지지 않는다 — 이게 물리다)

따라서 분기 이전 표본은 분기 이후 CC1 값과 **비트 단위로 무관**하다.
같은 noise group 의 p/mf/f 는 분기 전 구간이 정확히 같아야 하고,
파이프라인이 0 이 아닌 spread 를 보고하면 **파이프라인이 틀린 것이다.**

## 음성 대조도 함께 제공한다

`leak_seconds > 0` 으로 렌더하면 CC1 을 **미리 당겨** 적용한다(비인과적).
파이프라인이 그걸 잡아내지 못하면 검정력이 없다는 뜻이다. 양성 대조만으로는
"아무것도 검출 못 하는 파이프라인"과 구분되지 않는다.

⚠ 이건 바이올린 음향 모형이 아니다. 실연주 기준선(CEA/HCEL/CG)을 대신하지 않는다.
   목적은 오직 **측정 도구의 검증**이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np

# 활이 줄에 남는 주법 / 줄이 풀리는 주법. 감쇠 모형이 다르다.
RELEASED_TECHNIQUES = frozenset({"pizzicato", "spiccato", "staccato"})

# CC1(0~127) -> 진폭. 실제 활 압력처럼 배음도 함께 늘린다.
CC1_MIN_DB = -34.0
CC1_MAX_DB = -8.0
# 배음 개수는 세기에 따라 늘어난다(활 압력이 높을수록 스펙트럼이 밝다).
HARMONICS_MIN = 4
HARMONICS_MAX = 12
# 제어 신호 평활 시정수. 단측이라 미래를 보지 않는다.
CONTROL_TAU_S = 0.020
VIBRATO_HZ = 5.4
VIBRATO_CENTS = 18.0
# 내용과 무관한 고정 출력 이득. 전역 peak 정규화는 비인과라 쓸 수 없다.
OUTPUT_GAIN = 0.55


@dataclass(frozen=True)
class MidiEvents:
    notes: tuple[tuple[float, float, int, int], ...]  # (onset, duration, pitch, velocity)
    cc1: tuple[tuple[float, int], ...]                 # (time, value)
    duration_s: float


def read_events(path: str | Path) -> MidiEvents:
    """MIDI 를 (음, CC1) 로 편다. 키스위치(음높이 < 55)는 주법 신호라 제외한다."""
    midi = mido.MidiFile(path)
    tempo = mido.bpm2tempo(120)
    now = 0.0
    pending: dict[int, tuple[float, int]] = {}
    notes: list[tuple[float, float, int, int]] = []
    cc1: list[tuple[float, int]] = []
    for message in mido.merge_tracks(midi.tracks):
        now += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "control_change" and message.control == 1:
            cc1.append((now, int(message.value)))
        elif message.type == "note_on" and message.velocity > 0:
            if int(message.note) >= 55:
                pending[int(message.note)] = (now, int(message.velocity))
        elif message.type in ("note_off", "note_on"):
            pitch = int(message.note)
            if pitch in pending:
                onset, velocity = pending.pop(pitch)
                notes.append((onset, max(now - onset, 1e-3), pitch, velocity))
    return MidiEvents(tuple(sorted(notes)), tuple(sorted(cc1)), now)


def _cc1_track(events: MidiEvents, n: int, rate: int, leak_seconds: float) -> np.ndarray:
    """CC1 을 시간축에 깐다. VIOLET 과 같은 sample-and-hold.

    leak_seconds > 0 이면 **미리 당겨** 적용한다 — 의도적 비인과 음성 대조다.
    """
    track = np.full(n, 64.0, dtype=np.float64)
    if not events.cc1:
        return track
    for time_s, value in events.cc1:
        index = round((time_s - leak_seconds) * rate)
        index = max(0, min(n, index))
        track[index:] = float(value)
    return track


def _one_pole(signal: np.ndarray, tau_s: float, rate: int) -> np.ndarray:
    """단측 1차 저역통과. 표본 n 은 n 이하만 본다 — 미래를 보지 않는다."""
    alpha = math.exp(-1.0 / max(tau_s * rate, 1.0))
    out = np.empty_like(signal)
    state = float(signal[0])
    for index, value in enumerate(signal):
        state = alpha * state + (1.0 - alpha) * float(value)
        out[index] = state
    return out


def render(
    midi_path: str | Path,
    technique: str,
    sample_rate: int = 48000,
    clip_seconds: float = 10.0,
    seed: int = 0,
    leak_seconds: float = 0.0,
) -> np.ndarray:
    """MIDI 를 인과적으로 렌더한다.

    seed 는 아주 작은 바닥 잡음에만 쓴다. 같은 seed·같은 분기 전 조건이면
    분기 전 표본이 비트 단위로 같아야 한다 — 그게 이 렌더러의 계약이다.
    """
    events = read_events(midi_path)
    n = round(clip_seconds * sample_rate)
    time = np.arange(n, dtype=np.float64) / sample_rate

    cc1 = _cc1_track(events, n, sample_rate, leak_seconds)
    cc1_smooth = _one_pole(cc1, CONTROL_TAU_S, sample_rate)
    level = 10.0 ** (
        (CC1_MIN_DB + (CC1_MAX_DB - CC1_MIN_DB) * (cc1_smooth / 127.0)) / 20.0
    )

    rng = np.random.default_rng(seed)
    out = rng.normal(0.0, 10.0 ** (-86.0 / 20.0), n)
    released = technique in RELEASED_TECHNIQUES

    for onset, duration, pitch, velocity in events.notes:
        start = round(onset * sample_rate)
        if start >= n:
            continue
        stop = min(n, round((onset + duration) * sample_rate))
        if released:
            # 튕긴 줄은 시작 시점 세기로 울리고 스스로 잦아든다. 이후 CC1 변화에
            # 반응하지 않는다 — 손잡이를 돌려도 이미 놓은 줄은 다시 안 커진다.
            decay_tail = round(2.5 * sample_rate)
            stop = min(n, start + decay_tail)
            onset_cc1 = float(cc1_smooth[start])
            envelope = np.exp(-(np.arange(stop - start) / sample_rate) / 0.45)
            gain = 10.0 ** (
                (CC1_MIN_DB + (CC1_MAX_DB - CC1_MIN_DB) * (onset_cc1 / 127.0)) / 20.0
            )
            envelope = envelope * gain
            # 튕긴 줄은 시작 시점 밝기로 고정된다(물리). 상수라 인과적이다.
            brightness = np.full(stop - start, onset_cc1 / 127.0)
        else:
            # 활은 계속 닿아 있으므로 현재 CC1 을 따라간다 (여전히 인과적이다 —
            # 시각 t 의 값은 t 이하의 CC1 만 본다).
            attack = np.minimum(1.0, np.arange(stop - start) / (0.05 * sample_rate))
            release = np.minimum(
                1.0, (stop - start - np.arange(stop - start)) / (0.08 * sample_rate)
            )
            envelope = attack * release * level[start:stop]
            # ⚠ 음 전체 평균을 쓰면 안 된다. 음이 분기를 가로지르면 **분기 이후
            # CC1 이 분기 이전 음색에 섞인다.** 계약 검정이 잡아낸 두 번째 경로다.
            # 표본별 값(이미 단측 평활)을 그대로 쓴다.
            brightness = cc1_smooth[start:stop] / 127.0

        segment_time = time[start:stop] - onset
        frequency = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
        vibrato = 1.0 + (VIBRATO_CENTS / 1200.0 * math.log(2)) * np.sin(
            2 * math.pi * VIBRATO_HZ * segment_time
        )
        phase = 2 * math.pi * frequency * np.cumsum(vibrato) / sample_rate
        # 배음 수를 시간에 따라 부드럽게 연다. brightness 가 표본별 배열이므로
        # 각 배음의 가중치도 표본별이고, 전부 t 이하의 CC1 에서만 나온다.
        harmonic_span = HARMONICS_MIN + (HARMONICS_MAX - HARMONICS_MIN) * brightness
        tone = np.zeros(stop - start, dtype=np.float64)
        for partial in range(1, HARMONICS_MAX + 1):
            if frequency * partial >= sample_rate / 2:
                break
            weight = np.clip(harmonic_span - partial + 1.0, 0.0, 1.0)
            tone += np.sin(partial * phase) * weight / (partial ** 1.4)
        out[start:stop] += tone * envelope * (velocity / 127.0)

    # ⚠ 전역 peak 정규화를 쓰면 안 된다.
    #
    # 처음엔 `out *= 0.98 / max(|out|)` 를 넣었는데, 그 최대값은 **클립 전체**에서
    # 나오므로 분기 이후 내용이 분기 이전 표본의 크기를 바꾼다. 계약 검정이
    # 즉시 잡아냈다(분기 전 비트동일 = False). 우리가 VIOLET 에서 재는 것과
    # 정확히 같은 종류의 비인과 경로다.
    #
    # 대신 내용과 무관한 고정 이득을 쓴다. 넘치면 잘라내되, 자르기는 표본별
    # 연산이라 인과성을 깨지 않는다.
    return np.clip(out * OUTPUT_GAIN, -0.999, 0.999).astype(np.float32)
