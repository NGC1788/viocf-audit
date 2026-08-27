"""F0 추정기 실사 벤치마크 — 이 연구가 실제로 주장하는 양으로 겨룬다.

흔한 벤치마크(RPA, RCA)는 "50 cents 안에 들어왔나"를 센다. 그 기준으로는 아래 모든
추정기가 99% 를 넘고 우열이 안 갈린다. 하지만 이 연구의 headline 은
"강약을 바꿨더니 음정이 N cents 움직였다"이므로 필요한 건 다른 것이다:

  (A) 차분 정확도  : 같은 음에서 조건만 바꿨을 때의 '차이'를 몇 cents 오차로 재나
  (B) 변조 충실도  : 비브라토 폭을 얼마나 복원하나, 그리고 깊이에 대해 단조인가
  (C) 속도         : 20만 클립을 돌릴 수 있나

(A) 가 결정적이다. 절대 음정 편향은 같은 음끼리 비교하면 상쇄되지만,
차분 오차는 그대로 '가짜 누출' 또는 '놓친 누출'이 된다.
"""

from __future__ import annotations

import itertools
import time
import warnings

import librosa
import numpy as np

warnings.filterwarnings("ignore")

SR = 48_000
DUR = 2.5
REGISTERS = ((55, "G3"), (69, "A4"), (79, "G5"))


def tone(midi: float, cents: float = 0.0, vib_rate: float = 0.0, vib_cents: float = 0.0):
    f0 = 440.0 * 2 ** ((midi - 69) / 12) * 2 ** (cents / 1200)
    t = np.arange(int(DUR * SR)) / SR
    if vib_rate > 0:
        c = vib_cents / 2 * np.sin(2 * np.pi * vib_rate * t)
        inst = f0 * 2 ** (c / 1200)
    else:
        inst = np.full_like(t, f0)
    ph = 2 * np.pi * np.cumsum(inst) / SR
    y = np.zeros_like(t)
    for k in range(1, 17):
        if f0 * k > SR * 0.45:
            break
        y += (k ** -2.0) * np.sin(k * ph)
    y = y / np.max(np.abs(y)) * 0.2 * np.clip(t / 0.06, 0, 1)
    return y.astype(np.float32)


# ------------------------------------------------------------------ 추정기들
def est_yin(y):
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    f = librosa.yin(y16, fmin=150, fmax=2100, sr=16000, frame_length=512, hop_length=128)
    return f[8:-8], 16000 / 128


def est_pyin(y):
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    f, v, _ = librosa.pyin(y16, fmin=150, fmax=2100, sr=16000,
                           frame_length=1024, hop_length=128, resolution=0.05)
    f = np.where(np.asarray(v, bool), f, np.nan)
    return f[8:-8], 16000 / 128


def est_swiftf0(y):
    from swift_f0 import SwiftF0
    d = SwiftF0(fmin=150.0, fmax=2093.0, confidence_threshold=0.75)
    r = d.detect_from_array(y, SR)
    f = np.asarray(r.pitch_hz, float)
    f = np.where(np.asarray(r.voicing, bool), f, np.nan)
    return f, 62.5


def est_pesto(y):
    import pesto
    import torch
    x = torch.from_numpy(librosa.resample(y, orig_sr=SR, target_sr=44100)).float()
    # ⚠ pesto.predict 는 (timesteps, pitch, confidence, activations) 를 반환하고
    #   convert_to_freq=True 면 pitch 가 **이미 Hz** 다. "median<200 이면 MIDI" 같은
    #   추측 변환을 넣으면 G3(196 Hz)가 MIDI 로 오인돼 값이 폭발한다(실제로 겪음).
    out = pesto.predict(x, 44100, step_size=10.0, convert_to_freq=True)
    pitch = out[1]
    p = np.asarray(pitch.detach().cpu() if hasattr(pitch, "detach") else pitch, float).reshape(-1)
    return p[3:-3], 100.0


def est_crepe(y):
    import torch
    import torchcrepe
    x = torch.from_numpy(librosa.resample(y, orig_sr=SR, target_sr=16000)).float()[None]
    f = torchcrepe.predict(x, 16000, hop_length=160, fmin=150, fmax=1900,
                           model="full", batch_size=512, device="cpu",
                           decoder=torchcrepe.decode.weighted_argmax)
    return np.asarray(f, float).reshape(-1)[3:-3], 100.0


ESTIMATORS = [
    ("yin (2002)", est_yin),
    ("pyin r=0.05 (2014)", est_pyin),
    ("SwiftF0 (2025)", est_swiftf0),
    ("CREPE full (2018)", est_crepe),
    ("PESTO (2023/25)", est_pesto),
]


def median_cents(fn, midi, cents):
    f, _ = fn(tone(midi, cents))
    v = f[np.isfinite(f) & (f > 0)]
    if v.size < 8:
        return np.nan
    ref = 440.0 * 2 ** ((midi - 69) / 12)
    return float(1200 * np.log2(np.median(v) / ref))


def vib_extent(fn, midi, depth):
    import scipy.signal
    f, fs = fn(tone(midi, 0.0, 5.5, depth))
    v = f[np.isfinite(f) & (f > 0)]
    if v.size < 32:
        return np.nan
    c = 1200 * np.log2(v / np.median(v))
    c = c - c.mean()
    nyq = fs / 2
    hi = min(9.0, nyq * 0.95)
    if hi <= 3.0:
        return np.nan
    b, a = scipy.signal.butter(2, [3.0 / nyq, hi / nyq], btype="band")
    xb = scipy.signal.filtfilt(b, a, c)
    return float(2 * np.sqrt(2) * np.sqrt(np.mean(xb ** 2)))


def main():
    print("=" * 92)
    print("(A) 차분 정확도 — 조건 간 음정 '차이'의 오차 (cents). 주입값 7/23/47c")
    print("    * 10/25/50 같은 '둥근' 값을 쓰면 격자형 추정기(pyin r=0.05 -> 5c 격자)가")
    print("      우연히 맞아떨어져 0.00c 로 보인다. 격자와 서로소인 값으로 재야 진짜가 드러난다.")
    print("=" * 92)
    header = f"{'추정기':22s}" + "".join(f"{n}{d:>3.0f}c".rjust(11) for _, n in REGISTERS for d in (7, 23, 47))
    print(header)
    results = {}
    for name, fn in ESTIMATORS:
        try:
            errs, cells = [], []
            for midi, _ in REGISTERS:
                base = median_cents(fn, midi, 0.0)
                for d in (7.0, 23.0, 47.0):
                    got = median_cents(fn, midi, d) - base
                    errs.append(abs(got - d))
                    cells.append(f"{got - d:+.1f}".rjust(11))
            results[name] = float(np.nanmean(errs))
            print(f"{name:22s}" + "".join(cells) + f"   |평균오차 {results[name]:5.2f}c")
        except Exception as e:  # noqa: BLE001
            print(f"{name:22s}  실패: {type(e).__name__}: {str(e)[:60]}")

    print()
    print("=" * 92)
    print("(B) 비브라토 폭 복원 — A4, 주입 40/80/120 cents. 단조가 아니면 사용 불가")
    print("=" * 92)
    print(f"{'추정기':22s}{'40c':>10s}{'80c':>10s}{'120c':>10s}   단조?")
    for name, fn in ESTIMATORS:
        try:
            got = [vib_extent(fn, 69, d) for d in (40, 80, 120)]
            mono = all(b > a for a, b in itertools.pairwise(got)) if all(np.isfinite(got)) else False
            print(f"{name:22s}" + "".join(f"{g:10.1f}" for g in got) + f"   {'OK' if mono else '❌비단조'}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:22s}  실패: {type(e).__name__}")

    print()
    print("=" * 92)
    print("(C) 속도 — 2.5초 클립 1개")
    print("=" * 92)
    y = tone(69, 0.0)
    for name, fn in ESTIMATORS:
        try:
            fn(y)
            t0 = time.time()
            for _ in range(3):
                fn(y)
            per = (time.time() - t0) / 3
            print(f"{name:22s}{per:8.3f} s/클립   20만 클립 1코어 {per*200000/3600:7.1f} h")
        except Exception:  # noqa: BLE001
            print(f"{name:22s}  실패")


if __name__ == "__main__":
    main()
