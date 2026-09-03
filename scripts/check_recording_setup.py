#!/usr/bin/env python3
"""녹음 장비 사전 점검 — 48테이크 찍기 **전에** 30초로 확인한다.

## 왜 필요한가

이 연구의 주축은 강약(p/mf/f)이고, 그걸 RMS 로 잰다(CEA·HCEL·CG 전부).
그런데 **자동 게인(AGC)** 이 켜져 있으면 조용한 구간은 올리고 큰 구간은 눌러서
p 와 f 가 같은 레벨로 수렴한다. 그러면 녹음이 통째로 못 쓰게 된다.

맥북 내장 마이크, 화상회의 앱, 많은 녹음 앱이 **기본으로 켠다.**
30분 찍고 나서 알면 늦는다.

## 무엇을 재는가

세 가지를 본다. 두 번째가 결정적이다.

1. **형식** — 48 kHz 이상, 클리핑 없음
2. **바닥 노이즈 안정성** ← AGC 판별
   AGC 는 입력이 조용하면 게인을 올린다. 그래서 **조용한 구간의 바닥 노이즈가
   따라 올라간다.** 앞뒤 무음의 바닥이 크게 다르면 AGC 다.
3. **동적 범위** — 실제 바이올린의 p→f 는 15~25 dB 다. 6 dB 미만이면
   눌린 것이다(연주가 약했거나 AGC 다).

## 녹음 방법 (30초)

    1) 3초 가만히      (무음)
    2) 5초 아주 여리게  (p)   ← 같은 음 하나를 계속
    3) 5초 아주 세게    (f)
    4) 5초 다시 여리게  (p)
    5) 3초 가만히      (무음)

한 파일로 저장하고 이 스크립트에 넘긴다.

사용:
  python scripts/check_recording_setup.py 테스트녹음.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viocf.audio import amplitude_to_db, frame_rms, read_audio

TARGET_RATE = 48000
# 실제 바이올린 p→f 는 15~25 dB. 이보다 한참 작으면 눌린 것이다.
MIN_DYNAMIC_RANGE_DB = 6.0
GOOD_DYNAMIC_RANGE_DB = 12.0
# 앞뒤 무음의 바닥 노이즈 차이. AGC 는 조용할 때 게인을 올려 바닥을 들어올린다.
AGC_NOISE_DRIFT_DB = 4.0
# 연주 구간과 바닥의 최소 여유.
MIN_HEADROOM_DB = 25.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="테스트 녹음 파일")
    parser.add_argument("--frame-ms", type=float, default=50.0)
    args = parser.parse_args()

    path = Path(args.audio)
    if not path.is_file():
        print(f"파일이 없다: {path}", file=sys.stderr)
        return 2

    audio = read_audio(path, mono=True)
    samples, rate = audio.samples, audio.sample_rate
    duration = len(samples) / rate

    print("=" * 68)
    print(f"녹음 점검 — {path.name}")
    print("=" * 68)
    print(f"  길이 {duration:.1f}초 · {rate:,} Hz · 원본 {audio.channels}채널 · {audio.subtype}")

    problems: list[str] = []
    warnings: list[str] = []

    # ── 1. 형식 ────────────────────────────────────────────────────────
    if rate < TARGET_RATE:
        problems.append(
            f"표본율이 {rate:,} Hz 다. {TARGET_RATE:,} Hz 이상으로 녹음할 것 "
            "— 낮은 표본율은 되돌릴 수 없다."
        )
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    clipped = int(np.count_nonzero(np.abs(samples) >= 0.999))
    print(f"  최대 레벨 {amplitude_to_db(peak):.1f} dBFS", end="")
    if clipped:
        print(f"  ⚠ 클리핑 {clipped:,} 표본")
        problems.append(f"클리핑 {clipped:,} 표본. 입력 게인을 낮출 것.")
    elif peak > 0.89:
        print("  ⚠ 여유 부족")
        warnings.append("최대 레벨이 -1 dBFS 를 넘는다. 게인을 조금 낮추면 안전하다.")
    elif peak < 0.03:
        print("  ⚠ 너무 작음")
        problems.append("최대 레벨이 -30 dBFS 미만이다. 마이크를 가까이 하거나 게인을 올릴 것.")
    else:
        print("  ✓")

    if duration < 15:
        problems.append(f"길이가 {duration:.0f}초다. 안내대로 약 20~30초를 녹음할 것.")
        print()
        for item in problems:
            print(f"  ✗ {item}")
        return 1

    # ── 2. 바닥 노이즈 안정성 (AGC 판별) ───────────────────────────────
    frame_length = max(64, int(rate * args.frame_ms / 1000))
    curve = frame_rms(samples, frame_length, frame_length // 2)
    curve_db = np.array([amplitude_to_db(max(v, 1e-9)) for v in curve])
    n = len(curve_db)
    head = curve_db[: max(1, int(n * 0.08))]
    tail = curve_db[-max(1, int(n * 0.08)) :]
    # 각 구간의 하위 25 % 를 바닥으로 본다(연주가 조금 물려도 견딘다).
    floor_head = float(np.quantile(head, 0.25))
    floor_tail = float(np.quantile(tail, 0.25))
    drift = abs(floor_head - floor_tail)

    print()
    print("  바닥 노이즈 (앞뒤 무음)")
    print(f"    시작 {floor_head:7.1f} dBFS")
    print(f"    끝   {floor_tail:7.1f} dBFS")
    print(f"    차이 {drift:7.1f} dB", end="")
    if drift > AGC_NOISE_DRIFT_DB:
        print("  ✗")
        problems.append(
            f"앞뒤 바닥 노이즈가 {drift:.1f} dB 어긋난다. **자동 게인(AGC)이 켜져 있다.**\n"
            "     AGC 는 조용하면 게인을 올리므로 p 와 f 가 같은 레벨로 눌린다.\n"
            "     이 연구의 주축이 강약이라 이대로 찍으면 전부 못 쓴다.\n"
            "     끄는 법: 시스템 설정 → 사운드 → 입력에서 '주변 소음 감소' 끄기,\n"
            "     녹음 앱의 AGC/자동레벨/노이즈억제/Voice Isolation 전부 끄기.\n"
            "     내장 마이크는 끌 수 없는 경우가 많다 — 외장 마이크나\n"
            "     오디오 인터페이스를 쓰는 게 확실하다."
        )
    else:
        print("  ✓  게인이 고정돼 있다")

    # ── 3. 동적 범위 ───────────────────────────────────────────────────
    loud = float(np.quantile(curve_db, 0.97))
    quiet_band = curve_db[curve_db > max(floor_head, floor_tail) + 8]
    quiet = float(np.quantile(quiet_band, 0.10)) if quiet_band.size else float("nan")
    dynamic_range = loud - quiet if np.isfinite(quiet) else float("nan")

    print()
    print("  동적 범위 (연주 구간)")
    print(f"    센 쪽   {loud:7.1f} dBFS")
    print(f"    여린 쪽 {quiet:7.1f} dBFS")
    print(f"    차이    {dynamic_range:7.1f} dB", end="")
    if not np.isfinite(dynamic_range):
        print("  ⚠")
        warnings.append("연주 구간을 못 찾았다. 무음-여리게-세게 순서대로 녹음했는지 확인할 것.")
    elif dynamic_range < MIN_DYNAMIC_RANGE_DB:
        print("  ✗")
        problems.append(
            f"p 와 f 차이가 {dynamic_range:.1f} dB 뿐이다. 실제 바이올린은 15~25 dB 다.\n"
            "     AGC 가 눌렀거나, 강약 차이를 충분히 크게 연주하지 않았다."
        )
    elif dynamic_range < GOOD_DYNAMIC_RANGE_DB:
        print("  ⚠")
        warnings.append(
            f"p→f 가 {dynamic_range:.1f} dB 다. 쓸 수는 있지만 강약을 더 벌리면 "
            "효과 크기가 뚜렷해진다."
        )
    else:
        print("  ✓")

    headroom = loud - max(floor_head, floor_tail)
    print(f"    바닥 대비 여유 {headroom:.1f} dB", end="")
    if headroom < MIN_HEADROOM_DB:
        print("  ⚠")
        warnings.append(
            f"연주와 바닥의 차이가 {headroom:.1f} dB 다. 조용한 방에서, 마이크를 "
            "70 cm 거리에 두고 다시 시도할 것."
        )
    else:
        print("  ✓")

    # ── 판정 ───────────────────────────────────────────────────────────
    print()
    print("=" * 68)
    if problems:
        print("판정: 이대로 녹음하면 안 된다")
        print("=" * 68)
        for item in problems:
            print(f"  ✗ {item}")
    else:
        print("판정: 녹음해도 좋다")
        print("=" * 68)
        print("  ✓ 게인 고정 · 동적 범위 확보 · 형식 적합")
    for item in warnings:
        print(f"  ⚠ {item}")
    print()
    print("  ※ 48 테이크 내내 **마이크와 연주자 위치를 바꾸지 말 것.**")
    print("     레벨 비교가 전부 무너진다. 설정한 뒤 게인 노브도 건드리지 않는다.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
