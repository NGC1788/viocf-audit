#!/usr/bin/env python3
"""지연 분기 확장 결과 — 거리 곡선, 주법 부류, CC 유도 강도.

기존 지연 실험은 점 하나였다(오프셋 0.25 s, 주법 2개, w_cc 1.0). 거기서
51/51 그룹 누출과 누출비 0.618 vs 0.059 가 나왔다. 이 스크립트는 세 질문에 답한다.

## 1. 거리 곡선

인과적 생성기라면 **어떤 거리에서도** 분기 전 spread 가 0 이다. 전 구간 확산이면
거리와 무관하게 샐 수 있다.

  거리에 따라 급감      -> 국소적 의존. 짧은 지연에서만 문제.
  거리와 무관하게 평평  -> 창 전체에 조건이 박힌다. "3 초 뒤 일이 지금 소리에 있다"

⚠ 창 길이가 오프셋과 무관하게 고정돼 있어야 이 비교가 성립한다
   (features.REFERENCE_BRANCH_OFFSET_S, 개정 26). 스크립트가 확인한다.

## 2. 주법 부류

활이 줄에 남는 주법(sustain/legato_slur/tremolo)은 분기 시점의 세기 변화가
물리적으로 정당하다. 줄이 풀리는 주법(pizzicato/spiccato/staccato)은 불가능하다.
부류 안에서 반복되면 n=1 대 n=1 이 아니라 3 대 3 의 대비가 된다.

## 3. CC 유도 강도

"유도를 약하게 줘서 그런 것 아니냐"는 반론과, 그 반대 가능성 — 세게 줄수록
조건이 창 전체에 강하게 박혀 **더** 샌다 — 을 함께 본다.

사용:
  python scripts/analyze_delayed_sweep.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE = "prebranch_abs_rms_dbfs"
CENTROID = "prebranch_abs_centroid_hz"


def group_spreads(frame: pd.DataFrame) -> pd.DataFrame:
    """noise_group 안에서 강약 간 spread(peak-to-peak)를 낸다."""
    keys = ["w_cc", "technique", "technique_class", "branch_offset_s", "noise_group"]
    keys = [key for key in keys if key in frame.columns]
    records: list[dict] = []
    for key, group in frame.groupby(keys, dropna=False):
        cells = group.groupby("dynamic_label")[[FEATURE, CENTROID]].mean()
        if len(cells) < 2:
            continue
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for feature in (FEATURE, CENTROID):
            values = pd.to_numeric(cells[feature], errors="coerce").dropna().to_numpy()
            row[f"spread_{feature}"] = float(np.ptp(values)) if values.size >= 2 else np.nan
        records.append(row)
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = Path(args.features) if args.features else (
        root / "results" / "delayed_sweep" / "model_features.csv"
    )
    if not path.is_file():
        print(f"특징표가 없다: {path}\n"
              "먼저: bash scripts/run_delayed_sweep.sh 그리고 특징 추출",
              file=sys.stderr)
        return 2

    frame = pd.read_csv(path)
    if "silent_absolute" in frame.columns:
        before = len(frame)
        frame = frame.loc[~frame["silent_absolute"].astype(bool)]
        print(f"무음 {before - len(frame):,}개 제외 -> {len(frame):,} 클립")

    # 창 길이가 정말 고정돼 있는지 확인한다. 아니면 오프셋 비교가 무의미하다.
    if "prebranch_window_s" in frame.columns:
        widths = sorted(frame["prebranch_window_s"].dropna().unique())
        print(f"분기 전 창 길이: {widths}")
        if len(widths) > 1:
            print("  ⚠ 창 길이가 오프셋마다 다르다. 오프셋 간 비교를 하면 안 된다.",
                  file=sys.stderr)
            return 2

    spreads = group_spreads(frame)
    if spreads.empty:
        print("짝을 만들지 못했다.", file=sys.stderr)
        return 2
    key = f"spread_{FEATURE}"

    print()
    print("=" * 78)
    print("1. 거리 곡선 — 분기가 멀어져도 새는가 (분기 전 spread 중앙값, dB)")
    print("=" * 78)
    print("   인과적 생성기라면 모든 칸이 0.00 이어야 한다")
    table = spreads.pivot_table(
        index="branch_offset_s", columns="w_cc", values=key, aggfunc="median"
    )
    print(table.round(3).to_string())
    if len(table) >= 2:
        near, far = float(table.iloc[0].mean()), float(table.iloc[-1].mean())
        print()
        print(f"  가장 가까운 분기 {table.index[0]:.2f}s: {near:.3f} dB")
        print(f"  가장 먼 분기     {table.index[-1]:.2f}s: {far:.3f} dB")
        if near > 1e-9:
            print(f"  먼 쪽 / 가까운 쪽 = {far / near:.3f}")
            if far / near > 0.5:
                print("  -> 거리가 멀어져도 거의 줄지 않는다. 조건이 창 전체에 박힌다.")
                print("     '3 초 뒤에 일어날 일이 지금 소리에 이미 들어 있다.'")
            else:
                print("  -> 거리에 따라 줄어든다. 의존이 국소적이다.")

    print()
    print("=" * 78)
    print("2. 주법 부류 — 줄이 풀린 쪽에서 더 새는가")
    print("=" * 78)
    if "technique_class" in spreads.columns:
        by_class = spreads.pivot_table(
            index="technique_class", columns="branch_offset_s", values=key, aggfunc="median"
        )
        print(by_class.round(3).to_string())
        print()
        per_technique = spreads.groupby(["technique_class", "technique"])[key].agg(
            ["median", "size"]
        )
        print("주법별 (부류 안에서 반복되는지 — 한 주법이 끌고 가면 안 된다)")
        print(per_technique.round(3).to_string())
        if {"released", "sustained"}.issubset(set(spreads["technique_class"])):
            released = spreads.loc[spreads["technique_class"].eq("released"), key]
            sustained = spreads.loc[spreads["technique_class"].eq("sustained"), key]
            from scipy.stats import mannwhitneyu
            stat = mannwhitneyu(released.dropna(), sustained.dropna(),
                                alternative="greater")
            print()
            print(f"  released 중앙값 {released.median():.3f} dB  "
                  f"(n={released.notna().sum()})")
            print(f"  sustained 중앙값 {sustained.median():.3f} dB  "
                  f"(n={sustained.notna().sum()})")
            print(f"  Mann-Whitney U (released > sustained) p = {stat.pvalue:.2e}")

    print()
    print("=" * 78)
    print("3. CC 유도 강도 — 손잡이를 세게 돌리면 나아지나 나빠지나")
    print("=" * 78)
    by_cc = spreads.groupby("w_cc")[key].agg(["median", "mean", "size"])
    print(by_cc.round(3).to_string())
    levels = sorted(spreads["w_cc"].dropna().unique())
    if len(levels) >= 2:
        low = float(by_cc.loc[levels[0], "median"])
        high = float(by_cc.loc[levels[-1], "median"])
        print()
        print(f"  w_cc {levels[0]:g} -> {levels[-1]:g}: {low:.3f} -> {high:.3f} dB")
        if high > low * 1.2:
            print("  -> 유도를 세게 줄수록 **더 샌다.** 조건이 창 전체에 더 강하게")
            print("     박히기 때문이다. 손잡이를 세게 돌릴수록 덜 인과적이 된다.")
        elif high < low * 0.8:
            print("  -> 유도를 세게 주면 누출이 준다. '유도가 약해서'라는 반론에")
            print("     일부 근거가 있다. 기본값에서의 결론은 유지하되 함께 보고할 것.")
        else:
            print("  -> 유도 강도와 거의 무관하다. 설정이 아니라 구조의 문제다.")

    output = root / "results" / "delayed_sweep_spreads.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    spreads.to_csv(output, index=False)
    print()
    print(f"저장: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
