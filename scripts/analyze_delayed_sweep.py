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
POST = "postbranch_abs_rms_dbfs"


def group_spreads(frame: pd.DataFrame) -> pd.DataFrame:
    """noise_group 안에서 강약 간 spread(peak-to-peak)를 낸다."""
    keys = ["w_cc", "technique", "technique_class", "branch_offset_s", "noise_group"]
    keys = [key for key in keys if key in frame.columns]
    records: list[dict] = []
    for key, group in frame.groupby(keys, dropna=False):
        available = [c for c in (FEATURE, CENTROID, POST) if c in group.columns]
        cells = group.groupby("dynamic_label")[available].mean()
        if len(cells) < 2:
            continue
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for feature in available:
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

        # ⚠ 감쇠하느냐 평평하냐는 **부차적인 질문**이다.
        # 인과적 생성기는 어떤 거리에서도 정확히 0 이다. 우리는 그 0 을 실제로
        # 확인했다 — 인과 기준 렌더러에서 960/960 그룹, 중앙 spread 0.00000000 dB
        # (개정 28). 그러니 먼 거리에서도 0 이 아니면 그 자체로 결론이 선다.
        # 감쇠 여부는 '얼마나 국소적인가'를 말할 뿐 '인과적인가'를 바꾸지 않는다.
        print()
        farthest = table.iloc[-1]
        nonzero = farthest.loc[farthest > 1e-9]
        if len(nonzero):
            print(f"  ⭐ 가장 먼 분기({table.index[-1]:.2f}s)에서도 0 이 아니다: "
                  + ", ".join(f"w_cc {c:g} → {v:.3f} dB" for c, v in nonzero.items()))
            print("     인과 기준 렌더러는 같은 파이프라인에서 정확히 0.00000000 을")
            print("     냈다(960/960). 따라서 이 값은 측정 잡음이 아니다.")
            print(f"     -> {table.index[-1]:.2f} 초 뒤에 적용될 조건이 지금 소리에")
            print("        이미 반영돼 있다.")
        else:
            print("  가장 먼 분기에서는 0 이다 — 의존이 그 거리 안에서 끝난다.")
        if near > 1e-9 and far / near < 0.5:
            print()
            print("  거리에 따라 단조 감소한다. 이건 약점이 아니라 강점이다 —")
            print("  평평한 값 하나보다 **용량-반응 곡선**이 기제의 존재를 더 잘 보인다.")

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
        # ⚠ 부류 경계가 겹치는지 반드시 확인한다.
        # released/sustained 는 '활이 줄에서 떨어지는가'로 나눈 이분법인데, 실제
        # 물리는 연속적이다. staccato 는 짧게 끊어도 **활이 줄에 닿아 있다** —
        # 튕겨서 완전히 떨어지는 pizzicato 나 튀는 spiccato 와는 다르다.
        # 부류 안 최솟값이 다른 부류 최댓값보다 작으면 그 사실을 드러내야 한다.
        flat = per_technique["median"].reset_index()
        rel = flat.loc[flat["technique_class"].eq("released"), "median"]
        sus = flat.loc[flat["technique_class"].eq("sustained"), "median"]
        if len(rel) and len(sus) and rel.min() < sus.max():
            crossing = flat.loc[
                flat["technique_class"].eq("released") & flat["median"].lt(sus.max())
            ]["technique"].tolist()
            print()
            print(f"  ⚠ 부류가 겹친다: {', '.join(crossing)} 이(가) sustained 범위 안에 있다.")
            print("     이분법이 거친 것이다 — 활이 줄에 닿는 정도는 연속적이다.")
            print("     순서 자체를 보라. 활이 줄에서 떨어지는 정도와 일치하는가?")
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
    post_key = f"spread_{POST}"
    columns = ["median", "mean", "size"]
    by_cc = spreads.groupby("w_cc")[key].agg(columns)
    if post_key in spreads.columns:
        by_cc["분기후"] = spreads.groupby("w_cc")[post_key].median()
        # ⚠ 누출 크기만 보면 안 된다.
        # w_cc=0 은 CC 조건을 아예 안 쓰는 설정이라 p/mf/f 가 분기 전이든 후든
        # 전부 같아진다. 그때의 '누출 0' 은 인과적이 된 게 아니라 **제어가 아무
        # 일도 안 해서 샐 것이 없는 것**이다. 분기 후 효과로 나눠야 둘이 갈린다.
        by_cc["누출비"] = (by_cc["median"] / by_cc["분기후"]).where(
            by_cc["분기후"] > 1e-9
        )
    print(by_cc.round(3).to_string())
    levels = sorted(spreads["w_cc"].dropna().unique())

    if post_key in spreads.columns:
        print()
        print("  ⚠ 누출 크기만으로 판단하지 않는다. w_cc=0 은 CC 조건을 쓰지 않는")
        print("     설정이라 분기 후에도 아무 차이가 없다 — 그때의 '누출 0' 은")
        print("     인과적이 된 게 아니라 제어가 작동하지 않은 것이다.")
        inert = by_cc.loc[by_cc["분기후"] <= 1e-9]
        for level in inert.index:
            print(f"     w_cc {level:g}: 분기 후 효과도 0 -> **제어 자체가 무력하다**"
                  " (누출 판정 제외)")
        active = by_cc.loc[by_cc["분기후"] > 1e-9]
        if len(active) >= 2:
            first, last = active.index[0], active.index[-1]
            low, high = float(active.loc[first, "누출비"]), float(active.loc[last, "누출비"])
            print()
            print("  제어가 실제로 작동하는 구간에서 누출비:")
            print(f"     w_cc {first:g} -> {last:g}:  {low:.3f} -> {high:.3f}")
            if high > low * 1.2:
                print("  -> 유도를 세게 줄수록 **비율로도 더 샌다.** 크기가 같이")
                print("     커진 게 아니라 과거로 새는 몫이 늘어난다.")
                print("     손잡이를 세게 돌릴수록 덜 인과적이 된다.")
            elif high < low * 0.8:
                print("  -> 유도를 세게 주면 비율이 준다. '유도가 약해서'라는 반론에")
                print("     일부 근거가 있다. 기본값 결론은 유지하되 함께 보고할 것.")
            else:
                print("  -> 비율은 거의 일정하다. 유도는 누출과 효과를 같은 비로")
                print("     키운다 — 설정이 아니라 구조의 문제다.")
        elif len(active) == 1:
            print()
            print("  ⚠ 제어가 작동하는 w_cc 가 하나뿐이라 추세를 말할 수 없다.")
    elif len(levels) >= 2:
        print()
        print("  ⚠ postbranch_abs_rms_dbfs 열이 없다. 특징 추출을 다시 돌려야")
        print("     '유도를 끄면 누출이 0' 을 올바르게 해석할 수 있다.")

    output = root / "results" / "delayed_sweep_spreads.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    spreads.to_csv(output, index=False)
    print()
    print(f"저장: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
