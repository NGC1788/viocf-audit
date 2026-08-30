#!/usr/bin/env python3
"""지연 분기 실험이 무음 때문에 무너지는지 판정한다.

## 왜 필요한가

전수 재판정에서 `delayed_long x pizzicato` 무음률이 **34.4 %** 로 나왔다.
지연 분기는 이 연구의 핵심 실험이다 — 피치카토는 줄을 놓은 뒤 250 ms 지나서
갑자기 커질 수 없으므로, 분기 이전 구간이 조건에 따라 달라지면 그건 미래 조건이
과거로 샌 것이다.

그런데 `delayed_branch_strict_model_leak` 은 **noise_group 하나 안에 강약이 2개
이상 남아 있어야** 그 그룹을 쓴다. 무음 클립이 빠지면 그룹째 버려질 수 있다.
따라서 "클립 34.4 % 손실"이 곧 "그룹 34.4 % 손실"은 아니다. 실제로 몇 그룹이
남는지 세야 한다.

## 두 번째 질문: 무음이 seed 때문인가 표집 때문인가

설계상 `noise_group = "<prompt>-rep<NN>"` 이고 seed 가 거기서 나온다. 즉 한 그룹
안의 p/mf/f 는 **같은 초기 latent** 를 공유한다. 그렇다면 무음도 그룹 안에서
뭉쳐야 한다(같은 latent 가 무너지면 셋 다 무너진다).

검정 가능한 예측이다. 그룹별 무음 개수 분포를 같은 주변확률의 이항분포와 비교한다.

  뭉침(0개/3개가 과다)  ->  초기 latent 가 실패를 결정한다.
                            그룹 손실이 크지만, 남은 그룹은 온전하다.
  이항과 일치           ->  렌더마다 독립적으로 실패한다.
                            그룹은 덜 잃지만 대부분 강약이 하나씩 빈다.

어느 쪽이냐에 따라 대응이 다르다. 앞이면 seed 를 더 뽑으면 되고,
뒤면 조건마다 재시도가 필요하다.

사용:
  python scripts/delayed_branch_feasibility.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 검정에 그룹을 쓰려면 강약이 최소 몇 개 남아야 하는가.
# delayed_branch_strict_model_leak 이 요구하는 값과 같아야 한다.
MIN_DYNAMICS_PER_GROUP = 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="expanded")
    parser.add_argument("--rescore", default=None)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    rescore_path = Path(args.rescore) if args.rescore else (
        root / "results" / f"silence_rescore_{args.profile}.csv"
    )
    manifest_path = Path(args.manifest) if args.manifest else (
        root / "manifests" / f"{args.profile}_delayed_model.csv"
    )
    if not rescore_path.is_file():
        print(f"재판정 표가 없다: {rescore_path}\n"
              f"먼저: python scripts/rescore_silence.py --profile {args.profile}",
              file=sys.stderr)
        return 2
    if not manifest_path.is_file():
        print(f"지연 manifest 가 없다: {manifest_path}", file=sys.stderr)
        return 2

    rescore = pd.read_csv(rescore_path)
    manifest = pd.read_csv(manifest_path)
    # noise_group 은 manifest 에만 있다. clip_id 로 붙인다.
    keep = [c for c in ("clip_id", "noise_group", "technique", "dynamic_label", "replicate")
            if c in manifest.columns]
    data = rescore.merge(manifest[keep], on="clip_id", how="inner", suffixes=("", "_m"))
    if data.empty:
        print("재판정 표와 지연 manifest 가 clip_id 로 붙지 않는다.", file=sys.stderr)
        return 2
    if "noise_group" not in data.columns:
        print("manifest 에 noise_group 열이 없다.", file=sys.stderr)
        return 2

    print("=" * 74)
    print(f"지연 분기 실행 가능성 — {len(data):,} 클립")
    print("=" * 74)

    for technique, group in data.groupby("technique"):
        silent_rate = float(group["silent_absolute"].mean())
        groups = group.groupby("noise_group")
        total_groups = groups.ngroup().nunique()
        usable = groups.apply(
            lambda cell: int((~cell["silent_absolute"]).sum()) >= MIN_DYNAMICS_PER_GROUP,
            include_groups=False,
        )
        usable_count = int(usable.sum())
        print()
        print(f"[{technique}]")
        print(f"  클립 {len(group):,}개, 무음률 {silent_rate * 100:.1f} %")
        print(f"  noise_group {total_groups}개 중 쓸 수 있는 그룹 "
              f"(강약 {MIN_DYNAMICS_PER_GROUP}개 이상 생존): "
              f"{usable_count}  ({usable_count / max(total_groups, 1) * 100:.0f} %)")

        # 그룹별 무음 개수 분포 vs 같은 주변확률의 이항분포
        per_group = groups["silent_absolute"].sum().astype(int)
        size = int(groups.size().mode().iloc[0])
        observed = per_group.value_counts().reindex(range(size + 1), fill_value=0)
        from scipy.stats import binom
        expected = pd.Series(
            binom.pmf(range(size + 1), size, silent_rate) * total_groups,
            index=range(size + 1),
        )
        print(f"  그룹당 무음 개수 (그룹 크기 {size})")
        print(f"    {'무음수':>6} {'관측':>6} {'이항기대':>9}")
        for count in range(size + 1):
            print(f"    {count:6d} {int(observed[count]):6d} {expected[count]:9.1f}")
        if silent_rate > 0 and total_groups >= 5:
            # 뭉침이면 양 끝(0개, 전부)이 이항보다 많다.
            extremes_observed = int(observed[0] + observed[size])
            extremes_expected = float(expected[0] + expected[size])
            print(f"    양 끝(0개 또는 {size}개) 관측 {extremes_observed} "
                  f"vs 기대 {extremes_expected:.1f}")
            if extremes_observed > extremes_expected * 1.3:
                print("    -> 뭉쳐 있다. 초기 latent 가 실패를 결정한다.")
                print("       대응: seed(replicate)를 더 뽑으면 그룹이 늘어난다.")
            elif extremes_observed < extremes_expected * 0.8:
                print("    -> 이항보다 고르다. 조건별로 독립에 가깝게 실패한다.")
            else:
                print("    -> 이항과 크게 다르지 않다. 독립 실패로 보는 게 안전하다.")
                print("       대응: seed 를 늘려도 그룹당 결손은 그대로다.")

    print()
    print("=" * 74)
    print("판정")
    print("=" * 74)
    total_usable = 0
    for technique, group in data.groupby("technique"):
        groups = group.groupby("noise_group")
        usable = groups.apply(
            lambda cell: int((~cell["silent_absolute"]).sum()) >= MIN_DYNAMICS_PER_GROUP,
            include_groups=False,
        )
        total_usable += int(usable.sum())
    print(f"  쓸 수 있는 그룹 합계: {total_usable}")
    print()
    print("  이 검정의 귀무가설은 '분기 이전 구간이 조건에 관계없이 **정확히** 같다'")
    print("  이다(같은 latent + 분기 전 CC1 동일). 잡음이 있는 평균 비교가 아니라")
    print("  결정성 확인이므로, 통상적인 검정력 계산이 그대로 적용되지 않는다.")
    print("  그룹 수는 '누출 크기의 신뢰구간'을 좁히는 데 쓰인다.")
    if total_usable < 10:
        print("  ⚠ 그룹이 10개 미만이면 누출 크기 구간이 매우 넓다. seed 를 늘릴 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
