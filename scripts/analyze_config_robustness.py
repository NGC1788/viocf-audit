#!/usr/bin/env python3
"""설정 강건성 — "설정을 바꾸면 누출이 사라지나".

개정 31 에 약점이 둘 남았다.

  1. w_cc 누출비 0.104 -> 0.203 이 **점 두 개짜리 추세**다.
  2. sampling_steps 를 30 에 고정하고 한 번도 안 흔들었다.
     "샘플링이 부족해서 아니냐" 는 반론이 그대로 열려 있다.

이 스크립트는 두 축을 각각 곡선으로 만든다.

## 반드시 정규화한다

누출 크기만 보면 안 된다. w_cc = 0 근처에서는 제어가 약해서 분기 후 효과도
작아지고, 그러면 누출도 자동으로 작아진다. 그걸 '인과적이 됐다' 로 읽으면
정반대 결론이 나온다(개정 29 에서 실제로 그럴 뻔했다).

    누출비 = 분기 전 spread / 분기 후 효과

분기 후 효과가 0 에 가까운 설정은 '제어 무력' 로 표시하고 판정에서 뺀다.

## 판정

  누출비가 설정에 따라 크게 변한다 -> 설정 문제. 그 설정을 권고하면 된다.
  어느 설정에서도 0 이 아니다     -> **구조 문제.** 설정으로는 못 고친다.

두 번째면 "설정 어디를 만져도 안 없어진다" 가 되고, 그게 이 연구가 낼 수
있는 가장 강한 진술이다.

사용:
  python scripts/analyze_config_robustness.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE = "prebranch_abs_rms_dbfs"
POST = "postbranch_abs_rms_dbfs"
# 분기 후 효과가 이보다 작으면 제어가 사실상 작동하지 않은 것으로 본다.
INERT_POST_DB = 0.5


def group_spreads(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in ("w_cc", "sampling_steps", "technique", "technique_class",
                        "branch_offset_s", "noise_group") if k in frame.columns]
    columns = [c for c in (FEATURE, POST) if c in frame.columns]
    records: list[dict] = []
    for key, group in frame.groupby(keys, dropna=False):
        cells = group.groupby("dynamic_label")[columns].mean()
        if len(cells) < 2:
            continue
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for column in columns:
            values = pd.to_numeric(cells[column], errors="coerce").dropna().to_numpy()
            row[f"spread_{column}"] = float(np.ptp(values)) if values.size >= 2 else np.nan
        records.append(row)
    return pd.DataFrame(records)


def axis_report(spreads: pd.DataFrame, axis: str, label: str, hold: str) -> pd.Series | None:
    """한 축을 훑는다. 다른 축은 고정된 값만 남긴다."""
    key, post_key = f"spread_{FEATURE}", f"spread_{POST}"
    if axis not in spreads.columns:
        return None
    print()
    print("=" * 78)
    print(f"{label}   ({hold})")
    print("=" * 78)
    table = spreads.groupby(axis).agg(
        누출=(key, "median"), 분기후=(post_key, "median"), 그룹=(key, "size"),
    )
    table["누출비"] = (table["누출"] / table["분기후"]).where(
        table["분기후"] > INERT_POST_DB
    )
    print(table.round(4).to_string())

    inert = table.loc[table["분기후"] <= INERT_POST_DB]
    for level in inert.index:
        print(f"  ⚠ {axis} {level:g}: 분기 후 효과 "
              f"{table.loc[level, '분기후']:.3f} dB -> 제어가 사실상 무력하다."
              " 판정에서 제외.")

    active = table.loc[table["분기후"] > INERT_POST_DB, "누출비"].dropna()
    if len(active) < 2:
        print("  제어가 작동하는 설정이 2개 미만이라 추세를 말할 수 없다.")
        return active
    print()
    print(f"  제어가 작동하는 {len(active)}개 설정에서 누출비 "
          f"{active.min():.4f} ~ {active.max():.4f}")
    if float(active.min()) > 1e-9:
        print("  ⭐ **어느 설정에서도 0 이 아니다.**")
        print("     인과 기준 렌더러는 같은 파이프라인에서 정확히 0.00000000 을 냈다")
        print("     (960/960, 개정 28). 설정으로 없앨 수 있는 문제가 아니다.")
    spread_ratio = float(active.max() / max(active.min(), 1e-12))
    print(f"  최대/최소 = {spread_ratio:.2f}배")
    if spread_ratio > 2.0:
        best = active.idxmin()
        print(f"     설정에 따라 크게 달라진다. 가장 낮은 쪽은 {axis} {best:g}"
              f" ({active.min():.4f}) — 권고 설정으로 적을 수 있다.")
    else:
        print("     설정을 바꿔도 누출비가 크게 안 변한다 — 구조의 성질이다.")
    return active


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = Path(args.features) if args.features else (
        root / "results" / "config_robustness" / "model_features.csv"
    )
    if not path.is_file():
        print(f"특징표가 없다: {path}\n"
              "먼저: bash scripts/run_config_robustness.sh", file=sys.stderr)
        return 2

    frame = pd.read_csv(path)
    if POST not in frame.columns:
        print(f"{POST} 열이 없다. 특징 추출을 다시 돌릴 것.", file=sys.stderr)
        return 2
    if "silent_absolute" in frame.columns:
        before = len(frame)
        frame = frame.loc[~frame["silent_absolute"].astype(bool)]
        print(f"무음 {before - len(frame):,}개 제외 -> {len(frame):,} 클립")

    spreads = group_spreads(frame)
    if spreads.empty:
        print("짝을 만들지 못했다.", file=sys.stderr)
        return 2

    default_steps = int(frame["sampling_steps"].mode().iloc[0])
    cc_at = float(frame.loc[frame["sampling_steps"].ne(default_steps), "w_cc"].mode().iloc[0]) \
        if (frame["sampling_steps"] != default_steps).any() else float(frame["w_cc"].mode().iloc[0])

    axis_report(
        spreads.loc[spreads["sampling_steps"].eq(default_steps)],
        "w_cc", "1. CC 유도 강도 곡선", f"sampling_steps = {default_steps} 고정",
    )
    axis_report(
        spreads.loc[spreads["w_cc"].eq(cc_at)],
        "sampling_steps", "2. 샘플링 스텝 곡선", f"w_cc = {cc_at:g} 고정",
    )

    output = root / "results" / "config_robustness_spreads.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    spreads.to_csv(output, index=False)
    print()
    print(f"저장: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
