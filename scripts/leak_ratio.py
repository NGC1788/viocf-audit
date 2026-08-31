#!/usr/bin/env python3
"""누출비 — "피치카토가 감쇠 때문에 민감할 뿐"이라는 반론을 막는다.

## 반론

지연 실험 결과는 피치카토의 분기 전 누출(1.397)이 지속음(0.490)보다 2.85배 크다는
것이었다. 물리적으로 정확히 예측되는 방향이다 — 지속음은 t=1.0 s 에도 활이 줄에
닿아 있어 세기 변화가 정당하지만, 피치카토는 이미 줄을 놓았으므로 불가능하다.

그런데 명백한 반론이 있다. **피치카토는 감쇠가 급해서 고정 창의 RMS 가 원래 더
민감하지 않은가?** 그렇다면 큰 누출은 물리가 아니라 측정 민감도의 산물이다.

## 대조

주 요인설계 클립은 CC1 이 처음부터 끝까지 상수다. 같은 창(악보 onset+0.04 ~
onset+0.23)에서 p 와 f 가 다른 것이 **정상이다** — 조건이 애초에 다르므로.
그 크기를 분모로 쓰면 주법별 민감도가 약분된다.

    누출비 = 지연 조건의 분기 전 |f - p|  /  상수 조건의 같은 창 |f - p|

  누출비 ~ 0   ->  분기 전에는 조건 차이가 안 나타난다 = 인과적
  누출비 ~ 1   ->  분기 이후에만 적용될 조건이 **분기 전에 이미 전부 반영**돼 있다
  주법 간 비교가 민감도에 오염되지 않는다 (분자·분모가 같은 주법·같은 창)

⚠ 분모는 '정당한 차이'이지 잡음이 아니다. 그래서 이건 검정 통계량이 아니라
   **효과 크기의 해석 가능한 단위**다. 유의성은 strict 검정(0 이어야 한다)이 낸다.

사용:
  python scripts/leak_ratio.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("prebranch_abs_rms_dbfs", "prebranch_abs_centroid_hz")
# 상수 조건 대조는 단일 지속음 패턴에서 가져온다. 지연 프롬프트(4 s 단음)와
# 구조가 같아야 창 안의 내용이 비교 가능하다.
CONTROL_PATTERNS = ("long", "long_short")


def paired_delta(frame: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """같은 noise group 안에서 f - p 차이를 낸다."""
    records: list[dict] = []
    for key, group in frame.groupby(group_keys, dropna=False):
        cells = group.groupby("dynamic_label")[list(FEATURES)].mean()
        if "p" not in cells.index or "f" not in cells.index:
            continue
        row = dict(zip(group_keys, key if isinstance(key, tuple) else (key,)))
        for feature in FEATURES:
            row[feature] = abs(float(cells.loc["f", feature] - cells.loc["p", feature]))
        records.append(row)
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="expanded")
    parser.add_argument("--model-features", default=None)
    parser.add_argument("--delayed-features", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model_path = Path(args.model_features) if args.model_features else (
        root / "results" / args.profile / "model_features.csv"
    )
    delayed_path = Path(args.delayed_features) if args.delayed_features else (
        root / "results" / args.profile / "delayed_model_features.csv"
    )
    for path in (model_path, delayed_path):
        if not path.is_file():
            print(f"특징표가 없다: {path}", file=sys.stderr)
            return 2

    model = pd.read_csv(model_path)
    delayed = pd.read_csv(delayed_path)
    for frame in (model, delayed):
        for feature in FEATURES:
            if feature not in frame.columns:
                print(f"{feature} 열이 없다. 특징 추출을 다시 돌릴 것.", file=sys.stderr)
                return 2
        if "silent_absolute" in frame.columns:
            frame.drop(frame.loc[frame["silent_absolute"].astype(bool)].index, inplace=True)

    control = model.loc[model["pattern"].isin(CONTROL_PATTERNS)]
    if control.empty:
        print(f"대조용 패턴을 찾지 못했다: {CONTROL_PATTERNS}", file=sys.stderr)
        return 2

    delayed_delta = paired_delta(delayed, ["technique", "noise_group"])
    control_delta = paired_delta(control, ["technique", "prompt_id", "noise_group"])
    if delayed_delta.empty or control_delta.empty:
        print("짝을 만들지 못했다 (p 또는 f 가 없다).", file=sys.stderr)
        return 2

    print("=" * 74)
    print("누출비 — 지연 조건의 분기 전 차이 / 상수 조건의 같은 창 차이")
    print("=" * 74)
    print(f"  지연 짝 {len(delayed_delta)}개, 상수 대조 짝 {len(control_delta)}개")
    print(f"  대조 패턴: {', '.join(CONTROL_PATTERNS)}")
    print()

    rows: list[dict] = []
    for technique in sorted(set(delayed_delta["technique"]) & set(control_delta["technique"])):
        delayed_side = delayed_delta.loc[delayed_delta["technique"].eq(technique)]
        control_side = control_delta.loc[control_delta["technique"].eq(technique)]
        row: dict = {"technique": technique,
                     "n_delayed": len(delayed_side), "n_control": len(control_side)}
        for feature in FEATURES:
            numerator = float(delayed_side[feature].median())
            denominator = float(control_side[feature].median())
            row[f"delayed_{feature}"] = numerator
            row[f"control_{feature}"] = denominator
            row[f"ratio_{feature}"] = (
                numerator / denominator if denominator > 1e-12 else np.nan
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    for feature in FEATURES:
        print(f"[{feature}]  중앙값")
        view = table[["technique", f"delayed_{feature}", f"control_{feature}",
                      f"ratio_{feature}", "n_delayed", "n_control"]].copy()
        view.columns = ["주법", "지연(분기전)", "상수(대조)", "누출비", "n지연", "n대조"]
        print(view.round(3).to_string(index=False))
        print()

    print("=" * 74)
    print("판정")
    print("=" * 74)
    key = f"ratio_{FEATURES[0]}"
    if {"pizzicato", "sustain"}.issubset(set(table["technique"])):
        pizz = float(table.loc[table["technique"].eq("pizzicato"), key].iloc[0])
        sus = float(table.loc[table["technique"].eq("sustain"), key].iloc[0])
        print(f"  피치카토 누출비 {pizz:.3f}   지속음 누출비 {sus:.3f}")
        print()
        if np.isfinite(pizz) and np.isfinite(sus) and pizz > sus * 1.5:
            print("  민감도로 나눈 뒤에도 피치카토가 더 샌다.")
            print("  '감쇠 때문에 민감할 뿐'이라는 반론은 성립하지 않는다.")
            print("  줄을 이미 놓아 물리적으로 불가능한 조건에서 누출이 가장 크다.")
        elif np.isfinite(pizz) and np.isfinite(sus) and pizz < sus * 1.5:
            print("  민감도로 나누니 주법 간 차이가 사라진다.")
            print("  -> 원래의 '피치카토가 2.85배' 는 측정 민감도의 산물이었다.")
            print("     주법 간 비교는 철회하고, 전체 누출(51/51)만 주장할 것.")
        else:
            print("  분모가 0 에 가까워 비를 낼 수 없다. 원값으로 판단할 것.")
    print()
    print("  ⚠ 분모는 '정당한 차이'이지 잡음이 아니다. 이건 검정 통계량이 아니라")
    print("     효과 크기의 해석 단위다. 유의성은 strict 검정(0 이어야 한다)이 낸다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
