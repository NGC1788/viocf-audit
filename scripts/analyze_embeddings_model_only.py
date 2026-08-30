#!/usr/bin/env python3
"""녹음 없이, 모델 임베딩만으로 답할 수 있는 것을 잰다.

## 왜 필요한가

`embedding_contrast_c2st` 는 모델과 **실연주**를 비교하므로 녹음이 들어오기 전에는
`techniques: 0` 만 나온다. 그런데 MERT 임베딩 추출에는 이미 GPU 시간이 들어갔다.
그 표현으로 지금 당장 답할 수 있는 질문이 있다.

## 세 가지 검정

### A. 강약 제어가 음색을 바꾸는가, 아니면 볼륨 손잡이인가

진짜 바이올린은 세게 켜면 활 압력이 올라가 **배음 구조 자체가 바뀐다.** 단순히
커지는 게 아니다. 그래서 "강약을 바꿨을 때 음색이 함께 바뀌는가" 는 제어가
물리적으로 그럴듯한지 가르는 질문이다.

⚠⚠ 전제 정정 (실행 결과를 보고 발견한 내 오류)

  처음 이 스크립트는 "MERT 는 음량에 눈이 멀었다(A4 를 -12 dB 낮춰도 코사인
  1.0000)"를 근거로, C2ST 가 p 와 f 를 구별하지 못하면 음색 변화가 없는 것이라고
  했다. **그 근거는 여기에 쓸 수 없다.**

  코사인 유사도는 **벡터의 크기를 무시한다.** 음량이 임베딩의 노름만 키우고
  방향은 그대로 두면 코사인은 1.0000 이 나오지만, 원좌표를 쓰는 분류기는
  그 크기 차이만으로 두 집단을 완벽히 갈라낸다. 즉 C2ST 가 높다고 해서
  음색이 바뀌었다는 뜻이 되지 않는다.

  그래서 대조를 하나 더 건다.

    원좌표 C2ST 높음 + 정규화 C2ST 도 높음  ->  방향이 바뀌었다 = 진짜 음색 변화
    원좌표 C2ST 높음 + 정규화 C2ST ~ 0.5    ->  크기만 바뀌었다 = 사실상 볼륨 손잡이

  노름 자체도 강약별로 직접 출력한다. 노름이 p < mf < f 로 단조 증가하면
  임베딩 크기가 음량을 따라간다는 직접 증거다.

⚠ 양성 대조가 없으면 이 음성 결과는 반증 불가능하다("임베딩이 고장났을 뿐"과
   구분이 안 된다). 그래서 주법 대조를 함께 잰다. 피치카토와 지속음은 사람 귀에
   명백히 다른 음색이므로 C2ST 가 1.0 에 가까워야 한다. 그게 안 나오면 임베딩
   파이프라인을 의심해야 하고, A 의 결과는 해석하지 않는다.

⚠ 두 번째 대조: 특징표의 rms_dbfs 로 **음량은 실제로 바뀌었는지** 확인한다.
   음량마저 안 바뀌었다면 "볼륨 손잡이"가 아니라 "제어가 아무것도 안 했다"는
   전혀 다른(더 큰) 결론이다. 둘을 섞으면 안 된다.

### B. 제어축이 서로 직교하는가 (누출의 기하)

강약만 바꿨을 때의 이동 방향과 주법만 바꿨을 때의 이동 방향이 같은 쪽을 향하면,
두 제어가 표현 공간에서 얽혀 있다는 뜻이다.

  |cos| ~ 0  ->  독립적인 축
  |cos| 큼   ->  한 손잡이를 돌리면 다른 쪽도 따라 움직인다

### C. 합성성 간극 (모델 전용판)

강약만 바꾼 이동 + 주법만 바꾼 이동 == 둘 다 바꾼 이동 인가.
어긋난 크기를 상대값으로 낸다. 실연주 기준선이 필요한 CG 의 모델 전용 대응물이다.

사용:
  python scripts/analyze_embeddings_model_only.py
  python scripts/analyze_embeddings_model_only.py --embeddings results/embeddings/expanded_model_mert.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations, pairwise, permutations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viocf.embeddings import embedding_columns
from viocf.metrics import classifier_two_sample_test

SEED = 20260831
# 이 아래로는 C2ST 가 불안정하다. 조용히 답을 내는 것보다 건너뛰는 게 낫다.
MIN_PER_SIDE = 20
# C2ST 는 특징수 x 표본수에 비례해 느려진다. 3072 차원 x 18,000 행이면 100 분쯤 든다.
# 정확도 추정에 그만큼이 필요하지는 않다 — 한쪽 1,500 개면 표준오차가 약 0.9 %p 다.
# ⚠ 잘라낸 사실은 반드시 출력한다. 말없이 자르면 '전수를 봤다'로 읽힌다.
MAX_PER_SIDE = 1500


def cell_mean(frame: pd.DataFrame, columns: list[str]) -> np.ndarray | None:
    if frame.empty:
        return None
    return frame[columns].to_numpy(dtype=float).mean(axis=0)


def c2st_between(
    frame: pd.DataFrame,
    columns: list[str],
    column: str,
    left: str,
    right: str,
    max_per_side: int | None = None,
) -> dict | None:
    a = frame.loc[frame[column].eq(left), columns]
    b = frame.loc[frame[column].eq(right), columns]
    if len(a) < MIN_PER_SIDE or len(b) < MIN_PER_SIDE:
        return None
    full_left, full_right = len(a), len(b)
    cap = max_per_side or MAX_PER_SIDE
    rng = np.random.default_rng(SEED)
    if len(a) > cap:
        a = a.iloc[rng.choice(len(a), cap, replace=False)]
    if len(b) > cap:
        b = b.iloc[rng.choice(len(b), cap, replace=False)]
    if full_left > len(a) or full_right > len(b):
        print(f"  [표본 상한] {column} {left} vs {right}: "
              f"{full_left}/{full_right} -> {len(a)}/{len(b)} 로 줄여 계산")
    result = classifier_two_sample_test(
        a.to_numpy(dtype=float), b.to_numpy(dtype=float), seed=SEED
    )
    return {
        "contrast": f"{column}: {left} vs {right}",
        "n_available_left": full_left,
        "n_available_right": full_right,
        "accuracy": float(result["accuracy"]),
        "p_value": float(result["p_value"]),
        "n": int(result["n"]),
        "n_left": len(a),
        "n_right": len(b),
    }


def magnitude_versus_direction(frame: pd.DataFrame, columns: list[str],
                               max_per_side: int | None) -> dict:
    """원좌표 C2ST 가 크기 때문인지 방향 때문인지 가른다."""
    report: dict = {}
    if "dynamic_label" not in frame.columns:
        return report
    matrix = frame[columns].to_numpy(dtype=float)
    norms = np.linalg.norm(matrix, axis=1)

    print()
    print("=" * 72)
    print("A2. 크기인가 방향인가 — 원좌표 C2ST 는 둘을 구분하지 못한다")
    print("=" * 72)
    table = (
        pd.DataFrame({"dynamic_label": frame["dynamic_label"].to_numpy(), "norm": norms})
        .groupby("dynamic_label")["norm"].agg(["mean", "std", "size"])
    )
    print("임베딩 L2 노름 (강약별)")
    print(table.round(3).to_string())
    order = [label for label in ("p", "mf", "f", "ff") if label in table.index]
    if len(order) >= 3:
        means = [float(table.loc[label, "mean"]) for label in order]
        monotone = all(a < b for a, b in pairwise(means))
        report["norm_monotone_with_dynamic"] = bool(monotone)
        report["norm_by_dynamic"] = dict(zip(order, means))
        spread = (max(means) - min(means)) / max(np.mean(means), 1e-12)
        report["norm_relative_spread"] = float(spread)
        print(f"  {' < '.join(order)} 순으로 단조 증가하는가: "
              f"{'그렇다' if monotone else '아니다'}  (상대 변동폭 {spread * 100:.2f} %)")

    # 방향만 남기고 다시 C2ST. 크기 정보가 사라진다.
    unit = pd.DataFrame(
        matrix / np.maximum(norms[:, None], 1e-12), columns=columns, index=frame.index
    )
    unit["dynamic_label"] = frame["dynamic_label"].to_numpy()
    rows: list[dict] = []
    labels = [label for label in ("p", "mf", "f", "ff") if (unit["dynamic_label"] == label).any()]
    for left, right in combinations(labels, 2):
        item = c2st_between(unit, columns, "dynamic_label", left, right, max_per_side)
        if item:
            rows.append(item)
    if rows:
        normalized = pd.DataFrame(rows)
        print()
        print("L2 정규화 후 C2ST (크기를 지운 뒤에도 구별되는가)")
        print(normalized[["contrast", "accuracy", "p_value", "n"]].round(4)
              .to_string(index=False))
        mean_accuracy = float(normalized["accuracy"].mean())
        report["dynamic_c2st_normalized_mean"] = mean_accuracy
        print()
        print(f"  정규화 후 평균 정확도 {mean_accuracy:.4f}")
        if mean_accuracy < 0.60:
            print("  -> 크기를 지우니 구별이 무너진다. 원좌표의 높은 정확도는")
            print("     **음량(크기)** 때문이었다. 강약 제어는 사실상 볼륨 손잡이다.")
        elif mean_accuracy > 0.75:
            print("  -> 크기를 지워도 여전히 구별된다. 방향이 실제로 바뀌었다")
            print("     = 강약이 음색까지 바꾼다. 이건 실제 바이올린과 같은 방향의 성질이다.")
        else:
            print("  -> 중간이다. 크기와 방향이 둘 다 기여한다. 단정하지 말 것.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", nargs="*", default=None)
    parser.add_argument("--features", default=None,
                        help="rms_dbfs 대조용 특징표 (기본: results/<profile>/model_features.csv)")
    parser.add_argument("--profile", default="expanded")
    parser.add_argument("--max-per-side", type=int, default=MAX_PER_SIDE,
                        help=f"C2ST 한쪽 최대 표본 (기본 {MAX_PER_SIDE}). "
                             "줄인 사실은 출력한다.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.embeddings:
        paths = [Path(item) for item in args.embeddings]
    else:
        paths = sorted((root / "results" / "embeddings").glob(f"{args.profile}_*_mert.csv"))
    paths = [path for path in paths if path.is_file()]
    if not paths:
        print("임베딩 표를 찾지 못했다. 먼저: bash scripts/run_embeddings.sh expanded",
              file=sys.stderr)
        return 2

    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    columns = embedding_columns(frame)
    if not columns:
        print("임베딩 열(e0, e1, ...)이 없다.", file=sys.stderr)
        return 2

    # 무음 클립은 임베딩도 무의미하다. render_grade 가 있으면 거른다.
    if "render_grade" in frame.columns:
        before = len(frame)
        frame = frame.loc[frame["render_grade"].eq("ok")].copy()
        print(f"렌더 실패 {before - len(frame):,}개 제외 -> {len(frame):,} 클립")

    print(f"임베딩 {len(frame):,} 클립 x {len(columns)} 차원")
    report: dict[str, object] = {"n_clips": len(frame), "n_dims": len(columns)}

    # ── A. 강약 대조 + 주법 양성 대조 ──────────────────────────────────
    print()
    print("=" * 72)
    print("A. 강약이 음색을 바꾸는가  (MERT 는 음량에 눈이 멀어 있다)")
    print("=" * 72)
    rows: list[dict] = []
    if "dynamic_label" in frame.columns:
        labels = [label for label in ("p", "mf", "f", "ff") if (frame["dynamic_label"] == label).any()]
        for left, right in combinations(labels, 2):
            item = c2st_between(frame, columns, "dynamic_label", left, right,
                                args.max_per_side)
            if item:
                item["kind"] = "강약(관심 대상)"
                rows.append(item)

    if "technique" in frame.columns:
        techniques = sorted(frame["technique"].dropna().unique())
        # 양성 대조는 몇 쌍이면 충분하지만 **앵커가 한쪽으로 쏠리면 안 된다.**
        # combinations(...)[:6] 은 전부 첫 주법을 앵커로 잡는다(실제로 그랬다).
        # 겹치지 않는 쌍으로 흩어 고른다.
        candidates = list(combinations(techniques, 2))
        stride = max(1, len(candidates) // 6)
        for left, right in candidates[::stride][:6]:
            item = c2st_between(frame, columns, "technique", left, right,
                                args.max_per_side)
            if item:
                item["kind"] = "주법(양성 대조)"
                rows.append(item)

    if not rows:
        print("  대조를 만들 수 없다 (열 또는 표본 부족).")
    else:
        table = pd.DataFrame(rows)
        print(table[["kind", "contrast", "accuracy", "p_value", "n"]]
              .round(4).to_string(index=False))
        dyn = table.loc[table["kind"].str.startswith("강약"), "accuracy"]
        tech = table.loc[table["kind"].str.startswith("주법"), "accuracy"]
        report["dynamic_c2st_mean"] = float(dyn.mean()) if len(dyn) else None
        report["technique_c2st_mean"] = float(tech.mean()) if len(tech) else None
        print()
        if len(tech) and tech.mean() < 0.7:
            print("  ⚠ 양성 대조 실패: 주법조차 구별하지 못한다 "
                  f"(평균 {tech.mean():.3f}). 임베딩 파이프라인을 먼저 의심할 것. "
                  "강약 결과는 해석하지 않는다.")
            report["positive_control_passed"] = False
        elif len(tech):
            report["positive_control_passed"] = True
            print(f"  양성 대조 통과 — 주법 평균 정확도 {tech.mean():.3f}")
            if len(dyn):
                print(f"  강약 평균 정확도 {dyn.mean():.3f}")
                if dyn.mean() < 0.60:
                    print("  -> 강약을 바꿔도 음색이 (MERT 가 볼 만큼은) 바뀌지 않는다.")
                    print("     실제 바이올린은 활 압력이 배음 구조를 바꾼다. 다음 줄의")
                    print("     음량 대조와 반드시 함께 읽을 것.")

    report.update(magnitude_versus_direction(frame, columns, args.max_per_side))

    # ── 음량은 실제로 바뀌었는가 (해석을 가르는 대조) ──────────────────
    features_path = Path(args.features) if args.features else (
        root / "results" / args.profile / "model_features.csv"
    )
    if features_path.is_file():
        features = pd.read_csv(features_path)
        if {"dynamic_label", "rms_dbfs"}.issubset(features.columns):
            if "render_grade" in features.columns:
                features = features.loc[features["render_grade"].eq("ok")]
            levels = features.groupby("dynamic_label")["rms_dbfs"].agg(["mean", "count"])
            print()
            print("음량은 실제로 바뀌었는가 (rms_dbfs)")
            print(levels.round(2).to_string())
            spread = float(levels["mean"].max() - levels["mean"].min())
            report["dynamic_rms_spread_db"] = spread
            print(f"  강약 간 최대 차이 {spread:.2f} dB")
            if spread < 1.0:
                print("  -> 음량조차 거의 안 바뀌었다. '볼륨 손잡이'가 아니라")
                print("     **제어가 사실상 작동하지 않았다**는 뜻이다. 결론이 달라진다.")
    else:
        print()
        print(f"  (특징표가 없어 음량 대조 생략: {features_path})")

    # ── B. 제어축 직교성 ────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("B. 제어축이 직교하는가")
    print("=" * 72)
    if {"technique", "dynamic_label"}.issubset(frame.columns):
        base_dyn, target_dyn = "mf", "f"
        # ⚠ 앵커 편향 주의: 처음엔 combinations(...)[:6] 을 썼는데 정렬 순서상
        # 전부 harmonic 을 앵커로 하는 쌍이었다. 이 절은 C2ST 를 안 쓰고 평균만
        # 계산하므로 비싸지 않다 — 전 조합을 양방향으로 본다.
        techniques = sorted(frame["technique"].dropna().unique())
        pairs: list[dict] = []
        for base_tech, other_tech in permutations(techniques, 2):
            anchor = cell_mean(frame.loc[frame["technique"].eq(base_tech)
                                         & frame["dynamic_label"].eq(base_dyn)], columns)
            moved_dyn = cell_mean(frame.loc[frame["technique"].eq(base_tech)
                                            & frame["dynamic_label"].eq(target_dyn)], columns)
            moved_tech = cell_mean(frame.loc[frame["technique"].eq(other_tech)
                                             & frame["dynamic_label"].eq(base_dyn)], columns)
            if anchor is None or moved_dyn is None or moved_tech is None:
                continue
            delta_dyn = moved_dyn - anchor
            delta_tech = moved_tech - anchor
            denominator = np.linalg.norm(delta_dyn) * np.linalg.norm(delta_tech)
            if denominator < 1e-12:
                continue
            pairs.append({
                "anchor": f"{base_tech}/{base_dyn}",
                "delta_dyn": f"->{target_dyn}",
                "delta_tech": f"->{other_tech}",
                "cos": float(np.dot(delta_dyn, delta_tech) / denominator),
                "norm_dyn": float(np.linalg.norm(delta_dyn)),
                "norm_tech": float(np.linalg.norm(delta_tech)),
            })
        if pairs:
            table = pd.DataFrame(pairs)
            print(table.round(4).to_string(index=False))
            mean_abs = float(table["cos"].abs().mean())
            report["control_axis_abs_cosine_mean"] = mean_abs
            print()
            print(f"  |cos| 평균 {mean_abs:.4f}  (0 이면 독립, 크면 두 제어가 얽혀 있다)")
            # 이동 크기 비교는 그 자체로 말한다.
            ratio = float(table["norm_dyn"].mean() / max(table["norm_tech"].mean(), 1e-12))
            report["dyn_to_tech_step_ratio"] = ratio
            print(f"  이동 크기 비 (강약/주법) {ratio:.4f} — 1 보다 훨씬 작으면 "
                  "강약 제어가 표현을 거의 안 움직인다는 뜻이다.")
        else:
            print("  셀을 채우지 못했다.")
    else:
        print("  technique/dynamic_label 열이 없다.")

    # ── C. 합성성 간극 (모델 전용) ──────────────────────────────────────
    print()
    print("=" * 72)
    print("C. 합성성 간극 — 강약이동 + 주법이동 == 동시이동 인가")
    print("=" * 72)
    if {"technique", "dynamic_label"}.issubset(frame.columns):
        gaps: list[dict] = []
        techniques = sorted(frame["technique"].dropna().unique())
        for base_tech, other_tech in permutations(techniques, 2):
            def cell(tech: str, dyn: str) -> np.ndarray | None:
                return cell_mean(frame.loc[frame["technique"].eq(tech)
                                           & frame["dynamic_label"].eq(dyn)], columns)
            anchor = cell(base_tech, "mf")
            only_dyn = cell(base_tech, "f")
            only_tech = cell(other_tech, "mf")
            both = cell(other_tech, "f")
            if any(item is None for item in (anchor, only_dyn, only_tech, both)):
                continue
            predicted = (only_dyn - anchor) + (only_tech - anchor)
            observed = both - anchor
            residual = float(np.linalg.norm(observed - predicted))
            scale = float(np.linalg.norm(predicted))
            if scale < 1e-12:
                continue
            gaps.append({
                "cell": f"{base_tech}->{other_tech}, mf->f",
                "gap_ratio": residual / scale,
                "residual": residual,
                "predicted_norm": scale,
            })
        if gaps:
            table = pd.DataFrame(gaps)
            print(table.round(4).to_string(index=False))
            # ⚠ gap_ratio 는 분모(predicted_norm)가 작으면 부풀어 오른다.
            # legato_slur <-> sustain 처럼 임베딩상 거의 같은 주법 쌍이 그렇다
            # (norm 0.21). 분모가 작은 쌍을 빼고도 결론이 유지되는지 함께 본다.
            median_norm = float(table["predicted_norm"].median())
            stable = table.loc[table["predicted_norm"] >= 0.5 * median_norm]
            mean_gap = float(table["gap_ratio"].mean())
            if len(stable) < len(table):
                print()
                print(f"  분모가 작은 쌍 {len(table) - len(stable)}개를 빼면 "
                      f"평균 간극비 {float(stable['gap_ratio'].mean()):.4f} "
                      f"(전체 {mean_gap:.4f})")
                report["compositionality_gap_ratio_trimmed"] = float(
                    stable["gap_ratio"].mean()
                )
            report["compositionality_gap_ratio_mean"] = mean_gap
            print()
            print(f"  평균 간극비 {mean_gap:.4f}  "
                  "(0 이면 두 제어가 더해지듯 합쳐진다, 크면 서로 간섭한다)")
            print("  ⚠ 이건 모델 내부 일관성이다. 실연주 기준선이 있어야")
            print("     '사람도 이만큼 어긋나는가'를 답할 수 있다 — CG 의 예비판이다.")
        else:
            print("  셀을 채우지 못했다.")

    output = Path(args.output) if args.output else (
        root / "results" / f"embedding_model_only_{args.profile}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"저장: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
