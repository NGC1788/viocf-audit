#!/usr/bin/env python3
"""렌더 실패율 전수 측정 — 원본 평가로는 볼 수 없는 결과.

## 왜 이게 결과인가

VIOLET 원본은 near-silent 렌더가 나오면 **다른 seed 로 재시도**한다
(`DEFAULT_MAX_RENDER_ATTEMPTS`). 우리는 짝 실험을 지키려고 그 재시도를 껐다
(`+model.test_max_render_attempts=1`). 그 결과 **원본 평가가 구조적으로 가려 버리는
실패 모드를 우리는 그대로 관측한다.**

여기에 시드 32개 설계가 겹친다. 시드가 5개면 "이 조건에서 6% 확률로 무음"을
추정할 수 없다. 32개여야 셀마다 실패율에 의미가 생긴다.

## 무엇을 세는가

클립을 세 등급으로 나눈다.

  silent : peak < -50 dBFS            사실상 아무 소리도 없음
  weak   : -50 <= peak < -35 dBFS     소리는 있으나 비정상적으로 작음
  ok     : peak >= -35 dBFS

⚠ 등급 경계는 관례가 아니라 관측에서 나왔다. 정상 셀의 peak 중앙값이
-12 ~ -20 dBFS 에 몰려 있고, 무너지는 셀은 -34 ~ -57 dBFS 로 완전히 분리된다.
그 사이가 비어 있어서 -35 를 경계로 잡았다. 실행하면 히스토그램을 함께 출력하므로
분리가 유지되는지 매번 확인할 수 있다.

사용:
  python scripts/analyze_render_failures.py                    # expanded
  python scripts/analyze_render_failures.py --profile full
  python scripts/analyze_render_failures.py --manifest a.csv b.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SILENT_DBFS = -50.0
WEAK_DBFS = -35.0


def peak_dbfs(path: Path) -> float:
    try:
        samples, _ = sf.read(path, dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 - 읽기 실패도 결과의 일부다
        return float("nan")
    if samples.size == 0:
        return -np.inf
    return float(20 * np.log10(max(float(np.max(np.abs(samples))), 1e-9)))


def grade(value: float) -> str:
    if not np.isfinite(value):
        return "unreadable"
    if value < SILENT_DBFS:
        return "silent"
    if value < WEAK_DBFS:
        return "weak"
    return "ok"


def scan(manifests: list[Path], root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for manifest in manifests:
        if not manifest.is_file():
            continue
        frame = pd.read_csv(manifest)
        total = len(frame)
        for index, record in enumerate(frame.to_dict(orient="records"), start=1):
            path = Path(str(record.get("audio_path", "")))
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                continue
            value = peak_dbfs(path)
            rows.append({
                "clip_id": record.get("clip_id"),
                "prompt_id": record.get("prompt_id"),
                "pattern": record.get("pattern"),
                "technique": record.get("technique"),
                "dynamic_label": record.get("dynamic_label"),
                "replicate": record.get("replicate"),
                "analysis_tier": record.get("analysis_tier"),
                "peak_dbfs": value,
                "render_grade": grade(value),
            })
            if index % 2000 == 0:
                print(f"  {manifest.name}: {index}/{total}", file=sys.stderr)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="expanded")
    parser.add_argument("--manifest", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.manifest:
        manifests = [Path(item) for item in args.manifest]
    else:
        manifests = [
            root / "manifests" / f"{args.profile}_model.csv",
            root / "manifests" / f"{args.profile}_delayed_model.csv",
        ]

    print(f"스캔: {[m.name for m in manifests]}", file=sys.stderr)
    data = scan(manifests, root)
    if data.empty:
        print("오디오를 찾지 못했다. 수집을 확인할 것.")
        return 2

    output = Path(args.output) if args.output else root / "results" / f"render_grades_{args.profile}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)

    total = len(data)
    counts = data["render_grade"].value_counts()
    print("=" * 70)
    print(f"렌더 등급 — {total:,} 클립")
    print("=" * 70)
    for name in ("ok", "weak", "silent", "unreadable"):
        n = int(counts.get(name, 0))
        print(f"  {name:11s} {n:6,d}  {n / total * 100:5.2f} %")

    # 등급 경계가 실제로 분포의 빈 구간에 놓였는지 매번 확인한다.
    print()
    print("peak dBFS 분포 (경계가 골짜기에 놓여야 한다)")
    finite = data.loc[np.isfinite(data["peak_dbfs"]), "peak_dbfs"]
    hist, edges = np.histogram(finite, bins=np.arange(-90, 5, 5))
    for count, low in zip(hist, edges[:-1]):
        mark = " <- 무음 경계" if low == SILENT_DBFS else (" <- 약함 경계" if low == WEAK_DBFS else "")
        bar = "#" * int(60 * count / max(hist.max(), 1))
        print(f"  {low:+4.0f} dBFS {count:6,d} {bar}{mark}")

    print()
    print("=" * 70)
    print("패턴 x 주법별 실패율 (silent + weak)")
    print("=" * 70)
    data["failed"] = data["render_grade"].isin(("silent", "weak"))
    table = data.pivot_table(index="pattern", columns="technique",
                             values="failed", aggfunc="mean") * 100
    print(table.round(1).to_string())

    print()
    print("실패율이 높은 셀 (>= 5 %)")
    cell = (data.groupby(["pattern", "technique", "dynamic_label"])["failed"]
            .agg(["mean", "count"]).reset_index())
    cell["rate_pct"] = (cell["mean"] * 100).round(1)
    bad = cell.loc[cell["mean"] >= 0.05].sort_values("mean", ascending=False)
    print(bad[["pattern", "technique", "dynamic_label", "rate_pct", "count"]]
          .to_string(index=False) if len(bad) else "  없음")

    print()
    print("=" * 70)
    print("정상 조건의 기저 실패율 — 원본 평가가 재시도로 가려 버리는 값")
    print("=" * 70)
    # 지속음 패턴 x 짧은 주법은 악보와 지시가 모순되는 조합이라 따로 뺀다.
    short_techniques = {"pizzicato", "spiccato", "staccato"}
    sustained = {"long", "long_short"}
    degenerate = data["pattern"].isin(sustained) & data["technique"].isin(short_techniques)
    normal = data.loc[~degenerate]
    rate = normal["failed"].mean() * 100
    print(f"  모순 조합 제외 {len(normal):,} 클립 중 실패 {int(normal['failed'].sum()):,} "
          f"= {rate:.2f} %")
    print(f"  모순 조합      {int(degenerate.sum()):,} 클립 중 실패 "
          f"{int(data.loc[degenerate, 'failed'].sum()):,} "
          f"= {data.loc[degenerate, 'failed'].mean() * 100:.2f} %")
    print()
    print(f"저장: {output}")
    print()
    print("⚠ silent/weak 클립은 특징값이 무의미하므로 지표에서 제외해야 한다.")
    print("   metrics 는 render_grade 열이 있으면 자동으로 ok 행만 사용한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
