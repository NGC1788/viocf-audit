#!/usr/bin/env python3
"""QC 실패 5,795건(31.4%)을 이유별로 가른다.

## 왜 따로 봐야 하는가

QC 임계값은 **실연주 녹음**을 염두에 두고 정한 값이다.

  min_snr_db: 30                     방 소음 대비 신호
  clip_threshold: 0.999              오디오 인터페이스 과입력
  active_threshold_db_above_noise: 12

그런데 VIOLET 출력은 마이크로 잡은 소리가 아니라 **latent diffusion 디코드**다.
디지털 무음 대신 광대역 바닥 노이즈가 깔리고, 정규화 방식도 다르다. 그래서
같은 임계값이 모델 오디오에서는 전혀 다른 것을 잡을 수 있다.

실패를 두 부류로 나눠야 한다.

  실제 결과   near_silence  — 모델이 소리를 못 만든 것. 우리가 재시도를 껐기
                            때문에 볼 수 있는 값이다.
  임계값 문제 low_snr       — 디코더 바닥 노이즈 대비 30 dB 기준이 맞는가?
              clipping      — 모델 출력 정규화가 0.999 를 스치는가?

앞은 보고할 결과이고, 뒤는 임계값을 모델 오디오에 맞게 다시 정하거나
"이 지표는 모델 오디오에 적용할 수 없다"고 명시해야 하는 사안이다.
둘을 31.4% 라는 한 숫자로 뭉뚱그리면 아무것도 말할 수 없다.

⚠ qc_pass 는 지표에서 행을 거르지 않는다(보고 전용). 무음 제외는 render_grade
   가 따로 담당한다. 그러니 이 스크립트는 진단이지 필터가 아니다.

사용:
  python scripts/analyze_qc_failures.py                  # expanded
  python scripts/analyze_qc_failures.py --profile full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 실연주를 전제로 정한 값. 모델 오디오에 그대로 쓸 수 있는지가 이 스크립트의 질문이다.
REAL_RECORDING_MIN_SNR_DB = 30.0
CLIP_THRESHOLD = 0.999

# 모델이 소리를 못 만든 것(=결과) 과 임계값 판단(=방법론 문제) 을 나눈다.
GENERATIVE_FAILURE = {"near_silence", "missing_audio", "non_finite_samples"}


def histogram(values: pd.Series, marks: dict[float, str], width: int = 46) -> None:
    finite = values.loc[np.isfinite(values)]
    if finite.empty:
        print("  (유한한 값이 없다)")
        return
    low = float(np.floor(finite.min() / 5) * 5)
    high = float(np.ceil(finite.max() / 5) * 5) + 5
    edges = np.arange(low, high, 5.0)
    counts, edges = np.histogram(finite, bins=edges)
    peak = max(int(counts.max()), 1)
    for count, edge in zip(counts, edges[:-1]):
        label = ""
        for position, text in marks.items():
            if edge <= position < edge + 5:
                label = f"  <- {text}"
        bar = "#" * int(width * count / peak)
        print(f"  {edge:+6.0f} {count:7,d} {bar}{label}")


def report(frame: pd.DataFrame, name: str) -> dict[str, float]:
    total = len(frame)
    failed = ~frame["qc_pass"].fillna(False).astype(bool)
    print("=" * 72)
    print(f"{name} — {total:,} 클립 중 QC 실패 {int(failed.sum()):,} "
          f"({failed.mean() * 100:.1f} %)")
    print("=" * 72)

    # qc_reasons 는 ';' 로 이어 붙는다. 한 클립이 여러 이유로 실패할 수 있으므로
    # 이유별 개수의 합은 실패 클립 수보다 클 수 있다.
    exploded = (
        frame.loc[failed, "qc_reasons"].fillna("").str.split(";").explode().str.strip()
    )
    exploded = exploded.loc[exploded.ne("")]
    print()
    print("이유별 (한 클립이 여러 이유를 가질 수 있어 합이 실패 수보다 클 수 있다)")
    for reason, count in exploded.value_counts().items():
        kind = "결과" if reason in GENERATIVE_FAILURE else "임계값 판단"
        print(f"  {reason:24s} {count:7,d}  {count / total * 100:5.2f} %  [{kind}]")

    generative = frame.loc[failed].loc[
        frame.loc[failed, "qc_reasons"].fillna("").apply(
            lambda text: any(item in GENERATIVE_FAILURE for item in text.split(";"))
        )
    ]
    print()
    print(f"  생성 실패(모델이 소리를 못 냄)  {len(generative):6,d}  "
          f"{len(generative) / total * 100:5.2f} %   <- 보고할 결과")
    print(f"  임계값만 걸린 것                {int(failed.sum()) - len(generative):6,d}  "
          f"{(int(failed.sum()) - len(generative)) / total * 100:5.2f} %   <- 기준 재검토 대상")

    if "snr_db" in frame.columns:
        print()
        print(f"snr_db 분포 — 실연주 기준선 {REAL_RECORDING_MIN_SNR_DB:.0f} dB 가 "
              "분포 어디에 놓였는가")
        print("  (기준선이 분포 한가운데를 자르면 그 기준은 모델 오디오용이 아니다)")
        histogram(pd.to_numeric(frame["snr_db"], errors="coerce"),
                  {REAL_RECORDING_MIN_SNR_DB: "min_snr_db"})

    if "peak_dbfs" in frame.columns:
        print()
        print("peak_dbfs 분포")
        histogram(pd.to_numeric(frame["peak_dbfs"], errors="coerce"),
                  {-50.0: "무음", -35.0: "약함"})

    for column in ("technique", "dynamic_label", "pattern"):
        if column not in frame.columns:
            continue
        print()
        print(f"{column}별 실패율 %")
        table = frame.assign(_failed=failed).pivot_table(
            index=column, values="_failed", aggfunc=["mean", "count"]
        )
        table.columns = ["실패율", "n"]
        table["실패율"] = (table["실패율"] * 100).round(1)
        print(table.to_string())

    return {"total": total, "failed": int(failed.sum()), "generative": len(generative)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="expanded")
    parser.add_argument("--qc", nargs="*", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.qc:
        paths = [Path(item) for item in args.qc]
    else:
        paths = sorted((root / "results" / args.profile).glob("*_qc.csv"))

    paths = [path for path in paths if path.is_file()]
    if not paths:
        print(f"QC 표를 찾지 못했다: results/{args.profile}/*_qc.csv", file=sys.stderr)
        return 2

    for path in paths:
        frame = pd.read_csv(path)
        if "qc_pass" not in frame.columns:
            continue
        report(frame, path.stem)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
