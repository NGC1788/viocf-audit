#!/usr/bin/env python3
"""제어가 시간 축을 따르는가, 전역 토큰인가.

평균이 같고 모양만 다른 CC1 궤적 여섯을 넣었다. 두 가지를 잰다.

## 1. 추종 상관 — 소리 크기가 궤적을 따라가는가

클립마다 RMS 포락선과 의도한 CC1 궤적의 상관을 낸다.

    r ~ 1   제어가 시간 축을 따른다
    r ~ 0   따르지 않는다

`constant` 는 궤적이 상수라 상관이 정의되지 않으므로 제외한다.

## 2. 모양 구분 가능성 — 모양이 다르면 소리도 다른가

같은 초기 난수에서 모양만 바꾼 짝을 비교한다. 특히 `ramp_up` 과 `ramp_down`
은 평균·표준편차·최솟값·최댓값이 **완전히 같고 시간 방향만 뒤집힌** 쌍이다.
전역 토큰이라면 이 둘이 구별되지 않아야 한다.

## 판정

  추종 상관 높음 + 모양 구분됨        진짜 시간 제어
  추종 상관 ~0  + 모양 구분 안 됨     **전역 토큰** — 시계열인 척할 뿐
  그 사이                            부분적. 크기로 말할 것

⚠ 인과 기준 렌더러를 같은 분석에 넣어 r ≈ 1 이 나오는지 먼저 본다.
   그게 안 나오면 측정이 틀린 것이고 VIOLET 결과를 해석할 수 없다.

사용:
  python scripts/analyze_temporal_control.py
  python scripts/analyze_temporal_control.py --features 다른표.csv --label 기준렌더러
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viocf.audio import amplitude_to_db, frame_rms
from viocf.temporal_control import (
    _trajectory,
)

FRAME_SECONDS = 0.046
HOP_SECONDS = 0.010
# 포락선 앞뒤를 조금 버린다. 어택과 릴리스는 CC1 이 아니라 음의 시작·끝이 만든다.
EDGE_TRIM_S = 0.25


def follow_correlation(
    path: Path, shape: str, onset: float, note_seconds: float, steps: int,
) -> float:
    """RMS 포락선과 의도한 CC1 궤적의 피어슨 상관."""
    samples, rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    start = int((onset + EDGE_TRIM_S) * rate)
    stop = int((onset + note_seconds - EDGE_TRIM_S) * rate)
    if stop - start < rate // 2:
        return float("nan")
    segment = samples[start:stop]
    frame_length = max(64, round(FRAME_SECONDS * rate))
    hop = max(1, round(HOP_SECONDS * rate))
    envelope = frame_rms(segment, frame_length, hop)
    if envelope.size < 8:
        return float("nan")
    envelope_db = np.array([amplitude_to_db(max(float(v), 1e-9)) for v in envelope])

    # 같은 시간 위치의 CC1 값을 뽑는다. 궤적은 [onset, onset+note] 에 균등하다.
    values = np.array(_trajectory(shape, steps), dtype=float)
    frame_times = (
        (onset + EDGE_TRIM_S)
        + (np.arange(envelope_db.size) * hop + frame_length / 2) / rate
    )
    positions = np.clip(
        ((frame_times - onset) / note_seconds * steps).astype(int), 0, steps - 1
    )
    target = values[positions]
    if np.std(target) < 1e-9 or np.std(envelope_db) < 1e-9:
        return float("nan")
    return float(np.corrcoef(target, envelope_db)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--audio-column", default="audio_path")
    parser.add_argument("--label", default="VIOLET")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest = Path(args.manifest) if args.manifest else (
        root / "manifests" / "temporal_control" / "trajectories.csv"
    )
    if not manifest.is_file():
        print(f"manifest 가 없다: {manifest}", file=sys.stderr)
        return 2

    frame = pd.read_csv(manifest)
    records: list[dict] = []
    missing = 0
    for record in frame.to_dict(orient="records"):
        path = Path(str(record[args.audio_column]))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            missing += 1
            continue
        r = follow_correlation(
            path, str(record["trajectory_shape"]),
            float(record["note_onset_s"]), float(record["note_seconds"]),
            int(record["trajectory_steps"]),
        )
        records.append({
            "technique": record["technique"],
            "shape": record["trajectory_shape"],
            "noise_group": record["noise_group"],
            "follow_r": r,
            "audio_path": str(path),
        })
    if missing:
        print(f"오디오 없음 {missing:,}개 (생성·수집 확인)")
    if not records:
        print("분석할 오디오가 없다.", file=sys.stderr)
        return 2
    data = pd.DataFrame(records)

    print("=" * 74)
    print(f"1. 추종 상관 — 소리 크기가 궤적을 따라가는가  [{args.label}]")
    print("=" * 74)
    print("   진짜 시간 제어면 r → 1,  전역 토큰이면 r → 0")
    varying = data.loc[data["shape"].ne("constant") & data["follow_r"].notna()]
    table = varying.groupby("shape")["follow_r"].agg(["median", "mean", "std", "size"])
    print(table.round(4).to_string())
    overall = float(varying["follow_r"].median())
    print()
    print(f"  전체 중앙값 r = {overall:.4f}   (n = {len(varying):,})")

    print()
    print("=" * 74)
    print("2. 모양 구분 가능성 — 같은 난수, 모양만 다를 때")
    print("=" * 74)
    print("   ramp_up 과 ramp_down 은 평균·분산·최소·최대가 같고 시간만 뒤집힌 쌍이다.")
    pairs: list[dict] = []
    for (_, group_name), group in data.groupby(["technique", "noise_group"]):
        by_shape = group.set_index("shape")
        if not {"ramp_up", "ramp_down"}.issubset(by_shape.index):
            continue
        up = by_shape.loc["ramp_up", "follow_r"]
        down = by_shape.loc["ramp_down", "follow_r"]
        if np.isfinite(up) and np.isfinite(down):
            pairs.append({"noise_group": group_name, "up": up, "down": down,
                          "gap": float(up - down)})
    if pairs:
        pair_frame = pd.DataFrame(pairs)
        print(f"  상승 r 중앙값  {pair_frame['up'].median():+.4f}")
        print(f"  하강 r 중앙값  {pair_frame['down'].median():+.4f}")
        print(f"  짝 {len(pair_frame)}개")
        # 시간 제어가 실재하면 두 값 모두 +1 에 가까워야 한다(각자 자기 궤적을 따름).
        # 전역 토큰이면 둘 다 0 근처다.
        from scipy.stats import wilcoxon
        try:
            stat = wilcoxon(pair_frame["up"], pair_frame["down"])
            print(f"  Wilcoxon (상승 vs 하강) p = {stat.pvalue:.3e}")
        except ValueError:
            pass

    print()
    print("=" * 74)
    print("판정")
    print("=" * 74)
    if overall > 0.7:
        print(f"  추종 상관 {overall:.3f} — **제어가 시간 축을 따른다.**")
        print("  전역 토큰 가설은 기각된다. 누출은 다른 기제로 설명해야 한다.")
    elif overall < 0.3:
        print(f"  추종 상관 {overall:.3f} — **시간 축을 따르지 않는다.**")
        print("  평균이 같으면 모양이 달라도 같은 소리가 난다는 뜻이다.")
        print("  강약 제어는 시간 제어가 아니라 클립 전체에 붙는 전역 토큰이며,")
        print("  시계열의 모습으로 입력받을 뿐이다.")
        print()
        print("  ⚠ 이 결론은 인과 기준 렌더러에서 r ≈ 1 이 나올 때만 유효하다.")
        print("     같은 분석을 그쪽에도 돌려 확인할 것.")
    else:
        print(f"  추종 상관 {overall:.3f} — 부분적으로 따른다.")
        print("  단정하지 말고 모양별 값을 함께 보고할 것.")

    output = Path(args.output) if args.output else (
        root / "results" / f"temporal_control_{args.label}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    print()
    print(f"저장: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
