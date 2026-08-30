#!/usr/bin/env python3
"""무음을 앞구간에 의존하지 않는 절대 기준으로 다시 판정한다.

## 왜 세 번째 판정인가

1차: peak < -35 dBFS.  전수 분포에 골짜기가 없어 폐기(개정 8).
2차: 활성 구간 검출기(qc_reasons 의 near_silence).  **이것도 오염돼 있다.**

`detect_active_region` 의 문턱은

    threshold = max(noise_rms * 10^(12/20),  10^(-60/20))

이고 noise_rms 는 **클립 앞 0.35 초**의 RMS 다. 설계상 첫 음은 0.75 초에 들어오니
그 구간은 비어 있어야 한다. 그런데 VIOLET 이 0.35 초 안에 무언가를 내면
noise_rms 가 커지고 -> 문턱이 따라 올라가고 -> 어떤 프레임도 못 넘어서
**소리가 큰 클립이 '무음'으로 판정된다.**

전수 QC 가 정확히 그 증상을 보였다. 검출기가 무음이라 한 1,157 개의 peak 는
상위 25 % 가 -27.0 dBFS, 최대 -10.6 dBFS 였다. 그건 무음일 수 없다.

## 이 스크립트의 기준

    silent  <=>  max_t (프레임 RMS)(t)  <  -60 dBFS

앞구간을 전혀 보지 않는다. 조정할 상수도 문턱 하나뿐이고, 그건 "어떤 46 ms
구간도 -60 dBFS 를 못 넘으면 소리가 없다"는 절대 진술이다. 프레이밍은 검출기와
같게 둔다(46 ms 창, 10 ms 홉).

앞구간 RMS 도 함께 기록한다. **거기에 소리가 있다는 것 자체가 VIOLET 의 관측
가능한 성질이다** — 악보상 0.75 초까지는 아무 일도 일어나지 않아야 한다.

사용:
  python scripts/rescore_silence.py --profile expanded
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viocf.audio import amplitude_to_db, frame_rms, read_audio

ABSOLUTE_FLOOR_DBFS = -60.0
FRAME_SECONDS = 0.046
HOP_SECONDS = 0.010
PREROLL_SECONDS = 0.35
# 악보상 첫 음의 위치. 이 앞은 비어 있어야 한다.
NOTE_ONSET_SECONDS = 0.75


def score(path: Path) -> dict:
    audio = read_audio(path, mono=True)
    samples = audio.samples
    rate = audio.sample_rate
    if samples.size == 0:
        return {"frame_rms_max_dbfs": -np.inf, "preroll_rms_dbfs": -np.inf,
                "peak_dbfs": -np.inf, "readable": True, "empty": True}
    frame_length = max(32, round(FRAME_SECONDS * rate))
    hop_length = max(1, round(HOP_SECONDS * rate))
    curve = frame_rms(samples, frame_length, hop_length)
    preroll = samples[: max(1, round(PREROLL_SECONDS * rate))]
    return {
        "frame_rms_max_dbfs": float(amplitude_to_db(curve.max())) if curve.size else -np.inf,
        "preroll_rms_dbfs": float(
            amplitude_to_db(float(np.sqrt(np.mean(np.square(preroll.astype(np.float64))))))
        ),
        "peak_dbfs": float(amplitude_to_db(float(np.max(np.abs(samples))))),
        "readable": True,
        "empty": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="expanded")
    parser.add_argument("--manifest", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifests = ([Path(item) for item in args.manifest] if args.manifest else [
        root / "manifests" / f"{args.profile}_model.csv",
        root / "manifests" / f"{args.profile}_delayed_model.csv",
    ])

    rows: list[dict] = []
    for manifest in manifests:
        if not manifest.is_file():
            continue
        frame = pd.read_csv(manifest)
        total = len(frame)
        print(f"{manifest.name}: {total:,} 행", file=sys.stderr)
        for index, record in enumerate(frame.to_dict(orient="records"), start=1):
            path = Path(str(record.get("audio_path", "")))
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                continue
            try:
                measured = score(path)
            except Exception as exc:  # noqa: BLE001 - 읽기 실패도 결과다
                measured = {"frame_rms_max_dbfs": np.nan, "preroll_rms_dbfs": np.nan,
                            "peak_dbfs": np.nan, "readable": False,
                            "error": f"{type(exc).__name__}"}
            rows.append({
                "clip_id": record.get("clip_id"),
                "prompt_id": record.get("prompt_id"),
                "pattern": record.get("pattern"),
                "technique": record.get("technique"),
                "dynamic_label": record.get("dynamic_label"),
                "replicate": record.get("replicate"),
                **measured,
            })
            if index % 2000 == 0:
                print(f"  {index:,}/{total:,}", file=sys.stderr)

    if not rows:
        print("오디오를 찾지 못했다.", file=sys.stderr)
        return 2

    data = pd.DataFrame(rows)
    data["silent_absolute"] = data["frame_rms_max_dbfs"] < ABSOLUTE_FLOOR_DBFS
    output = Path(args.output) if args.output else (
        root / "results" / f"silence_rescore_{args.profile}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)

    total = len(data)
    silent = int(data["silent_absolute"].sum())
    print()
    print("=" * 74)
    print(f"절대 기준 재판정 — {total:,} 클립")
    print("=" * 74)
    print(f"  무음 (모든 프레임 RMS < {ABSOLUTE_FLOOR_DBFS:.0f} dBFS): "
          f"{silent:,}  {silent / total * 100:.2f} %")
    print()
    print("  ※ 이 값이 '생성 실패율'의 정본이다. 앞구간을 보지 않으므로")
    print("     오염되지 않는다.")

    print()
    print(f"프레임 RMS 최대값 분포 (경계 {ABSOLUTE_FLOOR_DBFS:.0f} dBFS)")
    finite = data.loc[np.isfinite(data["frame_rms_max_dbfs"]), "frame_rms_max_dbfs"]
    counts, edges = np.histogram(finite, bins=np.arange(-100, 5, 5))
    peak_count = max(int(counts.max()), 1)
    for count, edge in zip(counts, edges[:-1]):
        mark = "  <- 무음 경계" if edge == ABSOLUTE_FLOOR_DBFS else ""
        print(f"  {edge:+6.0f} {count:7,d} {'#' * int(50 * count / peak_count)}{mark}")

    print()
    print("=" * 74)
    print(f"앞 {PREROLL_SECONDS:.2f} 초 — 악보상 첫 음은 {NOTE_ONSET_SECONDS} 초다")
    print("=" * 74)
    preroll = data.loc[np.isfinite(data["preroll_rms_dbfs"]), "preroll_rms_dbfs"]
    for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0):
        print(f"  {q * 100:5.1f} %  {float(preroll.quantile(q)):7.1f} dBFS")
    leaked = int((preroll > ABSOLUTE_FLOOR_DBFS).sum())
    print(f"  앞구간이 -60 dBFS 를 넘는 클립: {leaked:,}  "
          f"{leaked / len(preroll) * 100:.2f} %")
    if leaked:
        print("  -> 소리가 나면 안 되는 구간에서 소리가 난다. 이것 자체가 결과다.")
        print("     (검출기의 노이즈 추정이 오염된 원인이기도 하다)")

    print()
    print("패턴 x 주법별 무음률 %")
    table = data.pivot_table(index="pattern", columns="technique",
                             values="silent_absolute", aggfunc="mean") * 100
    print(table.round(1).to_string())

    print()
    print(f"저장: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
