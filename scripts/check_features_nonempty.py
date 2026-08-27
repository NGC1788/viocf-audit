#!/usr/bin/env python3
"""특징표가 비어 있으면 분석을 중단시킨다.

왜 필요한가: 빈 표로 지표를 계산하면 숫자가 나오긴 하는데 전부 무의미하고,
그걸 결과로 착각하는 게 가장 위험하다. 실제로 겪었다 — 생성 17시간이 끝났는데
오디오 수집을 안 해서 T5 가 50초 만에 '완료'되고 T6 가 빈 CSV 로 죽었다.

사용: python scripts/check_features_nonempty.py results/a.csv results/b.csv ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main(paths: list[str]) -> int:
    problems: list[str] = []
    for name in paths:
        path = Path(name)
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"{path} (없거나 0바이트)")
            continue
        try:
            if len(pd.read_csv(path)) == 0:
                problems.append(f"{path} (0행)")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{path} ({type(exc).__name__}: {exc})")

    if not problems:
        return 0

    print("빈 특징표가 있어 분석을 중단한다:")
    for item in problems:
        print("  -", item)
    print()
    print("대개 오디오 수집을 안 한 것이다. VIOLET 은 오디오를")
    print("logs/violet/<run>/test_samples 에 쓰고, collect-violet 이 그걸")
    print("manifest 의 audio_path(data/model_audio/)로 옮긴다.")
    print()
    print("확인:")
    print("  ls data/model_audio | head")
    print("  bash scripts/collect_compute_sweep.sh                 # 스윕")
    print("  viocf collect-violet --run-dir ... --manifest ...     # 프로파일")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
