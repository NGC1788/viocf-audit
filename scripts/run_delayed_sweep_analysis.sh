#!/usr/bin/env bash
# 지연 분기 확장(8,640 클립) 분석.
#
# run_profile_analysis.sh 를 못 쓰는 이유: 그쪽은 manifests/<profile>_model.csv
# 규칙을 쓰는데, 확장 스윕은 w_cc 값마다 실행을 나눠야 해서
# manifests/delayed_sweep/wc<값>.csv 로 쪼개져 있다.
#
# QC 는 건너뛴다. 무음 판정(silent_absolute)이 특징표 안에 들어 있고(개정 17),
# 이 분석은 QC 의 다른 항목을 쓰지 않는다. 8,640 클립 SHA-256 을 다시 도는 건
# 낭비다.
#
# 사용: scripts/run_delayed_sweep_analysis.sh
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_RESULTS="${VIOCF_ROOT}/results/delayed_sweep"
VIOCF_WORKERS="${VIOCF_WORKERS:-$(( $(nproc 2>/dev/null || echo 4) - 2 ))}"

cd "${VIOCF_ROOT}"
[[ -f .venv/bin/activate ]] || { echo "먼저: bash scripts/bootstrap_analysis.sh"; exit 2; }
source .venv/bin/activate
mkdir -p "${VIOCF_RESULTS}"

echo "=============================================================="
echo "지연 확장 분석  (워커 ${VIOCF_WORKERS}개)"
echo "=============================================================="

VIOCF_PARTS=()
for VIOCF_MANIFEST in "${VIOCF_ROOT}"/manifests/delayed_sweep/wc*.csv; do
  [[ -f "${VIOCF_MANIFEST}" ]] || continue
  VIOCF_TAG="$(basename "${VIOCF_MANIFEST}" .csv)"
  VIOCF_OUT="${VIOCF_RESULTS}/${VIOCF_TAG}_features.csv"
  echo
  echo "── ${VIOCF_TAG} ──"
  viocf features \
    --manifest "${VIOCF_MANIFEST}" \
    --output "${VIOCF_OUT}" \
    --workers "${VIOCF_WORKERS}"
  VIOCF_PARTS+=("${VIOCF_OUT}")
done

if [[ "${#VIOCF_PARTS[@]}" -eq 0 ]]; then
  echo "manifest 를 찾지 못했다: manifests/delayed_sweep/wc*.csv"
  echo "먼저: bash scripts/run_delayed_sweep.sh"
  exit 2
fi

# w_cc 별 표를 하나로 합친다. analyze_delayed_sweep.py 가 w_cc 열로 다시 가른다.
echo
echo "표 합치기"
python - "${VIOCF_RESULTS}/model_features.csv" "${VIOCF_PARTS[@]}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

output = Path(sys.argv[1])
frames = []
for path in sys.argv[2:]:
    frame = pd.read_csv(path)
    if frame.empty:
        print(f"  ⚠ 빈 표: {path}", file=sys.stderr)
        continue
    frames.append(frame)
if not frames:
    raise SystemExit("특징표가 하나도 만들어지지 않았다. 오디오 수집을 확인할 것.")
merged = pd.concat(frames, ignore_index=True)
merged.to_csv(output, index=False)

silent = int(merged.get("silent_absolute", pd.Series(dtype=bool)).sum())
print(f"  {len(merged):,} 행  (무음 {silent:,}, {silent / len(merged) * 100:.2f} %)")
print(f"  w_cc: {sorted(merged['w_cc'].unique())}")
print(f"  오프셋: {sorted(merged['branch_offset_s'].unique())}")
print(f"  주법 {merged['technique'].nunique()}개")
PY

echo
python "${VIOCF_ROOT}/scripts/analyze_delayed_sweep.py"

echo
echo "완료: ${VIOCF_RESULTS}"
