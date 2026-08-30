#!/usr/bin/env bash
# 프로파일 하나를 끝까지 분석한다: QC -> 특징 추출 -> 지표 -> 그림.
#
# run_pilot_analysis.sh 는 pilot manifest 만 본다. 본체(expanded 18,624클립)를
# 분석하는 경로가 큐에 없었다. 이 스크립트가 그 자리를 메운다.
#
# 사용: scripts/run_profile_analysis.sh {pilot|full|expanded}
#
# ⚠ 실연주 녹음이 아직 없으면:
#   - 모델 쪽 QC·특징 추출은 **전부 정상 수행**된다 (여기까지가 이 단계의 대부분)
#   - 실악기 기준선이 필요한 지표(CEA/HCEL/CG)는 계산할 수 없다 -> 건너뛴다
#   - 모델만으로 가능한 분석(단조성, delayed branch 인과성)은 그대로 나온다
#   녹음이 들어온 뒤 이 스크립트를 다시 돌리면 지표까지 완성된다.
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_PROFILE="${1:-expanded}"
VIOCF_RESULTS="${VIOCF_ROOT}/results/${VIOCF_PROFILE}"

case "${VIOCF_PROFILE}" in
  pilot | full | expanded) ;;
  *)
    echo "프로파일은 pilot, full, expanded 중 하나여야 한다."
    exit 2
    ;;
esac
if [[ ! -f "${VIOCF_ROOT}/.venv/bin/activate" ]]; then
  echo "분석 환경이 없다. 먼저: bash scripts/bootstrap_analysis.sh"
  exit 2
fi

cd "${VIOCF_ROOT}"
source .venv/bin/activate
mkdir -p "${VIOCF_RESULTS}"

echo "=============================================================="
echo "프로파일 분석: ${VIOCF_PROFILE}"
echo "=============================================================="

VIOCF_FEATURE_FILES=()
for VIOCF_KIND in model real delayed_model delayed_real; do
  VIOCF_MANIFEST="${VIOCF_ROOT}/manifests/${VIOCF_PROFILE}_${VIOCF_KIND}.csv"
  [[ -f "${VIOCF_MANIFEST}" ]] || continue

  echo
  echo "── ${VIOCF_KIND} ──"
  viocf qc \
    --manifest "${VIOCF_MANIFEST}" \
    --output "${VIOCF_RESULTS}/${VIOCF_KIND}_qc.csv" || true

  VIOCF_FEATURES="${VIOCF_RESULTS}/${VIOCF_KIND}_features.csv"
  viocf features \
    --manifest "${VIOCF_MANIFEST}" \
    --output "${VIOCF_FEATURES}"

  # 오디오가 아직 없는 쪽(대개 실연주)은 0행이 나온다. 그건 오류가 아니라
  # '아직 안 찍었다'는 뜻이므로 건너뛰고 계속 간다.
  VIOCF_ROWS="$(python - "${VIOCF_FEATURES}" <<'PY'
import sys
from pathlib import Path
import pandas as pd
path = Path(sys.argv[1])
try:
    print(len(pd.read_csv(path)) if path.stat().st_size else 0)
except Exception:
    print(0)
PY
)"
  echo "  ${VIOCF_ROWS} 행"
  if [[ "${VIOCF_ROWS}" -gt 0 ]]; then
    VIOCF_FEATURE_FILES+=("${VIOCF_FEATURES}")
  fi
done

if [[ "${#VIOCF_FEATURE_FILES[@]}" -eq 0 ]]; then
  echo
  echo "특징표가 하나도 만들어지지 않았다. 오디오 수집을 확인할 것:"
  echo "  ls data/model_audio | head"
  exit 2
fi

echo
echo "=============================================================="
echo "지표 계산"
echo "=============================================================="
viocf metrics \
  --features "${VIOCF_FEATURE_FILES[@]}" \
  --output-dir "${VIOCF_RESULTS}/metrics"

echo
python - "${VIOCF_RESULTS}/metrics/metrics_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("지표 요약 파일이 없다.")
summary = json.loads(path.read_text(encoding="utf-8"))

# 실악기 기준선이 필요한 지표와 모델만으로 되는 지표를 나눠 보여준다.
real_based = ("effect_alignment", "excess_leakage", "compositionality_gap_mean")
model_only = ("monotonic_violation_rate", "delayed_model_only_prebranch")

print("── 실악기 기준선이 필요한 지표 ──")
for key in real_based:
    value = summary.get(key)
    print(f"  {key:32s} {'(아직 계산 불가 — 녹음 필요)' if value is None else value}")

print()
print("── 모델만으로 나오는 결과 ──")
for key in model_only:
    value = summary.get(key)
    print(f"  {key:32s} {'(없음)' if value is None else value}")
PY

echo
echo "그림 생성"
viocf figures \
  --features "${VIOCF_FEATURE_FILES[@]}" \
  --metrics-dir "${VIOCF_RESULTS}/metrics" \
  --output-dir "${VIOCF_RESULTS}/figures" || \
  echo "  (그림 일부는 실악기 데이터가 있어야 나온다 — 건너뜀)"

echo
echo "완료: ${VIOCF_RESULTS}"
