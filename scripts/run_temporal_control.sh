#!/usr/bin/env bash
# 시간 제어 시험 실행 — 평균 같고 모양 다른 CC1 궤적 6종.
#
# 샘플러 설정이 하나뿐이라 조각을 나눌 필요가 없다(w_cc·steps 를 안 흔든다).
# 576 클립, 약 40분.
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_REPLICATES="${1:-32}"
VIOCF_PROFILE="temporal_control"
VIOCF_RUN_ID="tctrl_$(date +%Y%m%d_%H%M%S)"
VIOCF_RUN_DIR="${VIOCF_ROOT}/logs/violet/${VIOCF_PROFILE}/${VIOCF_RUN_ID}"

cd "${VIOCF_ROOT}"
[[ -f .venv/bin/activate ]] || { echo "먼저: bash scripts/bootstrap_analysis.sh"; exit 2; }
source .venv/bin/activate

echo "=============================================================="
echo "시간 제어 시험  replicates=${VIOCF_REPLICATES}"
echo "=============================================================="
viocf make-temporal-control --replicates "${VIOCF_REPLICATES}"

VIOCF_MIDI_DIR="${VIOCF_ROOT}/data/midi/${VIOCF_PROFILE}"
VIOCF_MANIFEST="${VIOCF_ROOT}/manifests/${VIOCF_PROFILE}/trajectories.csv"

VIOCF_MIDI_DIR_OVERRIDE="${VIOCF_MIDI_DIR}" \
VIOCF_RUN_ID="${VIOCF_RUN_ID}" \
  bash "${VIOCF_ROOT}/scripts/run_violet.sh" "${VIOCF_PROFILE}"

bash "${VIOCF_ROOT}/scripts/verify_violet_run.sh" "${VIOCF_RUN_DIR}" "${VIOCF_MIDI_DIR}"

viocf collect-violet \
  --run-dir "${VIOCF_RUN_DIR}" \
  --manifest "${VIOCF_MANIFEST}" \
  --output "${VIOCF_ROOT}/results/collect_${VIOCF_PROFILE}.csv"

echo
echo "=============================================================="
echo "분석"
echo "=============================================================="
python "${VIOCF_ROOT}/scripts/analyze_temporal_control.py" --label VIOLET
