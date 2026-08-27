#!/usr/bin/env bash
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_SWEEP_RUN_ID="${VIOCF_SWEEP_RUN_ID:-}"
VIOCF_MANIFEST_DIR="${VIOCF_ROOT}/manifests/sweep"
VIOCF_REPORT_DIR="${VIOCF_ROOT}/results/sweep_collect"

# VIOCF_SWEEP_RUN_ID 는 이제 선택사항이다.
# 큐는 T2(dense)/T3(guidance)/T4(steps)를 **각각 따로** run_compute_sweep.sh 로 돌리고,
# 그때마다 run id 가 새 타임스탬프로 생긴다. 즉 하나의 id 로는 전부 찾을 수 없다.
# 비워 두면 라벨로 디렉터리를 자동 탐색한다.
if [[ -n "${VIOCF_SWEEP_RUN_ID}" ]]; then
  echo "run id 고정: ${VIOCF_SWEEP_RUN_ID}"
else
  echo "run id 자동 탐색 (라벨로 logs/violet/sweep 에서 찾는다)"
fi
if [[ ! -f "${VIOCF_ROOT}/.venv/bin/activate" ]]; then
  echo "Missing analysis environment. Run scripts/bootstrap_analysis.sh first."
  exit 2
fi

source "${VIOCF_ROOT}/.venv/bin/activate"
mkdir -p "${VIOCF_REPORT_DIR}"

# 라벨로 run 디렉터리를 찾는다. 같은 라벨이 여러 번 돌았으면 가장 최근 것을 쓴다.
find_run_dir() {
  local label="$1"
  if [[ -n "${VIOCF_SWEEP_RUN_ID}" ]]; then
    printf '%s' "${VIOCF_ROOT}/logs/violet/sweep/${VIOCF_SWEEP_RUN_ID}_${label}"
    return 0
  fi
  local newest=""
  local candidate
  for candidate in "${VIOCF_ROOT}"/logs/violet/sweep/*_"${label}"; do
    [[ -d "${candidate}" ]] || continue
    if [[ -z "${newest}" || "${candidate}" -nt "${newest}" ]]; then
      newest="${candidate}"
    fi
  done
  printf '%s' "${newest}"
}

collect_one() {
  local label="$1"
  local manifest="$2"
  local run_dir
  run_dir="$(find_run_dir "${label}")"
  local report="${VIOCF_REPORT_DIR}/${manifest##*/}"

  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "생성 결과를 찾지 못했다: 라벨 ${label}"
    echo "  찾은 위치: ${VIOCF_ROOT}/logs/violet/sweep/*_${label}"
    echo "  이 단계가 아직 안 돌았거나 다른 이름으로 돌았다. 확인:"
    echo "    ls ${VIOCF_ROOT}/logs/violet/sweep/"
    exit 2
  fi
  echo "[${label}] ${run_dir##*/}"
  viocf collect-violet \
    --run-dir "${run_dir}" \
    --manifest "${manifest}" \
    --output "${report}"
  python - "${report%.csv}.summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not summary.get("all_pass", False):
    raise SystemExit(f"Collection QA failed: {sys.argv[1]}")
PY
}

collect_one "dense" "${VIOCF_MANIFEST_DIR}/dense.csv"
for manifest in "${VIOCF_MANIFEST_DIR}"/guidance_wt*_wc*.csv; do
  stem="$(basename "${manifest}" .csv)"
  label="g_${stem#guidance_}"
  collect_one "${label}" "${manifest}"
done
for manifest in "${VIOCF_MANIFEST_DIR}"/steps_n*.csv; do
  stem="$(basename "${manifest}" .csv)"
  collect_one "${stem}" "${manifest}"
done

echo "All sweep audio collected into data/model_audio/sweep."
echo "QA reports: ${VIOCF_REPORT_DIR}"
