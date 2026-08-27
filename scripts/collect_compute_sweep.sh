#!/usr/bin/env bash
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_SWEEP_RUN_ID="${VIOCF_SWEEP_RUN_ID:-}"
VIOCF_MANIFEST_DIR="${VIOCF_ROOT}/manifests/sweep"
VIOCF_REPORT_DIR="${VIOCF_ROOT}/results/sweep_collect"

if [[ -z "${VIOCF_SWEEP_RUN_ID}" ]]; then
  echo "Set VIOCF_SWEEP_RUN_ID to the id used by scripts/run_compute_sweep.sh."
  exit 2
fi
if [[ ! -f "${VIOCF_ROOT}/.venv/bin/activate" ]]; then
  echo "Missing analysis environment. Run scripts/bootstrap_analysis.sh first."
  exit 2
fi

source "${VIOCF_ROOT}/.venv/bin/activate"
mkdir -p "${VIOCF_REPORT_DIR}"

collect_one() {
  local label="$1"
  local manifest="$2"
  local run_dir="${VIOCF_ROOT}/logs/violet/sweep/${VIOCF_SWEEP_RUN_ID}_${label}"
  local report="${VIOCF_REPORT_DIR}/${manifest##*/}"

  if [[ ! -d "${run_dir}" ]]; then
    echo "Missing run: ${run_dir}"
    exit 2
  fi
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
