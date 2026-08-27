#!/usr/bin/env bash
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_RESULTS="${VIOCF_ROOT}/results/sweep"

if [[ ! -f "${VIOCF_ROOT}/.venv/bin/activate" ]]; then
  echo "Missing analysis environment. Run scripts/bootstrap_analysis.sh first."
  exit 2
fi
source "${VIOCF_ROOT}/.venv/bin/activate"
mkdir -p "${VIOCF_RESULTS}"

viocf qc \
  --manifest "${VIOCF_ROOT}/manifests/sweep/dense.csv" \
  --output "${VIOCF_RESULTS}/dense_qc.csv"
viocf qc \
  --manifest "${VIOCF_ROOT}/manifests/sweep/guidance_all.csv" \
  --output "${VIOCF_RESULTS}/guidance_qc.csv"
viocf qc \
  --manifest "${VIOCF_ROOT}/manifests/sweep/steps_all.csv" \
  --output "${VIOCF_RESULTS}/steps_qc.csv"

viocf features \
  --manifest "${VIOCF_ROOT}/manifests/sweep/dense.csv" \
  --output "${VIOCF_RESULTS}/dense_features.csv" \
  --include-missing
viocf features \
  --manifest "${VIOCF_ROOT}/manifests/sweep/guidance_all.csv" \
  --output "${VIOCF_RESULTS}/guidance_features.csv" \
  --include-missing
viocf features \
  --manifest "${VIOCF_ROOT}/manifests/sweep/steps_all.csv" \
  --output "${VIOCF_RESULTS}/steps_features.csv" \
  --include-missing

python - \
  "${VIOCF_RESULTS}/dense_features.csv" \
  "${VIOCF_RESULTS}/guidance_features.csv" \
  "${VIOCF_RESULTS}/steps_features.csv" <<'PY'
import sys
from pathlib import Path

import pandas as pd

for value in sys.argv[1:]:
    path = Path(value)
    frame = pd.read_csv(path)
    missing = frame.get("feature_error", pd.Series(index=frame.index, dtype=object)).eq(
        "missing_audio"
    )
    if missing.any():
        raise SystemExit(f"{path}: {int(missing.sum())} audio files are missing")
PY

viocf train-surrogate \
  --features \
    "${VIOCF_RESULTS}/dense_features.csv" \
    "${VIOCF_RESULTS}/guidance_features.csv" \
    "${VIOCF_RESULTS}/steps_features.csv" \
  --output-dir "${VIOCF_RESULTS}/surrogate"

echo "Sweep analysis complete: ${VIOCF_RESULTS}"
