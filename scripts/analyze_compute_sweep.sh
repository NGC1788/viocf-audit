#!/usr/bin/env bash
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_RESULTS="${VIOCF_ROOT}/results/sweep"

# ⚠ GPU 생성 작업과 같이 돌 수 있으므로 워커를 제한한다.
# 기본값(코어수-2)으로 돌리면 코어를 다 먹어서 VIOLET 이 dataloader 와 오디오
# 인코딩을 못 하고 GPU 가 굶는다(실제로 겪음 — GPU 56% -> 32%).
# GPU 가 놀고 있으면 VIOCF_WORKERS 를 높여 주면 된다.
VIOCF_WORKERS="${VIOCF_WORKERS:-12}"
echo "스윕 분석 시작 — 워커 ${VIOCF_WORKERS}개 (GPU 작업과 병행 가능)"

if [[ ! -f "${VIOCF_ROOT}/.venv/bin/activate" ]]; then
  echo "Missing analysis environment. Run scripts/bootstrap_analysis.sh first."
  exit 2
fi
source "${VIOCF_ROOT}/.venv/bin/activate"
mkdir -p "${VIOCF_RESULTS}"

viocf qc \
  --manifest "${VIOCF_ROOT}/manifests/sweep/dense.csv" \
  --output "${VIOCF_RESULTS}/dense_qc.csv" \
  --workers "${VIOCF_WORKERS}"
viocf qc \
  --manifest "${VIOCF_ROOT}/manifests/sweep/guidance_all.csv" \
  --output "${VIOCF_RESULTS}/guidance_qc.csv" \
  --workers "${VIOCF_WORKERS}"
viocf qc \
  --manifest "${VIOCF_ROOT}/manifests/sweep/steps_all.csv" \
  --output "${VIOCF_RESULTS}/steps_qc.csv" \
  --workers "${VIOCF_WORKERS}"

viocf features \
  --manifest "${VIOCF_ROOT}/manifests/sweep/dense.csv" \
  --output "${VIOCF_RESULTS}/dense_features.csv" \
  --include-missing \
  --workers "${VIOCF_WORKERS}"
viocf features \
  --manifest "${VIOCF_ROOT}/manifests/sweep/guidance_all.csv" \
  --output "${VIOCF_RESULTS}/guidance_features.csv" \
  --include-missing \
  --workers "${VIOCF_WORKERS}"
viocf features \
  --manifest "${VIOCF_ROOT}/manifests/sweep/steps_all.csv" \
  --output "${VIOCF_RESULTS}/steps_features.csv" \
  --include-missing \
  --workers "${VIOCF_WORKERS}"

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
