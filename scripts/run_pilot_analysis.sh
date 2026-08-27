#!/usr/bin/env bash
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${VIOCF_ROOT}"
source .venv/bin/activate

viocf qc --manifest manifests/pilot_model.csv --output results/pilot_model_qc.csv
viocf qc --manifest manifests/pilot_real.csv --output results/pilot_real_qc.csv
viocf qc --manifest manifests/pilot_delayed_model.csv --output results/pilot_delayed_model_qc.csv
viocf qc --manifest manifests/pilot_delayed_real.csv --output results/pilot_delayed_real_qc.csv
viocf features --manifest manifests/pilot_model.csv --output results/pilot_model_features.csv
viocf features --manifest manifests/pilot_real.csv --output results/pilot_real_features.csv
viocf features --manifest manifests/pilot_delayed_model.csv --output results/pilot_delayed_model_features.csv
viocf features --manifest manifests/pilot_delayed_real.csv --output results/pilot_delayed_real_features.csv
viocf metrics \
  --features \
    results/pilot_model_features.csv \
    results/pilot_real_features.csv \
    results/pilot_delayed_model_features.csv \
    results/pilot_delayed_real_features.csv \
  --output-dir results/pilot_metrics

echo "Pilot analysis complete: ${VIOCF_ROOT}/results/pilot_metrics"
