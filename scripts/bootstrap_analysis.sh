#!/usr/bin/env bash
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_PYTHON="${VIOCF_PYTHON:-python3.11}"

if ! command -v "${VIOCF_PYTHON}" >/dev/null 2>&1; then
  VIOCF_PYTHON=python3
fi

cd "${VIOCF_ROOT}"
if [[ ! -d .venv ]]; then
  "${VIOCF_PYTHON}" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev,f0]'

viocf preflight --output results/preflight.json
viocf make-design --profile pilot
viocf make-smoke
viocf inspect-midi data/midi/pilot
viocf inspect-midi data/midi/smoke

python - <<'PY'
from swift_f0 import SwiftF0

detector = SwiftF0(fmin=196.0, fmax=2093.75, confidence_threshold=0.75)
print("SwiftF0 paper-analysis backend ready:", type(detector).__name__)
PY

echo
echo "Analysis environment ready: ${VIOCF_ROOT}/.venv"
echo "Next: read TODAY.md and request the VIOLET checkpoint before full recording."
