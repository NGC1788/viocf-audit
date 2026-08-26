#!/usr/bin/env bash
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_PROFILE="${1:-pilot}"
VIOCF_SESSION="viocf_${VIOCF_PROFILE}"
VIOCF_LOG="${VIOCF_ROOT}/logs/${VIOCF_SESSION}.log"

if tmux has-session -t "${VIOCF_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${VIOCF_SESSION}"
  exit 2
fi

tmux new-session -d -s "${VIOCF_SESSION}" \
  "cd '${VIOCF_ROOT}' && bash scripts/run_violet.sh '${VIOCF_PROFILE}' 2>&1 | tee '${VIOCF_LOG}'"

echo "Started ${VIOCF_SESSION}. Attach with: tmux attach -t ${VIOCF_SESSION}"
echo "Log: ${VIOCF_LOG}"

