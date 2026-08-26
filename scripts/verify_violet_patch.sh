#!/usr/bin/env bash
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_VIOLET_DIR="${VIOCF_VIOLET_DIR:-${VIOCF_ROOT}/vendor/VIOLET}"
VIOCF_PATCH="${VIOCF_ROOT}/patches/violet_counterfactual_noise.patch"

if [[ ! -d "${VIOCF_VIOLET_DIR}/.git" ]]; then
  echo "Missing VIOLET checkout: ${VIOCF_VIOLET_DIR}"
  exit 2
fi

if git -C "${VIOCF_VIOLET_DIR}" apply --reverse --check "${VIOCF_PATCH}" >/dev/null 2>&1; then
  echo "PASS: patch is already applied cleanly."
elif git -C "${VIOCF_VIOLET_DIR}" apply --check "${VIOCF_PATCH}" >/dev/null 2>&1; then
  echo "PASS: patch applies cleanly to this checkout."
else
  echo "FAIL: patch neither applies nor appears already applied."
  exit 2
fi

