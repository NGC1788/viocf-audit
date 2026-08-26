#!/usr/bin/env bash
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_VIOLET_DIR="${VIOCF_VIOLET_DIR:-${VIOCF_ROOT}/vendor/VIOLET}"
VIOCF_EXPECTED_COMMIT="cf0975a752a7ee3cc6e11bb573f9e47c64a0ef97"
VIOCF_PATCH="${VIOCF_ROOT}/patches/violet_counterfactual_noise.patch"

mkdir -p "$(dirname "${VIOCF_VIOLET_DIR}")"
if [[ ! -d "${VIOCF_VIOLET_DIR}/.git" ]]; then
  git clone https://github.com/User-tian/VIOLET.git "${VIOCF_VIOLET_DIR}"
fi

VIOCF_ACTUAL_COMMIT="$(git -C "${VIOCF_VIOLET_DIR}" rev-parse HEAD)"
echo "VIOLET commit: ${VIOCF_ACTUAL_COMMIT}"
if [[ "${VIOCF_ACTUAL_COMMIT}" != "${VIOCF_EXPECTED_COMMIT}" ]]; then
  echo "WARNING: patch was verified on ${VIOCF_EXPECTED_COMMIT}; checking compatibility with current HEAD."
fi

if git -C "${VIOCF_VIOLET_DIR}" apply --reverse --check "${VIOCF_PATCH}" >/dev/null 2>&1; then
  echo "Counterfactual-noise patch is already applied."
elif git -C "${VIOCF_VIOLET_DIR}" apply --check "${VIOCF_PATCH}"; then
  git -C "${VIOCF_VIOLET_DIR}" apply "${VIOCF_PATCH}"
  echo "Applied counterfactual-noise patch."
else
  echo "ERROR: patch does not apply cleanly. Preserve the repository and inspect upstream changes."
  exit 2
fi

git -C "${VIOCF_VIOLET_DIR}" status --short

