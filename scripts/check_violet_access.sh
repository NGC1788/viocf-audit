#!/usr/bin/env bash
set -euo pipefail

VIOCF_REPO_API="https://huggingface.co/api/datasets/User-tian/VIOLET"
VIOCF_HEADERS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  VIOCF_HEADERS=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

VIOCF_STATUS="$(curl -sS -o /dev/null -w '%{http_code}' "${VIOCF_HEADERS[@]}" "${VIOCF_REPO_API}")"
echo "VIOLET Hugging Face API status: ${VIOCF_STATUS}"

if [[ "${VIOCF_STATUS}" == "200" ]]; then
  echo "Repository metadata is accessible. Confirm that both EMA and DACVAE files can be downloaded."
  exit 0
fi

echo "Access is not available yet. Log into Hugging Face/request access and contact the authors."
echo "Do not start the full recording matrix until model inference is smoke-tested."
exit 2

