#!/usr/bin/env bash
set -euo pipefail

# ⚠ VIOLET 원본 README는 체크포인트를 datasets/ 주소로 링크하는데 그건 401이다.
# 실제 가중치는 models/ 레포에 **공개(게이팅 없음)** 로 올라와 있다. 2026-08-26 확인:
#   https://huggingface.co/api/datasets/User-tian/VIOLET -> 401
#   https://huggingface.co/api/models/User-tian/VIOLET   -> 200, gated=False
#     pretrained_checkpoint/ema_snapshots/ema_prof_99515  (581,281,166 bytes)
#     dacvae_ft/weights.pth                               (430,800,929 bytes)
VIOCF_REPO_API="https://huggingface.co/api/models/User-tian/VIOLET"
VIOCF_HEADERS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  VIOCF_HEADERS=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

VIOCF_STATUS="$(curl -sS -o /dev/null -w '%{http_code}' ${VIOCF_HEADERS[@]+"${VIOCF_HEADERS[@]}"} "${VIOCF_REPO_API}")"
echo "VIOLET Hugging Face API status: ${VIOCF_STATUS}"

if [[ "${VIOCF_STATUS}" == "200" ]]; then
  echo "Repository metadata is accessible (public, ungated)."
  echo
  echo "Download and SHA-256 verify the two required files with:"
  echo "  bash scripts/download_violet_checkpoints.sh"
  echo "The script writes to vendor/VIOLET/checkpoints, which is the runner's default."
  exit 0
fi

echo "models/ repo is unreachable (status ${VIOCF_STATUS}); this is unexpected as of 2026-08-26."
echo "Check the network first. Only if the repo itself went private should you contact the authors"
echo "using docs/VIOLET_ACCESS_REQUEST.txt."
echo "Do not start the full recording matrix until model inference is smoke-tested."
exit 2
