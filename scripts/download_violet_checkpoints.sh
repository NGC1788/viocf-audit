#!/usr/bin/env bash
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_VIOLET_DIR="${VIOCF_VIOLET_DIR:-${VIOCF_ROOT}/vendor/VIOLET}"
VIOCF_HF_BASE="https://huggingface.co/User-tian/VIOLET/resolve/main"

# 체크포인트는 VIOLET 체크아웃 안(vendor/VIOLET/checkpoints)에 놓아야 runner 가 찾는다.
# 그래서 clone 이 먼저다. 없으면 여기서 직접 돌려준다 — 순서를 헷갈려 다시 치는 일을 없앤다.
if [[ ! -d "${VIOCF_VIOLET_DIR}/.git" ]]; then
  echo "VIOLET 체크아웃이 없다. prepare_violet_repo.sh 를 먼저 실행한다."
  bash "${VIOCF_ROOT}/scripts/prepare_violet_repo.sh"
  echo
fi
if [[ ! -d "${VIOCF_VIOLET_DIR}/.git" ]]; then
  echo "VIOLET 체크아웃 생성 실패: ${VIOCF_VIOLET_DIR}"
  exit 2
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required."
  exit 2
fi

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "Neither sha256sum nor shasum is available." >&2
    return 2
  fi
}

download_one() {
  local file_rel="$1"
  local expected_bytes="$2"
  local expected_sha="$3"
  local destination="${VIOCF_VIOLET_DIR}/checkpoints/${file_rel}"
  local partial="${destination}.part"
  local actual_bytes
  local actual_sha

  mkdir -p "$(dirname "${destination}")"
  if [[ -f "${destination}" ]]; then
    actual_bytes="$(wc -c < "${destination}" | tr -d ' ')"
    actual_sha="$(sha256_file "${destination}")"
    if [[ "${actual_bytes}" == "${expected_bytes}" && "${actual_sha}" == "${expected_sha}" ]]; then
      echo "Verified existing checkpoint: ${destination}"
      return 0
    fi
    echo "Existing file failed size/SHA-256 validation: ${destination}"
    echo "Move it aside explicitly, then rerun this script; it will not overwrite it."
    return 2
  fi

  echo "Downloading ${file_rel} (resume enabled)..."
  curl --fail --location --retry 5 --retry-all-errors \
    --continue-at - --output "${partial}" "${VIOCF_HF_BASE}/${file_rel}"
  actual_bytes="$(wc -c < "${partial}" | tr -d ' ')"
  actual_sha="$(sha256_file "${partial}")"
  if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
    echo "Size mismatch for ${partial}: ${actual_bytes} != ${expected_bytes}"
    return 2
  fi
  if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    echo "SHA-256 mismatch for ${partial}: ${actual_sha} != ${expected_sha}"
    return 2
  fi
  mv "${partial}" "${destination}"
  echo "Verified: ${destination}"
}

# Values are Hugging Face's x-linked-size and x-linked-etag (SHA-256),
# independently checked on 2026-08-27. Total download is about 1.01 GB.
download_one \
  "pretrained_checkpoint/ema_snapshots/ema_prof_99515" \
  "581281166" \
  "5e33833e68459115a96b2b696e709a117fcd7abf6556b18d003d39875e7a7262"
download_one \
  "dacvae_ft/weights.pth" \
  "430800929" \
  "3d1e57c2dd75ae8c40ab6a26888e941ca542d9ed4acb955c002e3396e2a14663"

echo "Both VIOLET checkpoints are ready under ${VIOCF_VIOLET_DIR}/checkpoints."
