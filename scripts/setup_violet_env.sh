#!/usr/bin/env bash
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_VIOLET_DIR="${VIOCF_VIOLET_DIR:-${VIOCF_ROOT}/vendor/VIOLET}"
VIOCF_TORCH_INDEX_URL="${VIOCF_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [[ ! -d "${VIOCF_VIOLET_DIR}/.git" ]]; then
  echo "Run scripts/prepare_violet_repo.sh first."
  exit 2
fi
# VIOLET 은 Python 3.10 환경을 요구하는데 서버 기본 파이썬은 3.13(conda) 이다.
# uv 는 필요한 파이썬 버전을 알아서 받아오고, **sudo 없이** ~/.local/bin 에 설치된다.
# 없으면 여기서 직접 깐다 — 사용자가 문서를 찾아 헤매게 하지 않는다.
if ! command -v uv >/dev/null 2>&1; then
  echo "uv 가 없다. 설치한다 (sudo 불필요, ~/.local/bin)."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # 설치 직후엔 PATH 에 아직 없으므로 이번 셸에 직접 추가한다
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  hash -r 2>/dev/null || true
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv 설치에 실패했다. 수동 설치: https://docs.astral.sh/uv/"
  echo "설치 후 새 터미널을 열거나 다음을 실행: export PATH=\"\${HOME}/.local/bin:\${PATH}\""
  exit 2
fi
echo "uv: $(command -v uv) ($(uv --version 2>/dev/null))"

cd "${VIOCF_VIOLET_DIR}"
if [[ ! -d .venv-violet ]]; then
  uv venv .venv-violet --python 3.10
fi
source .venv-violet/bin/activate

uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url "${VIOCF_TORCH_INDEX_URL}"
uv pip install -r requirements.txt
uv pip install git+https://github.com/facebookresearch/dacvae.git
uv pip install --upgrade 'protobuf>=4.25,<6'

python - <<'PY'
import torch
print("torch", torch.__version__)
print("CUDA available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))
    print("VRAM GiB", torch.cuda.get_device_properties(0).total_memory / 2**30)
PY

echo "VIOLET environment ready. Do not continue unless CUDA available is True."

