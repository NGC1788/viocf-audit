#!/usr/bin/env bash
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_VIOLET_DIR="${VIOCF_VIOLET_DIR:-${VIOCF_ROOT}/vendor/VIOLET}"
VIOCF_TORCH_INDEX_URL="${VIOCF_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [[ ! -d "${VIOCF_VIOLET_DIR}/.git" ]]; then
  echo "Run scripts/prepare_violet_repo.sh first."
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/"
  exit 2
fi

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

