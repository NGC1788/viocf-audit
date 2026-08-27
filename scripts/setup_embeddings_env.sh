#!/usr/bin/env bash
# MERT 임베딩 전용 가상환경.
#
# 분석용 .venv 와 분리하는 이유:
#   - MERT 는 torch >= 2.6 을 요구한다(safetensors 미제공 + CVE-2025-32434 로
#     transformers 가 구버전 torch.load 를 거부). 분석 venv 에 torch 를 넣지 않는다는
#     기존 방침과 충돌한다.
#   - VIOLET venv 와도 분리한다. VIOLET 은 자기 torch 버전에 고정돼 있고
#     거기에 transformers 를 얹으면 의존성이 꼬인다.
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_EMBED_VENV="${VIOCF_ROOT}/.venv-embed"

if [[ ! -d "${VIOCF_EMBED_VENV}" ]]; then
  python3 -m venv "${VIOCF_EMBED_VENV}"
fi
"${VIOCF_EMBED_VENV}/bin/pip" install --upgrade pip
# ⚠ transformers 는 4.x 로 **상한을 건다**. MERT 의 원격 모델 코드는 4.x 시절 것이라
#   5.x 에서는 forward 가 output_hidden_states 인자를 받기만 하고 채우지 않아
#   hidden_states 가 None 으로 온다(실측 확인, 4.57.6 정상 / 5.16.1 실패).
#   embeddings.py 에 forward hook 대비책이 있지만 정상 경로를 쓰는 편이 안전하다.
"${VIOCF_EMBED_VENV}/bin/pip" install "torch>=2.6" "transformers>=4.40,<5" \
  librosa soundfile pandas numpy scikit-learn
"${VIOCF_EMBED_VENV}/bin/pip" install -e "${VIOCF_ROOT}"

echo
"${VIOCF_EMBED_VENV}/bin/python" - <<'PY'
import torch
import transformers
print(f"torch {torch.__version__}")
print(f"transformers {transformers.__version__}")
if int(transformers.__version__.split(".")[0]) >= 5:
    raise SystemExit(
        "transformers 5.x 에서는 MERT 가 hidden_states 를 반환하지 않는다. "
        "pip install 'transformers<5' 로 내릴 것."
    )
print(f"CUDA available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device {torch.cuda.get_device_name(0)}")
version = tuple(int(part) for part in torch.__version__.split(".")[:2])
if version < (2, 6):
    raise SystemExit("torch >= 2.6 이 필요하다 (MERT 로드 조건)")
PY

echo
echo "임베딩 환경 준비 완료: ${VIOCF_EMBED_VENV}"
echo "모델은 첫 실행 때 내려받는다(MERT-v1-95M 약 0.4 GB)."
echo "오프라인 서버면 미리 캐시할 것:"
echo "  HF_HOME=... ${VIOCF_EMBED_VENV}/bin/python -c \\"
echo "    \"from transformers import AutoModel; AutoModel.from_pretrained('m-a-p/MERT-v1-95M', trust_remote_code=True)\""
