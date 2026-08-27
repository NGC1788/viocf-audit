#!/usr/bin/env bash
# 생성된 오디오에서 MERT 임베딩을 뽑는다 (GPU).
#
# 사용: scripts/run_embeddings.sh {pilot|full|expanded|sweep}
#
# 재개 가능: 출력 CSV 에 이미 있는 clip_id 는 건너뛴다. 20만 클립 규모라
# 중간에 죽는 것을 전제로 짰다.
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_TARGET="${1:-pilot}"
VIOCF_EMBED_VENV="${VIOCF_ROOT}/.venv-embed"
VIOCF_OUT="${VIOCF_ROOT}/results/embeddings"
VIOCF_MODEL_ID="${VIOCF_MERT_MODEL:-m-a-p/MERT-v1-95M}"

if [[ ! -x "${VIOCF_EMBED_VENV}/bin/viocf" ]]; then
  echo "임베딩 환경이 없다. 먼저 실행: bash scripts/setup_embeddings_env.sh"
  exit 2
fi
mkdir -p "${VIOCF_OUT}"

VIOCF_MANIFESTS=()
case "${VIOCF_TARGET}" in
  pilot | full | expanded)
    for name in "${VIOCF_TARGET}_model" "${VIOCF_TARGET}_real" \
      "${VIOCF_TARGET}_delayed_model" "${VIOCF_TARGET}_delayed_real"; do
      [[ -f "${VIOCF_ROOT}/manifests/${name}.csv" ]] &&
        VIOCF_MANIFESTS+=("${VIOCF_ROOT}/manifests/${name}.csv")
    done
    ;;
  sweep)
    for path in "${VIOCF_ROOT}"/manifests/sweep/dense.csv \
      "${VIOCF_ROOT}"/manifests/sweep/guidance_all.csv \
      "${VIOCF_ROOT}"/manifests/sweep/steps_all.csv; do
      [[ -f "${path}" ]] && VIOCF_MANIFESTS+=("${path}")
    done
    ;;
  *)
    echo "대상은 pilot, full, expanded, sweep 중 하나여야 한다."
    exit 2
    ;;
esac

if [[ "${#VIOCF_MANIFESTS[@]}" -eq 0 ]]; then
  echo "manifest 를 찾지 못했다: ${VIOCF_TARGET}"
  exit 2
fi

echo "MERT 임베딩 추출 — ${VIOCF_TARGET}"
echo "모델: ${VIOCF_MODEL_ID}"
echo "manifest ${#VIOCF_MANIFESTS[@]}개"

for manifest in "${VIOCF_MANIFESTS[@]}"; do
  stem="$(basename "${manifest}" .csv)"
  echo
  echo "[${stem}]"
  "${VIOCF_EMBED_VENV}/bin/viocf" embed \
    --manifest "${manifest}" \
    --output "${VIOCF_OUT}/${stem}_mert.csv" \
    --model-id "${VIOCF_MODEL_ID}"
done

echo
echo "임베딩 대조(C2ST) 계산"
VIOCF_EMBED_FILES=("${VIOCF_OUT}"/*_mert.csv)
if [[ -f "${VIOCF_EMBED_FILES[0]}" ]]; then
  "${VIOCF_EMBED_VENV}/bin/viocf" embed-metrics \
    --embeddings "${VIOCF_EMBED_FILES[@]}" \
    --output "${VIOCF_ROOT}/results/embedding_c2st.csv"
fi

echo "완료: ${VIOCF_OUT}"
