#!/usr/bin/env bash
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_PROFILE="${1:-pilot}"
VIOCF_VIOLET_DIR="${VIOCF_VIOLET_DIR:-${VIOCF_ROOT}/vendor/VIOLET}"
VIOCF_MIDI_DIR="${VIOCF_MIDI_DIR_OVERRIDE:-${VIOCF_ROOT}/data/midi/${VIOCF_PROFILE}/model}"
VIOCF_EMA="${VIOCF_VIOLET_EMA:-${VIOCF_VIOLET_DIR}/checkpoints/pretrained_checkpoint/ema_snapshots/ema_prof_99515}"
VIOCF_DACVAE="${VIOCF_DACVAE_CKPT:-${VIOCF_VIOLET_DIR}/checkpoints/dacvae_ft/weights.pth}"
VIOCF_BASE_SEED="${VIOCF_BASE_SEED:-20260826}"
VIOCF_RUN_ID="${VIOCF_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
VIOCF_RUN_DIR="${VIOCF_ROOT}/logs/violet/${VIOCF_PROFILE}/${VIOCF_RUN_ID}"
VIOCF_W_TECH="${VIOCF_W_TECH:-1.0}"
VIOCF_W_CC="${VIOCF_W_CC:-1.0}"
VIOCF_SAMPLER_STEPS="${VIOCF_SAMPLER_STEPS:-30}"
VIOCF_CUDA_VISIBLE_DEVICES="${VIOCF_CUDA_VISIBLE_DEVICES:-0}"
VIOCF_SAVE_DEBUG_BRANCHES="${VIOCF_SAVE_DEBUG_BRANCHES:-false}"

if [[ ! "${VIOCF_W_TECH}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  [[ ! "${VIOCF_W_CC}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "VIOCF_W_TECH and VIOCF_W_CC must be non-negative numbers."
  exit 2
fi
if [[ ! "${VIOCF_SAMPLER_STEPS}" =~ ^[0-9]+$ ]] ||
  [[ "${VIOCF_SAMPLER_STEPS}" -lt 2 ]]; then
  echo "VIOCF_SAMPLER_STEPS must be an integer of at least 2."
  exit 2
fi
if [[ "${VIOCF_SAVE_DEBUG_BRANCHES}" != "true" && "${VIOCF_SAVE_DEBUG_BRANCHES}" != "false" ]]; then
  echo "VIOCF_SAVE_DEBUG_BRANCHES must be true or false."
  exit 2
fi

if [[ ! -e "${VIOCF_VIOLET_DIR}/src/eval.py" ]]; then
  echo "Missing VIOLET checkout: ${VIOCF_VIOLET_DIR}"
  exit 2
fi
if ! grep -q "save_test_render_manifest" \
  "${VIOCF_VIOLET_DIR}/src/models/violin_diffusion_module.py"; then
  echo "VIOLET counterfactual patch is not applied."
  echo "Run: VIOCF_VIOLET_DIR=${VIOCF_VIOLET_DIR} bash scripts/prepare_violet_repo.sh"
  exit 2
fi
for VIOCF_REQUIRED in "${VIOCF_EMA}" "${VIOCF_DACVAE}"; do
  if [[ ! -e "${VIOCF_REQUIRED}" ]]; then
    echo "Missing required VIOLET asset: ${VIOCF_REQUIRED}"
    exit 2
  fi
done
if [[ ! -d "${VIOCF_MIDI_DIR}" ]]; then
  echo "Missing MIDI directory: ${VIOCF_MIDI_DIR}"
  exit 2
fi
# -L 로 심볼릭 링크를 따라간다. 없으면 링크된 MIDI 를 0개로 세어 "No MIDI files found" 로 죽는다.
VIOCF_MIDI_COUNT="$(find -L "${VIOCF_MIDI_DIR}" -type f \( -name '*.mid' -o -name '*.midi' \) | wc -l | tr -d ' ')"
if [[ "${VIOCF_MIDI_COUNT}" -eq 0 ]]; then
  echo "No MIDI files found under: ${VIOCF_MIDI_DIR}"
  exit 2
fi

if [[ -d "${VIOCF_RUN_DIR}" ]] &&
  [[ -n "$(find "${VIOCF_RUN_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]] &&
  [[ "${VIOCF_ALLOW_EXISTING_RUN_DIR:-false}" != "true" ]]; then
  echo "Run directory is not empty: ${VIOCF_RUN_DIR}"
  echo "Choose a new VIOCF_RUN_ID; reusing a partial directory can duplicate JSONL records."
  exit 2
fi

mkdir -p "${VIOCF_RUN_DIR}"
cd "${VIOCF_VIOLET_DIR}"
if [[ -f .venv-violet/bin/activate ]]; then
  source .venv-violet/bin/activate
fi

echo "Run directory: ${VIOCF_RUN_DIR}"
echo "MIDI directory: ${VIOCF_MIDI_DIR}"
echo "MIDI files: ${VIOCF_MIDI_COUNT}"
echo "Guidance: w_tech=${VIOCF_W_TECH}, w_cc=${VIOCF_W_CC}"
echo "Sampler steps: ${VIOCF_SAMPLER_STEPS}"
echo "Debug branch audio: ${VIOCF_SAVE_DEBUG_BRANCHES}"

CUDA_VISIBLE_DEVICES="${VIOCF_CUDA_VISIBLE_DEVICES}" python src/eval.py \
  experiment=violin_synthesis_inference/violin_synthesis_inference.yaml \
  data=eval_midi "data.data_dir=${VIOCF_MIDI_DIR}" \
  data.batch_size=1 data.num_workers=0 \
  "seed=${VIOCF_BASE_SEED}" \
  "sampler_steps=${VIOCF_SAMPLER_STEPS}" \
  "sampler_w_tech=${VIOCF_W_TECH}" "sampler_w_cc=${VIOCF_W_CC}" \
  '+model.test_noise_group_delimiter=__' \
  +model.test_max_render_attempts=1 \
  +model.save_test_render_manifest=true \
  "+model.save_test_conditioning_debug=${VIOCF_SAVE_DEBUG_BRANCHES}" \
  +trainer.precision=32 long_audio.enabled=false \
  "model.ema_ckpt_path=${VIOCF_EMA}" \
  "encoder.finetuned_ckpt=${VIOCF_DACVAE}" \
  logger=csv "hydra.run.dir=${VIOCF_RUN_DIR}"

echo "VIOCF_RUN_DIR=${VIOCF_RUN_DIR}"
