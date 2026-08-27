#!/usr/bin/env bash
# GPU 가 놀지 않게 하는 우선순위 큐.
#
# 한 달짜리 고정 계획을 짜면 중간에 뭐 하나 막힐 때 GPU 가 논다. 대신 우선순위 큐를
# 앞에서부터 계속 뽑아 쓰고, **언제 끊겨도 그때까지가 완결된 결과집합**이 되도록 순서를 짠다.
#
#   T0 게이트   smoke -> pilot        : 짝이 맞는지. 실패하면 뒤는 전부 무의미하므로 여기서 멈춘다.
#   T1 본체     expanded 코어/delayed : 지표 3개가 나오는 최소 완결 집합.
#   T2~T4 확장  sweep(dense/guidance/steps)
#   T5 임베딩   MERT 표현 추출        : 해석가능 특징과 독립인 학습된 표현으로 교차확인.
#                                       GPU 작업이고 클립 수에 비례해 오래 돈다.
#   T6 분석     특징 -> 지표 -> 그림   : CPU. GPU 를 비우고 돌린다.
#
# 재개: 완료된 잡은 verify_violet_run.sh 로 확인하고 건너뛴다. 죽어도 다시 띄우면 이어간다.
# 진행률: 잡마다 실제 소요시간을 기록해 남은 시간을 계속 다시 추정한다.
#
# 사용:
#   tmux new -s viocf_queue 'bash scripts/run_queue.sh'      # 권장
#   VIOCF_QUEUE_FROM=T1 bash scripts/run_queue.sh            # 게이트 건너뛰고 재개
#   VIOCF_QUEUE_DRY_RUN=true bash scripts/run_queue.sh       # 계획만 출력
#
# 실행 위치에 대하여:
#   RustDesk 로 서버 데스크톱을 제어한다면 VS Code 통합 터미널에서 돌려도 된다.
#   SSH 와 달리 원격 연결이 끊겨도 데스크톱 세션은 서버 안에서 살아 있기 때문이다.
#   다만 VS Code 자동 업데이트·크래시·서버 재부팅에는 죽는다.
#   그래서 T1(가장 긴 단계)을 프롬프트 단위로 쪼개 두었다 — 죽어도 최대 26분치만 잃고,
#   다시 실행하면 끝난 청크는 건너뛴다.
#   더 안전하게 하려면 tmux 안에서 돌리면 된다: tmux new -s q 'bash scripts/run_queue.sh' 
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_QUEUE_FROM="${VIOCF_QUEUE_FROM:-T0}"
VIOCF_QUEUE_DRY_RUN="${VIOCF_QUEUE_DRY_RUN:-false}"
VIOCF_QUEUE_PROFILE="${VIOCF_QUEUE_PROFILE:-expanded}"
VIOCF_QUEUE_STATE="${VIOCF_ROOT}/logs/queue_state.tsv"
VIOCF_QUEUE_LOG="${VIOCF_ROOT}/logs/queue.log"

if [[ "${VIOCF_QUEUE_DRY_RUN}" != "true" && "${VIOCF_QUEUE_DRY_RUN}" != "false" ]]; then
  echo "VIOCF_QUEUE_DRY_RUN must be true or false."
  exit 2
fi
if [[ "${VIOCF_QUEUE_PROFILE}" != "full" && "${VIOCF_QUEUE_PROFILE}" != "expanded" ]]; then
  echo "VIOCF_QUEUE_PROFILE must be full or expanded."
  exit 2
fi

mkdir -p "$(dirname "${VIOCF_QUEUE_STATE}")"
touch "${VIOCF_QUEUE_STATE}"

# 단계 정의: 티어 <TAB> 종류 <TAB> 인자 <TAB> 설명
# 종류 profile = scripts/run_violet.sh <인자>
# 종류 sweep   = scripts/run_compute_sweep.sh (VIOCF_SWEEP_PHASE=<인자>)
VIOCF_STAGES="$(
  cat <<STAGES
T0	profile	smoke	게이트: 같은 그룹 p/f 두 개의 seed 가 같은지
T0	profile	pilot	게이트: 60클립 파일럿, pairing_pass 전수 확인
T1	chunked	${VIOCF_QUEUE_PROFILE}	본체: 코어 factorial + delayed (프롬프트 단위 재개)
T2	sweep	dense	CC1 응답곡선
T3	sweep	guidance	guidance 4x4 격자 (16잡)
T4	sweep	steps	확산 스텝 6단계 (6잡)
T4b	collect	sweep	스윕 오디오 수집 (생성 -> data/model_audio)
T5	embed	${VIOCF_QUEUE_PROFILE}	MERT 임베딩 추출 (GPU)
T5	embed	sweep	MERT 임베딩 추출 — 스윕 (GPU)
T6	analyze	${VIOCF_QUEUE_PROFILE}	특징 추출 -> 지표 -> 그림 (CPU)
STAGES
)"

log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "${VIOCF_QUEUE_LOG}"; }

# grep -P 는 macOS 기본 grep 에 없다. awk 로 필드 비교해서 이식성을 지킨다.
stage_done() {
  awk -F'\t' -v key="$1" '$1 == key && $2 == "done" { found = 1 } END { exit !found }' \
    "${VIOCF_QUEUE_STATE}" 2>/dev/null
}

mark_done() {
  printf '%s\tdone\t%s\t%s\n' "$1" "$(date '+%F %T')" "$2" >>"${VIOCF_QUEUE_STATE}"
}

tier_at_or_after() {
  # 문자열 비교로 충분하다 (T0..T9)
  [[ "$1" > "${VIOCF_QUEUE_FROM}" || "$1" == "${VIOCF_QUEUE_FROM}" ]]
}

log "=============================================================="
log "VioCF 큐 시작  profile=${VIOCF_QUEUE_PROFILE}  from=${VIOCF_QUEUE_FROM}  dry_run=${VIOCF_QUEUE_DRY_RUN}"
log "상태파일: ${VIOCF_QUEUE_STATE}"
log "=============================================================="

VIOCF_QUEUE_START="$(date +%s)"
VIOCF_STAGE_INDEX=0
VIOCF_STAGE_TOTAL="$(printf '%s\n' "${VIOCF_STAGES}" | grep -c .)"

while IFS=$'\t' read -r VIOCF_TIER VIOCF_KIND VIOCF_ARG VIOCF_DESC; do
  [[ -n "${VIOCF_TIER}" ]] || continue
  VIOCF_STAGE_INDEX=$((VIOCF_STAGE_INDEX + 1))
  VIOCF_KEY="${VIOCF_TIER}:${VIOCF_KIND}:${VIOCF_ARG}"

  if ! tier_at_or_after "${VIOCF_TIER}"; then
    log "[${VIOCF_STAGE_INDEX}/${VIOCF_STAGE_TOTAL}] ${VIOCF_KEY} 건너뜀 (VIOCF_QUEUE_FROM=${VIOCF_QUEUE_FROM})"
    continue
  fi
  if stage_done "${VIOCF_KEY}"; then
    log "[${VIOCF_STAGE_INDEX}/${VIOCF_STAGE_TOTAL}] ${VIOCF_KEY} 이미 완료 — 건너뜀"
    continue
  fi

  log ""
  log "[${VIOCF_STAGE_INDEX}/${VIOCF_STAGE_TOTAL}] ${VIOCF_KEY}  — ${VIOCF_DESC}"

  if [[ "${VIOCF_QUEUE_DRY_RUN}" == "true" ]]; then
    case "${VIOCF_KIND}" in
      profile) log "  DRY RUN: scripts/run_violet.sh ${VIOCF_ARG}" ;;
      sweep) log "  DRY RUN: VIOCF_SWEEP_PHASE=${VIOCF_ARG} scripts/run_compute_sweep.sh" ;;
    esac
    continue
  fi

  VIOCF_STAGE_START="$(date +%s)"
  VIOCF_STAGE_OK=true
  case "${VIOCF_KIND}" in
    profile)
      VIOCF_RUN_ID="queue_$(date +%Y%m%d_%H%M%S)"
      VIOCF_RUN_ID="${VIOCF_RUN_ID}" bash "${VIOCF_ROOT}/scripts/run_violet.sh" "${VIOCF_ARG}" \
        || VIOCF_STAGE_OK=false
      if [[ "${VIOCF_STAGE_OK}" == "true" ]]; then
        bash "${VIOCF_ROOT}/scripts/verify_violet_run.sh" \
          "${VIOCF_ROOT}/logs/violet/${VIOCF_ARG}/${VIOCF_RUN_ID}" \
          "${VIOCF_ROOT}/data/midi/${VIOCF_ARG}/model" \
          || VIOCF_STAGE_OK=false
      fi
      ;;
    chunked)
      # 프롬프트 단위로 쪼개 돌린다. 죽어도 20~30분치만 잃는다.
      bash "${VIOCF_ROOT}/scripts/run_profile_chunked.sh" "${VIOCF_ARG}" \
        || VIOCF_STAGE_OK=false
      ;;
    sweep)
      VIOCF_SWEEP_PHASE="${VIOCF_ARG}" bash "${VIOCF_ROOT}/scripts/run_compute_sweep.sh" \
        || VIOCF_STAGE_OK=false
      ;;
    collect)
      # 생성만 하고 수집을 안 하면 오디오가 logs/violet 안에 남아
      # 뒤 단계가 전부 빈 결과를 낸다. 큐에 명시적 단계로 넣는다.
      bash "${VIOCF_ROOT}/scripts/collect_compute_sweep.sh" \
        || VIOCF_STAGE_OK=false
      ;;
    embed)
      # 학습된 표현(MERT)으로 교차확인한다. 해석가능 특징이 놓친 누출이 있는지,
      # 그리고 우리가 본 누출이 표현공간에서도 보이는지. GPU 를 쓴다.
      bash "${VIOCF_ROOT}/scripts/run_embeddings.sh" "${VIOCF_ARG}" \
        || VIOCF_STAGE_OK=false
      ;;
    analyze)
      # CPU 단계. GPU 가 비므로 다음 생성 잡과 겹쳐 돌려도 된다.
      bash "${VIOCF_ROOT}/scripts/run_pilot_analysis.sh" || VIOCF_STAGE_OK=false
      ;;
    *)
      log "  알 수 없는 종류: ${VIOCF_KIND}"
      exit 2
      ;;
  esac
  VIOCF_STAGE_SECONDS=$(( $(date +%s) - VIOCF_STAGE_START ))

  if [[ "${VIOCF_STAGE_OK}" != "true" ]]; then
    log "  실패: ${VIOCF_KEY} (${VIOCF_STAGE_SECONDS}s)"
    if [[ "${VIOCF_TIER}" == "T0" ]]; then
      log ""
      log "  ⚠ 게이트 단계가 실패했다. 여기서 멈춘다."
      log "    T0 는 '같은 noise group 안에서 seed 가 하나인가'를 확인하는 단계다."
      log "    이게 깨진 채로 뒤를 돌리면 짝 실험이 성립하지 않아 전부 버리게 된다."
      log "    logs/violet/${VIOCF_ARG}/ 아래 render manifest 를 먼저 확인할 것."
      exit 3
    fi
    log "  게이트가 아니므로 다음 단계로 넘어간다. 나중에 이 단계만 다시 돌리면 된다."
    continue
  fi

  mark_done "${VIOCF_KEY}" "${VIOCF_STAGE_SECONDS}s"
  VIOCF_ELAPSED=$(( $(date +%s) - VIOCF_QUEUE_START ))
  log "  완료 (${VIOCF_STAGE_SECONDS}s).  큐 누적 $(( VIOCF_ELAPSED / 3600 ))h $(( (VIOCF_ELAPSED % 3600) / 60 ))m"
done <<<"${VIOCF_STAGES}"

log ""
log "=============================================================="
log "큐 종료. 총 $(( ( $(date +%s) - VIOCF_QUEUE_START ) / 3600 ))시간"
log "다음: bash scripts/collect_compute_sweep.sh 로 오디오 수집 후 분석"
log "=============================================================="
