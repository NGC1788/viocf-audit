#!/usr/bin/env bash
# 결과 중 **작은 것만** 저장소에 남긴다 (드라이브가 또 죽어도 숫자는 산다).
#
# 왜: results/ 는 gitignore 다(특징표 18 MB, 오디오 32 GB — 전부 재생성 가능하니까).
# 그런데 2026-08-31 에 NVMe 가 dead 상태로 떨어지면서 40시간치 생성물을 잃을 뻔했다.
# 이번엔 ext4 셧다운 덕에 살았지만 다음은 모른다.
#
# 재생성 비용이 큰 것과 작은 것을 나눠 보면:
#   오디오 33,600개   ~32 GB   GPU 40시간   -> 저장소에 못 넣는다. 시드가 결정적이라 재생성 가능.
#   특징표             18 MB   CPU 15분    -> 안 넣는다. 오디오만 있으면 금방 나온다.
#   요약 JSON·그림      < 1 MB   위 전부 필요 -> **이건 넣는다.** 여기 헤드라인 숫자가 다 들어 있다.
#
# 즉 이 스냅샷은 "재생성 비용 대비 용량이 압도적으로 싼 것"만 고른다.
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_DEST="${VIOCF_ROOT}/docs/results_snapshot"
cd "${VIOCF_ROOT}"

# ⚠ 매번 새로 만든다. 이어 붙이면 안 된다.
# 대상 목록을 좁힌 뒤에도 **이전 실행이 복사해 둔 파일이 남아** 크기 검사에
# 걸렸다(40개만 복사했는데 24 MB 로 나왔다 — 옛 69개의 잔해가 같이 세어졌다).
# 이 디렉터리는 results/ 에서 파생되는 것이므로 통째로 다시 만드는 게 맞다.
rm -rf "${VIOCF_DEST}"
mkdir -p "${VIOCF_DEST}"

# 파일 하나당 상한. 이름만으로 고르면 '요약'이라는 이름의 큰 파일이 섞인다 —
# collect_*.summary.json 이 실제로 956 KB 였고 전체 상한에 걸려서 알았다.
# 크기로도 한 번 더 거르면 앞으로 무엇이 추가돼도 안전하다.
VIOCF_MAX_FILE_KB=512

VIOCF_COPIED=0
VIOCF_SKIPPED=""
while IFS= read -r VIOCF_SRC; do
  [[ -f "${VIOCF_SRC}" ]] || continue
  VIOCF_KB="$(du -k "${VIOCF_SRC}" | cut -f1)"
  if [[ "${VIOCF_KB}" -gt "${VIOCF_MAX_FILE_KB}" ]]; then
    VIOCF_SKIPPED="${VIOCF_SKIPPED}  ${VIOCF_SRC} (${VIOCF_KB} KB)
"
    continue
  fi
  VIOCF_REL="${VIOCF_SRC#results/}"
  VIOCF_OUT="${VIOCF_DEST}/${VIOCF_REL}"
  mkdir -p "$(dirname "${VIOCF_OUT}")"
  cp -p "${VIOCF_SRC}" "${VIOCF_OUT}"
  VIOCF_COPIED=$((VIOCF_COPIED + 1))
done < <(
  find results -type f \( \
    -name 'metrics_summary.json' -o \
    -name '*_qc.summary.json' -o \
    -name 'embedding_model_only_*.json' -o \
    -name 'fig*.png' -o \
    -name 'effect_alignment.csv' -o \
    -name 'excess_leakage.csv' -o \
    -name 'compositionality_gap.csv' -o \
    -name 'monotonicity.csv' -o \
    -name 'delayed_branch*.csv' -o \
    -name 'embedding_c2st.csv' \
  \) 2>/dev/null | sort
)

# 재개 상태 파일도 넣는다. 몇 KB 인데 이게 없으면 **완료한 조각을 다시 돌린다** —
# 지연 확장 18조각(GPU 10시간), 본체 25청크(GPU 23시간)의 진행 기록이다.
# 오디오 자체(40 GB)는 재생성 가능하지만 '어디까지 했나'는 여기에만 있다.
for VIOCF_STATE in logs/queue_state.tsv logs/chunk_state_*.tsv logs/delayed_sweep_state.tsv; do
  [[ -f "${VIOCF_STATE}" ]] || continue
  mkdir -p "${VIOCF_DEST}/state"
  cp -p "${VIOCF_STATE}" "${VIOCF_DEST}/state/$(basename "${VIOCF_STATE}")"
  VIOCF_COPIED=$((VIOCF_COPIED + 1))
done

# 크기를 확인한다. 저장소를 부풀리면 이 장치의 취지가 무너진다.
VIOCF_SIZE_KB="$(du -sk "${VIOCF_DEST}" | cut -f1)"
echo "스냅샷 ${VIOCF_COPIED}개 파일, ${VIOCF_SIZE_KB} KB -> docs/results_snapshot/"
if [[ -n "${VIOCF_SKIPPED}" ]]; then
  # 말없이 자르지 않는다 — 무엇이 빠졌는지 보이지 않으면 '전부 담았다'로 읽힌다.
  echo "크기 상한(${VIOCF_MAX_FILE_KB} KB)으로 제외:"
  printf '%s' "${VIOCF_SKIPPED}"
fi
if [[ "${VIOCF_SIZE_KB}" -gt 20480 ]]; then
  echo "⚠ 20 MB 를 넘었다. 무엇이 커졌는지 보고 대상 목록을 좁힐 것:"
  du -sk "${VIOCF_DEST}"/* | sort -rn | head -5
  exit 1
fi

# 언제 무엇을 찍었는지 남긴다. 나중에 '이 숫자가 어느 실행에서 나왔나'를 답해야 한다.
{
  echo "# 결과 스냅샷"
  echo
  echo "생성: $(date '+%F %T')"
  echo "커밋: $(git rev-parse --short HEAD)"
  echo "파일: ${VIOCF_COPIED}개 (${VIOCF_SIZE_KB} KB)"
  echo
  echo "재생성 비용이 큰 것(오디오 32 GB / GPU 40시간, 특징표 18 MB)은 넣지 않는다."
  echo "시드가 결정적이라 코드만 있으면 비트 단위로 동일하게 다시 만들 수 있다."
  echo "여기 있는 것은 그 전부가 있어야 나오는 최종 요약이다."
  echo
  echo "## 목록"
  echo
  (cd "${VIOCF_DEST}" && find . -type f ! -name 'README.md' | sort | sed 's|^\./|- |')
} > "${VIOCF_DEST}/README.md"

echo "다음: git add docs/results_snapshot && git commit && git push"
