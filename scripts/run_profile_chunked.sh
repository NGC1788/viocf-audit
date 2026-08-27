#!/usr/bin/env bash
# 프로파일 생성을 프롬프트 단위로 쪼개서 돌린다 (중간 재개 가능).
#
# 왜 필요한가: expanded 코어는 18,432 클립짜리 한 덩어리다. 클립당 2초면 10시간인데
# 9시간째에 죽으면 10시간을 통째로 다시 해야 한다. VS Code 터미널에서 돌리면
# 편집기 재시작·크래시만으로도 그 일이 벌어진다.
#
# 프롬프트마다 따로 돌리면 죽어도 20~30분치만 잃는다.
# 원본 MIDI 디렉터리 구조는 건드리지 않는다 — 프롬프트별 하드링크 디렉터리를
# 임시로 만들어 VIOCF_MIDI_DIR_OVERRIDE 로 넘긴다.
#
# 사용: scripts/run_profile_chunked.sh {pilot|full|expanded}
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_PROFILE="${1:-pilot}"
VIOCF_STAGE_DIR="${VIOCF_ROOT}/logs/chunks/${VIOCF_PROFILE}"
VIOCF_STATE="${VIOCF_ROOT}/logs/chunk_state_${VIOCF_PROFILE}.tsv"

case "${VIOCF_PROFILE}" in
  pilot | full | expanded) ;;
  *)
    echo "프로파일은 pilot, full, expanded 중 하나여야 한다."
    exit 2
    ;;
esac

mkdir -p "${VIOCF_STAGE_DIR}" "$(dirname "${VIOCF_STATE}")"
touch "${VIOCF_STATE}"

# 프롬프트별로 MIDI 를 모아 하드링크 디렉터리를 만들고 계획을 출력한다.
VIOCF_PLAN="$({ python3 - \
  "${VIOCF_ROOT}" "${VIOCF_PROFILE}" "${VIOCF_STAGE_DIR}" <<'PY'
from __future__ import annotations

import csv
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1]).resolve()
profile = sys.argv[2]
stage_root = Path(sys.argv[3]).resolve()

manifests = [
    root / "manifests" / f"{profile}_model.csv",
    root / "manifests" / f"{profile}_delayed_model.csv",
]
groups: dict[str, list[Path]] = defaultdict(list)
for manifest in manifests:
    if not manifest.is_file():
        continue
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"빈 manifest: {manifest}")
    for row in rows:
        # delayed 는 프롬프트가 하나뿐이라 별도 청크로 둔다
        key = row.get("prompt_id") or "unknown"
        if row.get("profile") == "delayed":
            key = f"{key}__delayed"
        path = Path(row["midi_path"])
        path = path if path.is_absolute() else root / path
        path = path.resolve()
        if not path.is_file():
            raise SystemExit(f"MIDI 없음: {path}")
        groups[key].append(path)

if not groups:
    raise SystemExit(f"manifest 를 찾지 못했다: {profile}")

for key, paths in sorted(groups.items()):
    if len(paths) != len(set(paths)):
        raise SystemExit(f"{key}: 중복 midi_path")
    stage = stage_root / key
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for path in paths:
        target = stage / path.name
        # ⚠ 심볼릭 링크를 쓰면 안 된다.
        # run_violet.sh 의 개수 확인이 `find -type f` 인데 심볼릭 링크는 -type l 이라
        # 세지 않는다 -> 768개를 만들어 두고 "No MIDI files found" 로 죽는다(실제로 겪음).
        # 하드 링크는 일반 파일과 구분되지 않으므로 하위 도구가 전부 그대로 동작한다.
        # 같은 파일시스템이 아니면(드묾) 복사로 물러난다.
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
    print(f"{key}\t{stage}\t{len(paths)}")
PY
} 2>&1)" || {
  echo "${VIOCF_PLAN}"
  exit 2
}

VIOCF_TOTAL="$(printf '%s\n' "${VIOCF_PLAN}" | grep -c .)"
echo "프로파일: ${VIOCF_PROFILE}"
echo "청크: ${VIOCF_TOTAL}개 (프롬프트 단위)"
echo

VIOCF_INDEX=0
while IFS=$'\t' read -r VIOCF_KEY VIOCF_MIDI_DIR VIOCF_COUNT; do
  [[ -n "${VIOCF_KEY}" ]] || continue
  VIOCF_INDEX=$((VIOCF_INDEX + 1))

  if awk -F'\t' -v key="${VIOCF_KEY}" \
    '$1 == key && $2 == "done" { found = 1 } END { exit !found }' \
    "${VIOCF_STATE}"; then
    echo "[${VIOCF_INDEX}/${VIOCF_TOTAL}] ${VIOCF_KEY} 이미 완료 — 건너뜀"
    continue
  fi

  VIOCF_JOB_RUN_ID="chunk_${VIOCF_KEY}"
  VIOCF_JOB_RUN_DIR="${VIOCF_ROOT}/logs/violet/${VIOCF_PROFILE}/${VIOCF_JOB_RUN_ID}"
  echo "[${VIOCF_INDEX}/${VIOCF_TOTAL}] ${VIOCF_KEY} — ${VIOCF_COUNT} 클립"

  VIOCF_STARTED="$(date +%s)"
  VIOCF_MIDI_DIR_OVERRIDE="${VIOCF_MIDI_DIR}" \
  VIOCF_RUN_ID="${VIOCF_JOB_RUN_ID}" \
    bash "${VIOCF_ROOT}/scripts/run_violet.sh" "${VIOCF_PROFILE}"

  bash "${VIOCF_ROOT}/scripts/verify_violet_run.sh" \
    "${VIOCF_JOB_RUN_DIR}" "${VIOCF_MIDI_DIR}"

  VIOCF_ELAPSED=$(( $(date +%s) - VIOCF_STARTED ))
  printf '%s\tdone\t%s\t%ss\t%s clips\n' \
    "${VIOCF_KEY}" "$(date '+%F %T')" "${VIOCF_ELAPSED}" "${VIOCF_COUNT}" \
    >>"${VIOCF_STATE}"
  echo "  완료 (${VIOCF_ELAPSED}s, 클립당 $(awk -v e="${VIOCF_ELAPSED}" -v c="${VIOCF_COUNT}" \
    'BEGIN { printf "%.2f", e / c }')s)"
  echo
done <<<"${VIOCF_PLAN}"

echo "프로파일 완료: ${VIOCF_PROFILE}"
echo "상태: ${VIOCF_STATE}"
