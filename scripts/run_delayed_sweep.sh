#!/usr/bin/env bash
# 지연 분기 확장 실행 — 분기 오프셋 x 주법부류 x CC 유도 강도.
#
# 왜: 기존 지연 실험은 점 하나였다(오프셋 0.25 s, 주법 2개, w_cc 1.0).
# 세 축으로 넓혀 다음에 답한다.
#   1) 거리가 멀어져도 새는가 (오프셋 0.25 ~ 3.00 s)
#   2) '줄을 놓았는가'가 원인인가 (활이 남는 주법 3 vs 풀리는 주법 3)
#   3) 손잡이를 세게 돌리면 나아지나 나빠지나 (w_cc 0.0 / 1.0 / 2.0)
#
# w_cc 는 샘플러 설정이라 값마다 별도 실행이 필요하다. 그래서 manifest 를
# w_cc 별로 나눠 쓰고 여기서 순회한다.
#
# 재개 가능: logs/delayed_sweep_state.tsv 에 완료한 (w_cc, 주법) 조각을 남긴다.
# 오프셋·강약·반복은 한 조각 안에서 함께 돌아간다.
#
# 사용: scripts/run_delayed_sweep.sh [replicates]
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_REPLICATES="${1:-32}"
VIOCF_STATE="${VIOCF_ROOT}/logs/delayed_sweep_state.tsv"
VIOCF_PROFILE="delayed_sweep"

cd "${VIOCF_ROOT}"
if [[ ! -f .venv/bin/activate ]]; then
  echo "분석 환경이 없다. 먼저: bash scripts/bootstrap_analysis.sh"
  exit 2
fi
source .venv/bin/activate
mkdir -p "$(dirname "${VIOCF_STATE}")"
touch "${VIOCF_STATE}"

echo "=============================================================="
echo "지연 분기 확장  replicates=${VIOCF_REPLICATES}"
echo "=============================================================="
viocf make-delayed-sweep --replicates "${VIOCF_REPLICATES}"

# 조각 = (w_cc 태그, 주법). 주법 단위로 쪼개야 죽어도 잃는 시간이 짧다.
VIOCF_PLAN="$(python3 - "${VIOCF_ROOT}" <<'PY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
manifest_dir = root / "manifests" / "delayed_sweep"
for manifest in sorted(manifest_dir.glob("wc*.csv")):
    tag = manifest.stem
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"빈 manifest: {manifest}")
    by_technique = defaultdict(list)
    for row in rows:
        by_technique[row["technique"]].append(row)
    w_cc = rows[0]["w_cc"]
    w_tech = rows[0]["w_tech"]
    steps = rows[0]["sampling_steps"]
    for technique, group in sorted(by_technique.items()):
        # 조각별 MIDI 디렉터리와 manifest 를 따로 만든다.
        # (전체 manifest 를 collect 에 넘기면 all_pass 가 항상 false 가 된다 —
        #  run_profile_chunked.sh 에서 겪은 것과 같은 함정)
        stage = root / "logs" / "chunks" / "delayed_sweep" / f"{tag}__{technique}"
        print("\t".join([
            f"{tag}__{technique}", tag, technique, str(stage),
            w_tech, w_cc, steps, str(len(group)), str(manifest),
        ]))
PY
)"

VIOCF_TOTAL="$(printf '%s\n' "${VIOCF_PLAN}" | grep -c .)"
echo
echo "조각 ${VIOCF_TOTAL}개 (w_cc 태그 x 주법)"
echo

VIOCF_INDEX=0
while IFS=$'\t' read -r VIOCF_KEY VIOCF_TAG VIOCF_TECH VIOCF_STAGE \
  VIOCF_WT VIOCF_WC VIOCF_STEPS VIOCF_COUNT VIOCF_MANIFEST; do
  [[ -n "${VIOCF_KEY}" ]] || continue
  VIOCF_INDEX=$((VIOCF_INDEX + 1))

  if awk -F'\t' -v key="${VIOCF_KEY}" \
    '$1 == key && $2 == "done" { found = 1 } END { exit !found }' "${VIOCF_STATE}"; then
    echo "[${VIOCF_INDEX}/${VIOCF_TOTAL}] ${VIOCF_KEY} 이미 완료 — 건너뜀"
    continue
  fi

  # 이 조각의 MIDI 를 하드링크로 모으고 조각 manifest 를 쓴다.
  # ⚠ 심볼릭 링크는 안 된다 — run_violet.sh 의 개수 확인이 find -type f 다.
  python3 - "${VIOCF_ROOT}" "${VIOCF_MANIFEST}" "${VIOCF_TECH}" "${VIOCF_STAGE}" <<'PY'
import csv
import os
import shutil
import sys
from pathlib import Path

root, manifest, technique, stage = (
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
)
with manifest.open(newline="", encoding="utf-8") as handle:
    rows = [row for row in csv.DictReader(handle) if row["technique"] == technique]
if not rows:
    raise SystemExit(f"조각이 비었다: {technique}")
if stage.exists():
    shutil.rmtree(stage)
stage.mkdir(parents=True)
for row in rows:
    source = root / row["midi_path"]
    if not source.is_file():
        raise SystemExit(f"MIDI 없음: {source}")
    target = stage / source.name
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
chunk_manifest = stage.with_suffix(".manifest.csv")
with chunk_manifest.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
PY

  VIOCF_RUN_ID="dsweep_${VIOCF_KEY}"
  VIOCF_RUN_DIR="${VIOCF_ROOT}/logs/violet/${VIOCF_PROFILE}/${VIOCF_RUN_ID}"
  echo "[${VIOCF_INDEX}/${VIOCF_TOTAL}] ${VIOCF_KEY} — ${VIOCF_COUNT} 클립 (w_cc=${VIOCF_WC})"

  # ⚠ 상태파일에 done 이 없는 조각은 **정의상 미완성**이다. 그 실행 디렉터리에는
  # 중간에 끊긴 기록만 들어 있고, run_violet.sh 는 재사용을 거부한다
  # ("Run directory is not empty ... can duplicate JSONL records" — 옳은 판단이다).
  # 여기서 지워야 재시도가 성립한다. done 인 조각은 위에서 이미 건너뛰었으므로
  # 이 지점에 도달했다는 것 자체가 '다시 만들어도 된다'는 뜻이다.
  # (2026-08-31 디스크 장애로 중단된 뒤 재시작이 여기서 막혔다)
  if [[ -d "${VIOCF_RUN_DIR}" ]]; then
    echo "  미완성 실행 디렉터리 제거: ${VIOCF_RUN_DIR}"
    rm -rf "${VIOCF_RUN_DIR}"
  fi
  VIOCF_STARTED="$(date +%s)"

  VIOCF_MIDI_DIR_OVERRIDE="${VIOCF_STAGE}" \
  VIOCF_RUN_ID="${VIOCF_RUN_ID}" \
  VIOCF_W_TECH="${VIOCF_WT}" \
  VIOCF_W_CC="${VIOCF_WC}" \
  VIOCF_SAMPLER_STEPS="${VIOCF_STEPS}" \
    bash "${VIOCF_ROOT}/scripts/run_violet.sh" "${VIOCF_PROFILE}"

  bash "${VIOCF_ROOT}/scripts/verify_violet_run.sh" "${VIOCF_RUN_DIR}" "${VIOCF_STAGE}"

  viocf collect-violet \
    --run-dir "${VIOCF_RUN_DIR}" \
    --manifest "${VIOCF_STAGE}.manifest.csv" \
    --output "${VIOCF_ROOT}/results/collect_delayed_sweep_${VIOCF_KEY}.csv"

  VIOCF_ELAPSED=$(( $(date +%s) - VIOCF_STARTED ))
  printf '%s\tdone\t%s\t%ss\t%s clips\n' \
    "${VIOCF_KEY}" "$(date '+%F %T')" "${VIOCF_ELAPSED}" "${VIOCF_COUNT}" >>"${VIOCF_STATE}"
  echo "  완료 (${VIOCF_ELAPSED}s, 클립당 $(awk -v e="${VIOCF_ELAPSED}" -v c="${VIOCF_COUNT}" \
    'BEGIN { printf "%.2f", e / c }')s)"
  echo
done <<<"${VIOCF_PLAN}"

echo "=============================================================="
echo "지연 분기 확장 완료. 상태: ${VIOCF_STATE}"
echo "다음: bash scripts/run_delayed_sweep_analysis.sh"
echo "=============================================================="
