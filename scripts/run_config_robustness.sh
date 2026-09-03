#!/usr/bin/env bash
# 설정 강건성 실행 — 분기 오프셋 x 주법부류 x CC 유도 강도.
#
# 왜: 개정 31 에 약점이 둘 남았다.
#   1) w_cc 누출비 0.104 -> 0.203 이 **점 두 개짜리 추세**다.
#      w_cc 0.5 / 1.0 / 2.0 / 3.0 / 4.0 으로 곡선을 만든다.
#   2) sampling_steps 를 30 에 고정하고 한 번도 안 흔들었다.
#      "샘플링이 부족해서 아니냐" 는 반론이 열려 있다. 10 / 30 / 60 / 120.
#
# 둘 다 채우면 "설정 어디를 만져도 안 없어진다" 가 된다.
#
# 격자는 줄였다(주법 4, 오프셋 2). 축 하나당 점을 늘리는 게 목적이므로
# 나머지는 양 끝만 남긴다. w_cc 1.0/2.0 도 다시 돌린다 — 격자가 다르면
# 기존 값과 비교가 성립하지 않는다.
#
# w_cc 와 steps 는 샘플러 설정이라 조합마다 별도 실행이 필요하다.
#
# 재개 가능: logs/config_robustness_state.tsv 에 완료한 (설정, 주법) 조각을 남긴다.
# 오프셋·강약·반복은 한 조각 안에서 함께 돌아간다.
#
# 사용: scripts/run_config_robustness.sh [replicates]
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_REPLICATES="${1:-32}"
VIOCF_STATE="${VIOCF_ROOT}/logs/config_robustness_state.tsv"
VIOCF_PROFILE="config_robustness"

cd "${VIOCF_ROOT}"
if [[ ! -f .venv/bin/activate ]]; then
  echo "분석 환경이 없다. 먼저: bash scripts/bootstrap_analysis.sh"
  exit 2
fi
source .venv/bin/activate
mkdir -p "$(dirname "${VIOCF_STATE}")"
touch "${VIOCF_STATE}"

echo "=============================================================="
echo "설정 강건성  replicates=${VIOCF_REPLICATES}"
echo "=============================================================="
viocf make-config-robustness --replicates "${VIOCF_REPLICATES}"

# 조각 = (w_cc 태그, 주법). 주법 단위로 쪼개야 죽어도 잃는 시간이 짧다.
VIOCF_PLAN="$(python3 - "${VIOCF_ROOT}" <<'PY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
manifest_dir = root / "manifests" / "config_robustness"
# ⚠ 이 스크립트는 run_delayed_sweep.sh 에서 파생됐다. 거기서는 manifest 이름이
# wc0p0.csv 였지만 여기서는 cc0p5_n030.csv 처럼 (w_cc, steps) 조합 이름이다.
# glob 을 안 고쳐서 하나도 못 찾고, 빈 결과에 grep -c 가 1 을 반환해
# set -e 로 죽었다. 파생 스크립트는 이런 잔재를 반드시 훑을 것.
for manifest in sorted(manifest_dir.glob("*.csv")):
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
    # ⚠ 한 manifest 안에서 샘플러 설정이 섞이면 안 된다. 섞이면 조각 하나에
    # 서로 다른 설정의 클립이 들어가고 결과를 가를 수 없다.
    if len({r["w_cc"] for r in rows}) != 1 or len({r["sampling_steps"] for r in rows}) != 1:
        raise SystemExit(f"{manifest}: w_cc 또는 sampling_steps 가 섞여 있다")
    for technique, group in sorted(by_technique.items()):
        # 조각별 MIDI 디렉터리와 manifest 를 따로 만든다.
        # (전체 manifest 를 collect 에 넘기면 all_pass 가 항상 false 가 된다 —
        #  run_profile_chunked.sh 에서 겪은 것과 같은 함정)
        stage = root / "logs" / "chunks" / "config_robustness" / f"{tag}__{technique}"
        print("\t".join([
            f"{tag}__{technique}", tag, technique, str(stage),
            w_tech, w_cc, steps, str(len(group)), str(manifest),
        ]))
PY
)"

VIOCF_TOTAL="$(printf '%s\n' "${VIOCF_PLAN}" | grep -c . || true)"
if [[ "${VIOCF_TOTAL}" -eq 0 ]]; then
  echo "조각 계획이 비었다. manifests/${VIOCF_PROFILE}/ 에 CSV 가 있는지,"
  echo "그리고 위 python 의 glob 이 그 이름과 맞는지 확인할 것."
  ls -l "${VIOCF_ROOT}/manifests/${VIOCF_PROFILE}/" 2>&1 | head
  exit 2
fi
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

  VIOCF_RUN_ID="crob_${VIOCF_KEY}"
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
    --output "${VIOCF_ROOT}/results/collect_config_robustness_${VIOCF_KEY}.csv"

  VIOCF_ELAPSED=$(( $(date +%s) - VIOCF_STARTED ))
  printf '%s\tdone\t%s\t%ss\t%s clips\n' \
    "${VIOCF_KEY}" "$(date '+%F %T')" "${VIOCF_ELAPSED}" "${VIOCF_COUNT}" >>"${VIOCF_STATE}"
  echo "  완료 (${VIOCF_ELAPSED}s, 클립당 $(awk -v e="${VIOCF_ELAPSED}" -v c="${VIOCF_COUNT}" \
    'BEGIN { printf "%.2f", e / c }')s)"
  echo
done <<<"${VIOCF_PLAN}"

echo "=============================================================="
echo "설정 강건성 완료. 상태: ${VIOCF_STATE}"
echo "다음: bash scripts/run_config_robustness_analysis.sh"
echo "=============================================================="
