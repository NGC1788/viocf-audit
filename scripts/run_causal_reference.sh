#!/usr/bin/env bash
# 인과적 기준 렌더러로 파이프라인 전체를 검증한다 (CPU, GPU 와 병행 가능).
#
# 질문: "분기 전 spread 2.03 dB 가 측정 잡음이 아니라는 걸 어떻게 아는가?"
#
# 답: 인과성이 보장된 렌더러로 같은 MIDI 를 렌더해 같은 파이프라인에 넣는다.
#
#   양성 대조 (causal)  분기 전이 비트 단위로 같다 -> spread 가 0.00 이어야 한다
#                       0 이 아니면 **우리 측정이 틀린 것이다**
#   음성 대조 (leaky)   CC1 을 0.6 s 미리 당겨 적용 -> spread 가 커야 한다
#                       안 잡히면 검정력이 없다는 뜻이다
#
# 양성 대조만으로는 '아무것도 검출 못 하는 파이프라인'과 구분되지 않는다.
# 둘 다 있어야 검증이다.
#
# 사용: scripts/run_causal_reference.sh [클립수제한]
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_LIMIT="${1:-0}"

cd "${VIOCF_ROOT}"
[[ -f .venv/bin/activate ]] || { echo "먼저: bash scripts/bootstrap_analysis.sh"; exit 2; }
source .venv/bin/activate

for VIOCF_ARM in causal leaky; do
  echo "=============================================================="
  echo "기준 렌더 — ${VIOCF_ARM}"
  echo "=============================================================="
  viocf render-reference \
    --manifest manifests/delayed_sweep/wc1p0.csv \
    --arm "${VIOCF_ARM}" \
    --limit "${VIOCF_LIMIT}" \
    --output "manifests/reference_${VIOCF_ARM}.csv"

  viocf features \
    --manifest "manifests/reference_${VIOCF_ARM}.csv" \
    --output "results/reference/${VIOCF_ARM}_features.csv"

  viocf metrics \
    --features "results/reference/${VIOCF_ARM}_features.csv" \
    --output-dir "results/reference/${VIOCF_ARM}_metrics"
done

echo
echo "=============================================================="
echo "판정"
echo "=============================================================="
python - <<'PY'
import json
from pathlib import Path

root = Path("results/reference")
verdict = {}
for arm in ("causal", "leaky"):
    path = root / f"{arm}_metrics" / "metrics_summary.json"
    if not path.is_file():
        print(f"{arm}: 요약이 없다")
        continue
    summary = json.loads(path.read_text(encoding="utf-8"))
    block = summary.get("delayed_model_only_prebranch") or {}
    measured = block.get("noise_groups_measured", 0)
    identical = block.get("identical_groups", 0)
    median = (block.get("median_spread") or {}).get("spread_prebranch_abs_rms_dbfs")
    verdict[arm] = (measured, identical, median)
    print(f"[{arm}]  그룹 {measured}  동일 {identical}  중앙 spread "
          f"{'없음' if median is None else f'{median:.6f} dB'}")

print()
ok = True
if "causal" in verdict:
    measured, identical, median = verdict["causal"]
    if measured and identical == measured:
        print("  ✓ 양성 대조 통과 — 인과적 렌더러에서 모든 그룹이 '동일' 판정.")
        print("    파이프라인이 없는 누출을 만들어내지 않는다.")
    else:
        ok = False
        print(f"  ✗ 양성 대조 실패 — {measured}개 중 {identical}개만 동일하다.")
        print("    분기 전이 비트 단위로 같은데 다르다고 했다. **측정이 틀렸다.**")
if "leaky" in verdict:
    measured, identical, median = verdict["leaky"]
    if measured and identical == 0:
        print("  ✓ 음성 대조 통과 — 일부러 넣은 누출을 전부 잡아냈다.")
    else:
        ok = False
        print(f"  ✗ 음성 대조 실패 — {identical}개 그룹을 '동일'로 봤다. 검정력 부족.")
print()
print("  결론: 파이프라인 " + ("검증됨" if ok else "**신뢰할 수 없음**"))
raise SystemExit(0 if ok else 1)
PY
