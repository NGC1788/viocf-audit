#!/usr/bin/env bash
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_SWEEP_PHASE="${VIOCF_SWEEP_PHASE:-all}"
VIOCF_SWEEP_RUN_ID="${VIOCF_SWEEP_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
VIOCF_SWEEP_DRY_RUN="${VIOCF_SWEEP_DRY_RUN:-false}"
VIOCF_SWEEP_MANIFEST_DIR="${VIOCF_SWEEP_MANIFEST_DIR:-${VIOCF_ROOT}/manifests/sweep}"

if [[ "${VIOCF_SWEEP_PHASE}" != "all" &&
  "${VIOCF_SWEEP_PHASE}" != "dense" &&
  "${VIOCF_SWEEP_PHASE}" != "guidance" ]]; then
  echo "VIOCF_SWEEP_PHASE must be all, dense, or guidance."
  exit 2
fi
if [[ "${VIOCF_SWEEP_DRY_RUN}" != "true" && "${VIOCF_SWEEP_DRY_RUN}" != "false" ]]; then
  echo "VIOCF_SWEEP_DRY_RUN must be true or false."
  exit 2
fi

# The Python preflight treats each CSV as an atomic render job. It deliberately
# rejects mixed guidance weights and extra MIDI files, since either would make
# a run impossible to audit from its manifest alone.
VIOCF_PLAN="$({ python3 - \
  "${VIOCF_ROOT}" \
  "${VIOCF_SWEEP_MANIFEST_DIR}" \
  "${VIOCF_SWEEP_PHASE}" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path


root = Path(sys.argv[1]).resolve()
manifest_dir = Path(sys.argv[2]).resolve()
phase = sys.argv[3]

if not manifest_dir.is_dir():
    raise SystemExit(
        f"Missing sweep manifests: {manifest_dir}\n"
        "Run: viocf make-sweep"
    )

manifests: list[Path] = []
if phase in {"all", "dense"}:
    manifests.append(manifest_dir / "dense.csv")
if phase in {"all", "guidance"}:
    # guidance_all.csv is an analysis convenience file with mixed weights;
    # render only the one-weight-pair manifests.
    manifests.extend(sorted(manifest_dir.glob("guidance_wt*_wc*.csv")))

if not manifests or any(not path.is_file() for path in manifests):
    missing = [str(path) for path in manifests if not path.is_file()]
    raise SystemExit(f"Missing sweep manifest(s): {missing or manifest_dir}")

for manifest in manifests:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Empty sweep manifest: {manifest}")

    required = {"clip_id", "midi_path", "w_tech", "w_cc"}
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise SystemExit(f"{manifest}: missing columns {sorted(missing_columns)}")

    pairs = {(float(row["w_tech"]), float(row["w_cc"])) for row in rows}
    if len(pairs) != 1:
        raise SystemExit(f"{manifest}: mixed guidance pairs {sorted(pairs)}")
    w_tech, w_cc = next(iter(pairs))

    midi_paths = []
    for row in rows:
        path = Path(row["midi_path"])
        path = path if path.is_absolute() else root / path
        path = path.resolve()
        if not path.is_file():
            raise SystemExit(f"{manifest}: missing MIDI {path}")
        midi_paths.append(path)
    if len(midi_paths) != len(set(midi_paths)):
        raise SystemExit(f"{manifest}: duplicate midi_path rows")

    parents = {path.parent for path in midi_paths}
    if len(parents) != 1:
        raise SystemExit(f"{manifest}: MIDI files span multiple directories")
    midi_dir = next(iter(parents))
    actual = {*midi_dir.rglob("*.mid"), *midi_dir.rglob("*.midi")}
    planned = set(midi_paths)
    if actual != planned:
        missing = sorted(str(path) for path in planned - actual)
        extra = sorted(str(path) for path in actual - planned)
        raise SystemExit(
            f"{manifest}: directory/manifest mismatch; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    label = manifest.stem.replace("guidance_", "g_")
    print(f"{label}\t{midi_dir}\t{w_tech:g}\t{w_cc:g}\t{len(rows)}")
PY
} 2>&1)" || {
  echo "${VIOCF_PLAN}"
  exit 2
}

if [[ -z "${VIOCF_PLAN}" ]]; then
  echo "No sweep jobs selected."
  exit 2
fi

echo "Sweep run id: ${VIOCF_SWEEP_RUN_ID}"
echo "Sweep phase: ${VIOCF_SWEEP_PHASE}"
echo "Planned jobs: $(printf '%s\n' "${VIOCF_PLAN}" | wc -l | tr -d ' ')"

while IFS=$'\t' read -r VIOCF_LABEL VIOCF_MIDI_DIR VIOCF_W_TECH VIOCF_W_CC VIOCF_EXPECTED; do
  [[ -n "${VIOCF_LABEL}" ]] || continue
  VIOCF_JOB_RUN_ID="${VIOCF_SWEEP_RUN_ID}_${VIOCF_LABEL}"
  VIOCF_JOB_RUN_DIR="${VIOCF_ROOT}/logs/violet/sweep/${VIOCF_JOB_RUN_ID}"

  echo
  echo "[${VIOCF_LABEL}] ${VIOCF_EXPECTED} clips; w_tech=${VIOCF_W_TECH}, w_cc=${VIOCF_W_CC}"
  if awk -v wt="${VIOCF_W_TECH}" -v wc="${VIOCF_W_CC}" \
    'BEGIN { exit !(wt == 0 && wc > 0) }'; then
    echo "WARNING: w_tech=0,w_cc>0 is a diagnostic mixture, not a pure CC-only branch."
  fi

  if [[ -d "${VIOCF_JOB_RUN_DIR}" ]] &&
    [[ -n "$(find "${VIOCF_JOB_RUN_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    if bash "${VIOCF_ROOT}/scripts/verify_violet_run.sh" \
      "${VIOCF_JOB_RUN_DIR}" "${VIOCF_MIDI_DIR}" \
      "${VIOCF_W_TECH}" "${VIOCF_W_CC}"; then
      echo "[${VIOCF_LABEL}] already complete; skipping."
      continue
    fi
    echo "[${VIOCF_LABEL}] existing run is partial or invalid."
    echo "Use a new VIOCF_SWEEP_RUN_ID; partial directories are never overwritten."
    exit 2
  fi

  if [[ "${VIOCF_SWEEP_DRY_RUN}" == "true" ]]; then
    echo "DRY RUN: VIOCF_MIDI_DIR_OVERRIDE=${VIOCF_MIDI_DIR} VIOCF_W_TECH=${VIOCF_W_TECH} VIOCF_W_CC=${VIOCF_W_CC} VIOCF_RUN_ID=${VIOCF_JOB_RUN_ID} scripts/run_violet.sh sweep"
    continue
  fi

  VIOCF_MIDI_DIR_OVERRIDE="${VIOCF_MIDI_DIR}" \
  VIOCF_W_TECH="${VIOCF_W_TECH}" \
  VIOCF_W_CC="${VIOCF_W_CC}" \
  VIOCF_RUN_ID="${VIOCF_JOB_RUN_ID}" \
    bash "${VIOCF_ROOT}/scripts/run_violet.sh" sweep

  bash "${VIOCF_ROOT}/scripts/verify_violet_run.sh" \
    "${VIOCF_JOB_RUN_DIR}" "${VIOCF_MIDI_DIR}" \
    "${VIOCF_W_TECH}" "${VIOCF_W_CC}"
done <<< "${VIOCF_PLAN}"

echo "Sweep phase complete: ${VIOCF_SWEEP_PHASE}"
