#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 5 ]]; then
  echo "Usage: $0 RUN_DIR MIDI_DIR [EXPECTED_W_TECH EXPECTED_W_CC EXPECTED_STEPS]"
  exit 2
fi

VIOCF_RUN_DIR="$1"
VIOCF_MIDI_DIR="$2"
VIOCF_EXPECTED_W_TECH="${3:-}"
VIOCF_EXPECTED_W_CC="${4:-}"
VIOCF_EXPECTED_STEPS="${5:-}"

python3 - \
  "${VIOCF_RUN_DIR}" \
  "${VIOCF_MIDI_DIR}" \
  "${VIOCF_EXPECTED_W_TECH}" \
  "${VIOCF_EXPECTED_W_CC}" \
  "${VIOCF_EXPECTED_STEPS}" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


run_dir = Path(sys.argv[1]).resolve()
midi_dir = Path(sys.argv[2]).resolve()
expected_w_tech = None if not sys.argv[3] else float(sys.argv[3])
expected_w_cc = None if not sys.argv[4] else float(sys.argv[4])
expected_steps = None if not sys.argv[5] else int(sys.argv[5])

if not run_dir.is_dir():
    raise SystemExit(f"FAIL: missing run directory: {run_dir}")
if not midi_dir.is_dir():
    raise SystemExit(f"FAIL: missing MIDI directory: {midi_dir}")

manifest_paths = sorted(run_dir.rglob("conditioning_debug.jsonl"))
if len(manifest_paths) != 1:
    raise SystemExit(
        f"FAIL: expected exactly one conditioning_debug.jsonl, found {len(manifest_paths)}"
    )
manifest_path = manifest_paths[0]
records = [
    json.loads(line)
    for line in manifest_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

midi_paths = sorted([*midi_dir.rglob("*.mid"), *midi_dir.rglob("*.midi")])
midi_stems = [path.stem for path in midi_paths]
if len(midi_stems) != len(set(midi_stems)):
    duplicates = sorted(stem for stem, count in Counter(midi_stems).items() if count > 1)
    raise SystemExit(f"FAIL: duplicate MIDI stems would overwrite audio: {duplicates[:5]}")

record_names = [str(record.get("filename")) for record in records]
if len(record_names) != len(set(record_names)):
    raise SystemExit("FAIL: duplicate filename records in render manifest")
if set(record_names) != set(midi_stems):
    missing = sorted(set(midi_stems) - set(record_names))
    extra = sorted(set(record_names) - set(midi_stems))
    raise SystemExit(
        f"FAIL: render/MIDI mismatch; missing={missing[:5]}, extra={extra[:5]}"
    )

seeds_by_group: dict[str, set[int]] = defaultdict(set)
for record in records:
    filename = str(record["filename"])
    noise_group = record.get("noise_group")
    if not noise_group:
        raise SystemExit(f"FAIL: {filename} has no noise_group")
    if filename.split("__", 1)[0] != noise_group:
        raise SystemExit(f"FAIL: {filename} has inconsistent noise_group={noise_group}")

    base_seed = int(record["base_seed"])
    render_seed = int(record["render_seed"])
    attempt = int(record["render_attempt"]) - 1
    payload = f"{base_seed}\0{noise_group}\0{attempt}".encode("utf-8")
    expected_seed = int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    ) & ((1 << 63) - 1)
    if render_seed != expected_seed:
        raise SystemExit(
            f"FAIL: {filename} seed={render_seed}, expected={expected_seed}"
        )
    seeds_by_group[str(noise_group)].add(render_seed)

    audio_path = manifest_path.parent / str(record["saved_audio"])
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise SystemExit(f"FAIL: missing/empty rendered WAV: {audio_path}")

    if not record.get("debug_branches_enabled", False) and record.get(
        "saved_branch_audio"
    ):
        raise SystemExit(f"FAIL: unexpected debug branch audio for {filename}")

    if expected_w_tech is not None and not math.isclose(
        float(record["effective_w_tech"]), expected_w_tech, abs_tol=1e-9
    ):
        raise SystemExit(f"FAIL: wrong w_tech in {filename}")
    if expected_w_cc is not None and not math.isclose(
        float(record["effective_w_cc"]), expected_w_cc, abs_tol=1e-9
    ):
        raise SystemExit(f"FAIL: wrong w_cc in {filename}")
    if expected_steps is not None:
        actual_steps = record.get("effective_sampling_steps")
        if actual_steps is None:
            raise SystemExit(f"FAIL: no sampling-step audit field in {filename}")
        if int(actual_steps) != expected_steps:
            raise SystemExit(
                f"FAIL: wrong sampling steps in {filename}: "
                f"{actual_steps} != {expected_steps}"
            )

bad_groups = {group: seeds for group, seeds in seeds_by_group.items() if len(seeds) != 1}
if bad_groups:
    raise SystemExit(f"FAIL: paired counterfactuals used multiple seeds: {bad_groups}")

print(
    "PASS: "
    f"{len(records)} renders, {len(seeds_by_group)} paired-noise groups, "
    f"manifest={manifest_path}"
)
PY
