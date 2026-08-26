from __future__ import annotations

import csv
from pathlib import Path

import pytest

from viocf.recording import RecordingSession, cue_frame, load_real_manifest

FIELDS = [
    "clip_id",
    "source",
    "prompt_id",
    "pattern",
    "technique",
    "dynamic_label",
    "cc1_initial",
    "cc1_final",
    "branch_offset_s",
    "note_onset_s",
    "violin_id",
    "performer_id",
    "take",
    "recording_order",
    "midi_path",
    "raw_audio_path",
]


def _write_manifest(path: Path, source: str = "real") -> None:
    rows = [
        {
            "clip_id": "clip-a",
            "source": source,
            "prompt_id": "long_mid",
            "pattern": "long",
            "technique": "sustain",
            "dynamic_label": "p",
            "cc1_initial": "32",
            "cc1_final": "32",
            "branch_offset_s": "",
            "note_onset_s": "0.75",
            "violin_id": "V1",
            "performer_id": "P1",
            "take": "1",
            "recording_order": "2",
            "midi_path": "data/ref-a.mid",
            "raw_audio_path": "data/clip-a.wav",
        },
        {
            "clip_id": "clip-b",
            "source": source,
            "prompt_id": "delayed_A4",
            "pattern": "delayed_long",
            "technique": "sustain",
            "dynamic_label": "f",
            "cc1_initial": "64",
            "cc1_final": "96",
            "branch_offset_s": "0.25",
            "note_onset_s": "0.75",
            "violin_id": "V1",
            "performer_id": "P1",
            "take": "1",
            "recording_order": "1",
            "midi_path": "data/ref-b.mid",
            "raw_audio_path": "data/clip-b.wav",
        },
        {
            "clip_id": "clip-c",
            "source": source,
            "prompt_id": "scale_mid",
            "pattern": "scale",
            "technique": "staccato",
            "dynamic_label": "mf",
            "cc1_initial": "64",
            "cc1_final": "64",
            "branch_offset_s": "",
            "note_onset_s": "0.75",
            "violin_id": "V1",
            "performer_id": "P1",
            "take": "1",
            "recording_order": "3",
            "midi_path": "data/ref-c.mid",
            "raw_audio_path": "data/clip-c.wav",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_order_and_seed_are_deterministic(tmp_path: Path) -> None:
    manifest = tmp_path / "real.csv"
    _write_manifest(manifest)
    ordered = load_real_manifest(manifest)
    assert [item.clip_id for item in ordered] == ["clip-b", "clip-a", "clip-c"]

    first = load_real_manifest(manifest, seed=20260826)
    second = load_real_manifest(manifest, seed=20260826)
    assert [item.clip_id for item in first] == [item.clip_id for item in second]
    assert {item.clip_id for item in first} == {"clip-a", "clip-b", "clip-c"}


def test_manifest_rejects_model_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "model.csv"
    _write_manifest(manifest, source="model")
    with pytest.raises(ValueError, match="only accepts real-instrument rows"):
        load_real_manifest(manifest)


def test_delayed_cue_switches_at_note_onset_plus_offset(tmp_path: Path) -> None:
    manifest = tmp_path / "real.csv"
    _write_manifest(manifest)
    delayed = load_real_manifest(manifest)[0]
    assert delayed.branch_time_s == pytest.approx(1.0)

    before = cue_frame(delayed, elapsed_s=0.80, duration_s=10.0)
    assert before.phase == "initial"
    assert before.dynamic_label == "mf"
    assert before.cc1 == 64
    assert before.switch_in_s == pytest.approx(0.20)

    after = cue_frame(delayed, elapsed_s=1.0, duration_s=10.0)
    assert after.phase == "target"
    assert after.dynamic_label == "f"
    assert after.cc1 == 96
    assert cue_frame(delayed, elapsed_s=10.0, duration_s=10.0).phase == "done"


def test_session_redo_resume_and_truncated_log_recovery(tmp_path: Path) -> None:
    manifest = tmp_path / "real.csv"
    log = tmp_path / "session.jsonl"
    _write_manifest(manifest)

    session = RecordingSession.open(manifest, log)
    assert session.current is not None
    assert session.current.clip_id == "clip-b"
    assert session.attempt_for() == 1

    session.mark("redo", note="bow noise")
    assert session.current is not None
    assert session.current.clip_id == "clip-b"
    assert session.attempt_for() == 2
    session.mark("accept")
    assert session.current is not None
    assert session.current.clip_id == "clip-a"
    session.mark("skip", note="string broke")
    assert session.completed == 2
    assert session.accepted == 1
    assert session.skipped == 1

    with log.open("ab") as handle:
        handle.write(b'{"event":"acc')
    resumed = RecordingSession.open(manifest, log, seed=999)
    assert resumed.repaired_log
    assert resumed.session_id == session.session_id
    assert resumed.completed == 2
    assert resumed.current is not None
    assert resumed.current.clip_id == "clip-c"
    assert resumed.attempt_for(resumed.items[0]) == 2

    resumed.mark("accept")
    assert resumed.done
    snapshot = log.with_suffix(".csv")
    with snapshot.open("r", encoding="utf-8", newline="") as handle:
        statuses = {row["clip_id"]: row["status"] for row in csv.DictReader(handle)}
    assert statuses == {"clip-b": "accept", "clip-a": "skip", "clip-c": "accept"}


def test_resume_detects_changed_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "real.csv"
    log = tmp_path / "session.jsonl"
    _write_manifest(manifest)
    RecordingSession.open(manifest, log)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Manifest changed"):
        RecordingSession.open(manifest, log)
