from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Decision = Literal["accept", "redo", "skip"]


def _optional_float(value: str | None) -> float | None:
    text = "" if value is None else value.strip()
    if not text:
        return None
    return float(text)


def _required_text(row: dict[str, str], key: str, row_number: int) -> str:
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"Manifest row {row_number} has no {key!r}")
    return value


@dataclass(frozen=True)
class RecordingItem:
    clip_id: str
    prompt_id: str
    pattern: str
    technique: str
    dynamic_label: str
    cc1_initial: int
    cc1_final: int
    branch_offset_s: float | None
    note_onset_s: float
    violin_id: str
    performer_id: str
    take: int
    recording_order: int
    midi_path: str
    raw_audio_path: str

    @property
    def delayed(self) -> bool:
        return self.branch_offset_s is not None

    @property
    def branch_time_s(self) -> float | None:
        if self.branch_offset_s is None:
            return None
        return self.note_onset_s + self.branch_offset_s


@dataclass(frozen=True)
class CueFrame:
    phase: Literal["initial", "target", "done"]
    dynamic_label: str
    cc1: int
    switch_in_s: float | None


def cc1_label(value: int) -> str:
    """Return the benchmark's nearest nominal dynamic label for a CC1 value."""
    anchors = ((32, "p"), (64, "mf"), (96, "f"))
    return min(anchors, key=lambda pair: abs(pair[0] - value))[1]


def cue_frame(item: RecordingItem, elapsed_s: float, duration_s: float) -> CueFrame:
    """Map recording time to the visual cue shown to the performer."""
    if elapsed_s >= duration_s:
        return CueFrame("done", item.dynamic_label, item.cc1_final, None)
    branch_time = item.branch_time_s
    if branch_time is not None and elapsed_s < branch_time:
        return CueFrame(
            "initial",
            cc1_label(item.cc1_initial),
            item.cc1_initial,
            max(0.0, branch_time - elapsed_s),
        )
    return CueFrame("target", item.dynamic_label, item.cc1_final, None)


def manifest_fingerprint(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_real_manifest(path: str | Path, seed: int | None = None) -> list[RecordingItem]:
    """Load, validate, and order a real-instrument recording manifest.

    With no seed, the pre-randomized ``recording_order`` column is respected.
    Supplying a seed performs an additional deterministic shuffle.
    """
    source = Path(path).expanduser().resolve()
    items: list[RecordingItem] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {source}")
        for row_number, row in enumerate(reader, start=2):
            source_kind = _required_text(row, "source", row_number)
            if source_kind != "real":
                raise ValueError(
                    f"Manifest row {row_number} is source={source_kind!r}; "
                    "the recording helper only accepts real-instrument rows"
                )
            try:
                item = RecordingItem(
                    clip_id=_required_text(row, "clip_id", row_number),
                    prompt_id=_required_text(row, "prompt_id", row_number),
                    pattern=_required_text(row, "pattern", row_number),
                    technique=_required_text(row, "technique", row_number),
                    dynamic_label=_required_text(row, "dynamic_label", row_number),
                    cc1_initial=int(_required_text(row, "cc1_initial", row_number)),
                    cc1_final=int(_required_text(row, "cc1_final", row_number)),
                    branch_offset_s=_optional_float(row.get("branch_offset_s")),
                    note_onset_s=float(_required_text(row, "note_onset_s", row_number)),
                    violin_id=_required_text(row, "violin_id", row_number),
                    performer_id=_required_text(row, "performer_id", row_number),
                    take=int(_required_text(row, "take", row_number)),
                    recording_order=int(_required_text(row, "recording_order", row_number)),
                    midi_path=_required_text(row, "midi_path", row_number),
                    raw_audio_path=_required_text(row, "raw_audio_path", row_number),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid manifest row {row_number}: {exc}") from exc
            items.append(item)

    if not items:
        raise ValueError(f"Manifest contains no recording rows: {source}")
    clip_ids = [item.clip_id for item in items]
    duplicates = sorted({clip_id for clip_id in clip_ids if clip_ids.count(clip_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate clip_id values in manifest: {duplicates[:5]}")
    orders = [item.recording_order for item in items]
    if len(set(orders)) != len(orders):
        raise ValueError("recording_order must be unique")

    items.sort(key=lambda item: item.recording_order)
    if seed is not None:
        random.Random(seed).shuffle(items)
    return items


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl_recover(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Read an append-only log and discard only a crash-truncated final record."""
    if not path.exists() or path.stat().st_size == 0:
        return [], False
    raw_lines = path.read_bytes().splitlines(keepends=True)
    events: list[dict[str, Any]] = []
    valid_bytes = 0
    repaired = False
    for index, raw_line in enumerate(raw_lines):
        stripped = raw_line.strip()
        if not stripped:
            valid_bytes += len(raw_line)
            continue
        try:
            event = json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index != len(raw_lines) - 1:
                raise ValueError(f"Corrupt session log at line {index + 1}: {path}") from exc
            repaired = True
            break
        if not isinstance(event, dict):
            raise TypeError(f"Session log line {index + 1} is not an object: {path}")
        events.append(event)
        valid_bytes += len(raw_line)

    if repaired:
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    return events, repaired


class RecordingSession:
    """Crash-recoverable state over a deterministic sequence of recording items."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        log_path: Path,
        items: list[RecordingItem],
        session_id: str,
        events: list[dict[str, Any]],
        repaired_log: bool = False,
    ) -> None:
        self.manifest_path = manifest_path
        self.log_path = log_path
        self.items = tuple(items)
        self.session_id = session_id
        self.repaired_log = repaired_log
        self._decisions: dict[str, Literal["accept", "skip"]] = {}
        self._attempts = {item.clip_id: 1 for item in items}
        self._replay(events)

    @classmethod
    def open(
        cls,
        manifest_path: str | Path,
        log_path: str | Path,
        *,
        seed: int | None = None,
    ) -> RecordingSession:
        manifest = Path(manifest_path).expanduser().resolve()
        log = Path(log_path).expanduser().resolve()
        base_items = load_real_manifest(manifest)
        item_by_id = {item.clip_id: item for item in base_items}
        fingerprint = manifest_fingerprint(manifest)
        events, repaired = _read_jsonl_recover(log)

        if events:
            header = events[0]
            if header.get("event") != "session_start":
                raise ValueError(f"First session log event is not session_start: {log}")
            if header.get("manifest_sha256") != fingerprint:
                raise ValueError(
                    "Manifest changed since this session began; use a new log path "
                    "or restore the original manifest"
                )
            ordered_ids = header.get("ordered_clip_ids")
            if not isinstance(ordered_ids, list) or set(ordered_ids) != set(item_by_id):
                raise ValueError("Session log clip set does not match the manifest")
            items = [item_by_id[str(clip_id)] for clip_id in ordered_ids]
            session_id = str(header.get("session_id") or "")
            if not session_id:
                raise ValueError("Session header has no session_id")
            session = cls(
                manifest_path=manifest,
                log_path=log,
                items=items,
                session_id=session_id,
                events=events[1:],
                repaired_log=repaired,
            )
            session.write_snapshot()
            return session

        items = load_real_manifest(manifest, seed=seed)
        session_id = uuid.uuid4().hex
        log.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "schema_version": 1,
            "event": "session_start",
            "timestamp_utc": _utc_now(),
            "session_id": session_id,
            "manifest_path": str(manifest),
            "manifest_sha256": fingerprint,
            "shuffle_seed": seed,
            "ordered_clip_ids": [item.clip_id for item in items],
        }
        cls._append(log, header)
        session = cls(
            manifest_path=manifest,
            log_path=log,
            items=items,
            session_id=session_id,
            events=[],
        )
        session.write_snapshot()
        return session

    @staticmethod
    def _append(path: Path, event: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def _replay(self, events: list[dict[str, Any]]) -> None:
        valid_ids = set(self._attempts)
        for line_number, event in enumerate(events, start=2):
            action = event.get("event")
            clip_id = str(event.get("clip_id") or "")
            if action not in {"accept", "redo", "skip"} or clip_id not in valid_ids:
                raise ValueError(f"Invalid session event at log line {line_number}")
            if clip_id in self._decisions:
                raise ValueError(f"Event after terminal decision for {clip_id!r}")
            if action == "redo":
                self._attempts[clip_id] += 1
            else:
                self._decisions[clip_id] = action

    @property
    def current(self) -> RecordingItem | None:
        return next(
            (item for item in self.items if item.clip_id not in self._decisions),
            None,
        )

    @property
    def completed(self) -> int:
        return len(self._decisions)

    @property
    def accepted(self) -> int:
        return sum(value == "accept" for value in self._decisions.values())

    @property
    def skipped(self) -> int:
        return sum(value == "skip" for value in self._decisions.values())

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def done(self) -> bool:
        return self.current is None

    def attempt_for(self, item: RecordingItem | None = None) -> int:
        target = item or self.current
        if target is None:
            raise RuntimeError("The recording session is complete")
        return self._attempts[target.clip_id]

    def mark(self, decision: Decision, note: str = "") -> RecordingItem:
        if decision not in {"accept", "redo", "skip"}:
            raise ValueError(f"Unknown recording decision: {decision}")
        item = self.current
        if item is None:
            raise RuntimeError("The recording session is complete")
        attempt = self._attempts[item.clip_id]
        event = {
            "schema_version": 1,
            "event": decision,
            "timestamp_utc": _utc_now(),
            "session_id": self.session_id,
            "clip_id": item.clip_id,
            "attempt": attempt,
            "raw_audio_path": item.raw_audio_path,
            "note": note,
        }
        self._append(self.log_path, event)
        if decision == "redo":
            self._attempts[item.clip_id] += 1
        else:
            self._decisions[item.clip_id] = decision
        self.write_snapshot()
        return item

    def write_snapshot(self, path: str | Path | None = None) -> Path:
        """Materialize a human-readable CSV; the JSONL remains the source of truth."""
        target = (
            Path(path).expanduser().resolve()
            if path is not None
            else self.log_path.with_suffix(".csv")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "clip_id",
            "status",
            "attempts",
            "recording_order",
            "prompt_id",
            "technique",
            "dynamic_label",
            "violin_id",
            "take",
            "raw_audio_path",
        ]
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for item in self.items:
                    writer.writerow(
                        {
                            "clip_id": item.clip_id,
                            "status": self._decisions.get(item.clip_id, "pending"),
                            "attempts": self._attempts[item.clip_id],
                            "recording_order": item.recording_order,
                            "prompt_id": item.prompt_id,
                            "technique": item.technique,
                            "dynamic_label": item.dynamic_label,
                            "violin_id": item.violin_id,
                            "take": item.take,
                            "raw_audio_path": item.raw_audio_path,
                        }
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return target
