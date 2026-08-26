from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import mido

TICKS_PER_BEAT = 480
TECHNIQUE_LEAD_SECONDS = 0.02
KEYSWITCH_DURATION_SECONDS = 0.01


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    onset_s: float
    duration_s: float
    velocity: int = 80


@dataclass(frozen=True)
class CCEvent:
    time_s: float
    value: int


def _validate_midi_value(value: int, name: str) -> int:
    value = int(value)
    if not 0 <= value <= 127:
        raise ValueError(f"{name} must be in [0, 127], got {value}")
    return value


def write_violet_midi(
    output_path: str | Path,
    notes: Sequence[NoteEvent],
    technique_keyswitch: int,
    cc1_events: Iterable[CCEvent],
    tempo_bpm: int = 96,
) -> Path:
    """Write a VIOLET-compatible monophonic MIDI file.

    Technique labels are encoded as short keyswitch notes 20 ms before each
    playable note. Dynamics are CC1 events. Musical-note velocity is fixed by
    the caller so CC1, rather than velocity, is the intended dynamics control.
    """
    if not notes:
        raise ValueError("At least one playable note is required")
    technique_keyswitch = _validate_midi_value(technique_keyswitch, "technique_keyswitch")
    tempo_bpm = int(tempo_bpm)
    if tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be positive")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tempo = mido.bpm2tempo(tempo_bpm)

    # (absolute seconds, tie-break priority, message)
    events: list[tuple[float, int, mido.Message | mido.MetaMessage]] = []
    events.append((0.0, 0, mido.MetaMessage("set_tempo", tempo=tempo, time=0)))
    events.append((0.0, 1, mido.Message("program_change", program=40, channel=0, time=0)))

    for cc in cc1_events:
        value = _validate_midi_value(cc.value, "CC1 value")
        events.append(
            (max(0.0, float(cc.time_s)), 2, mido.Message("control_change", control=1, value=value, channel=0, time=0))
        )

    for note in notes:
        pitch = _validate_midi_value(note.pitch, "pitch")
        velocity = _validate_midi_value(note.velocity, "velocity")
        onset = float(note.onset_s)
        duration = float(note.duration_s)
        if pitch < 55:
            raise ValueError(f"Playable violin pitch must be >=55 (G3), got {pitch}")
        if onset < 0 or duration <= 0:
            raise ValueError(f"Invalid note onset/duration: {onset}, {duration}")

        ks_on = max(0.0, onset - TECHNIQUE_LEAD_SECONDS)
        ks_off = min(onset - 0.001, ks_on + KEYSWITCH_DURATION_SECONDS)
        if ks_off <= ks_on:
            ks_off = ks_on + 0.001
        events.extend(
            [
                (ks_on, 3, mido.Message("note_on", note=technique_keyswitch, velocity=100, channel=0, time=0)),
                (ks_off, 4, mido.Message("note_off", note=technique_keyswitch, velocity=0, channel=0, time=0)),
                (onset, 5, mido.Message("note_on", note=pitch, velocity=velocity, channel=0, time=0)),
                (onset + duration, 6, mido.Message("note_off", note=pitch, velocity=0, channel=0, time=0)),
            ]
        )

    events.sort(key=lambda item: (item[0], item[1]))
    midi = mido.MidiFile(type=0, ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    previous_tick = 0
    for seconds, _, message in events:
        absolute_tick = round(mido.second2tick(seconds, TICKS_PER_BEAT, tempo))
        delta = max(0, absolute_tick - previous_tick)
        previous_tick = absolute_tick
        message.time = delta
        track.append(message)
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(output)
    return output


def inspect_violet_midi(path: str | Path) -> dict[str, object]:
    """Return a compact validation report without requiring VIOLET itself."""
    midi = mido.MidiFile(path)
    absolute_s = 0.0
    tempo = mido.bpm2tempo(120)
    playable_notes = 0
    keyswitch_notes = 0
    cc1_values: list[int] = []
    for message in mido.merge_tracks(midi.tracks):
        absolute_s += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "control_change" and message.control == 1:
            cc1_values.append(int(message.value))
        elif message.type == "note_on" and message.velocity > 0:
            if int(message.note) < 55:
                keyswitch_notes += 1
            else:
                playable_notes += 1
    return {
        "path": str(path),
        "duration_s": absolute_s,
        "playable_note_onsets": playable_notes,
        "keyswitch_onsets": keyswitch_notes,
        "cc1_values": cc1_values,
        "valid": bool(playable_notes > 0 and keyswitch_notes == playable_notes and cc1_values),
    }
