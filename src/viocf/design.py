from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig, project_root_from_config
from .midi import CCEvent, NoteEvent, inspect_violet_midi, write_violet_midi


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    pattern: str
    register: str
    notes: tuple[NoteEvent, ...]
    reference_midi: int | None
    single_pitch: bool


REGISTER_ROOTS = {"low": 55, "mid": 67, "high": 79}


def _pattern_notes(pattern: str, root: int, onset: float, tempo_bpm: int) -> tuple[NoteEvent, ...]:
    beat = 60.0 / tempo_bpm
    if pattern == "long":
        return (NoteEvent(root + 2, onset, 3.5),)
    if pattern == "scale":
        pitches = [root, root + 2, root + 4, root + 5, root + 7, root + 5, root + 4, root + 2]
    elif pattern == "repeat":
        pitches = [root + 7] * 8
    elif pattern == "leap":
        pitches = [root, root + 7, root + 2, root + 9, root + 4, root + 11, root + 7, root]
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    ioi = 0.75 * beat
    duration = 0.68 * beat
    return tuple(NoteEvent(pitch, onset + i * ioi, duration) for i, pitch in enumerate(pitches))


def build_prompts(config: ExperimentConfig, profile: str) -> list[Prompt]:
    patterns = ("long", "scale", "repeat", "leap")
    registers = ("low", "mid", "high")
    prompts = []
    for pattern in patterns:
        for register in registers:
            root = REGISTER_ROOTS[register]
            notes = _pattern_notes(pattern, root, config.note_onset_seconds, config.tempo_bpm)
            single_pitch = len({note.pitch for note in notes}) == 1
            prompts.append(
                Prompt(
                    prompt_id=f"{pattern}_{register}",
                    pattern=pattern,
                    register=register,
                    notes=notes,
                    reference_midi=notes[0].pitch if single_pitch else None,
                    single_pitch=single_pitch,
                )
            )
    if profile == "pilot":
        selected = {"long_mid", "scale_mid"}
        prompts = [prompt for prompt in prompts if prompt.prompt_id in selected]
    elif profile not in {"full", "expanded"}:
        raise ValueError("profile must be one of: pilot, full, expanded")
    return prompts


def stable_group_seed(base_seed: int, group_key: str, attempt: int = 0) -> int:
    """Mirror the supplied VIOLET patch's process-independent group seed."""
    payload = f"{int(base_seed)}\0{group_key}\0{int(attempt)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _model_clip_id(
    prompt_id: str,
    technique: str,
    dynamic_label: str,
    replicate: int,
    profile: str = "constant",
) -> str:
    # The supplied VIOLET patch treats the token before the first "__" as the
    # shared-noise group. All T x D cells in a prompt/replicate block share it.
    group_key = f"{prompt_id}-rep{int(replicate):02d}"
    return (
        f"{group_key}__t-{technique}__d-{dynamic_label}"
        f"__p-{profile}"
    )


def _real_clip_id(
    prompt_id: str,
    technique: str,
    dynamic_label: str,
    violin_id: str,
    take: int,
    profile: str = "constant",
) -> str:
    return (
        f"r__{prompt_id}__t-{technique}__d-{dynamic_label}"
        f"__p-{profile}__v-{violin_id}__take{take:02d}"
    )


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _constant_cc(value: int) -> list[CCEvent]:
    return [CCEvent(0.0, value)]


def _delayed_cc(config: ExperimentConfig, final_value: int, branch_offset_s: float) -> list[CCEvent]:
    branch_absolute = config.note_onset_seconds + branch_offset_s
    return [CCEvent(0.0, 64), CCEvent(config.note_onset_seconds, 64), CCEvent(branch_absolute, final_value)]


def _write_reference_midi(
    root: Path,
    profile: str,
    prompt: Prompt,
    technique: str,
    keyswitch: int,
    dynamic_label: str,
    cc_events: Iterable[CCEvent],
    tempo_bpm: int,
    suffix: str = "constant",
) -> Path:
    ref_id = f"ref__{prompt.prompt_id}__t-{technique}__d-{dynamic_label}__p-{suffix}"
    path = root / "data" / "midi" / profile / "reference" / f"{ref_id}.mid"
    if not path.exists():
        write_violet_midi(path, prompt.notes, keyswitch, cc_events, tempo_bpm)
    return path


def create_design(config: ExperimentConfig, profile: str = "pilot") -> dict[str, Path]:
    root = project_root_from_config(config)
    prompts = build_prompts(config, profile)
    techniques = config.techniques
    dynamics = config.dynamics
    base_seed = int(config.raw["model"]["base_seed"])
    configured_replicates = int(config.raw["model"]["replicates"])
    if profile == "pilot":
        model_replicates = 2
        violins = [str(config.raw["real"]["violins"][0])]
        takes = 2
    else:
        model_replicates = configured_replicates
        violins = [str(value) for value in config.raw["real"]["violins"]]
        takes = int(config.raw["real"]["takes_per_cell"])

    model_rows: list[dict[str, object]] = []
    real_rows: list[dict[str, object]] = []
    for prompt in prompts:
        for technique, keyswitch in techniques.items():
            for dynamic_label, cc1 in dynamics.items():
                reference_path = _write_reference_midi(
                    root,
                    profile,
                    prompt,
                    technique,
                    keyswitch,
                    dynamic_label,
                    _constant_cc(cc1),
                    config.tempo_bpm,
                )
                for replicate in range(1, model_replicates + 1):
                    noise_group = f"{prompt.prompt_id}-rep{replicate:02d}"
                    seed = stable_group_seed(base_seed, noise_group)
                    clip_id = _model_clip_id(
                        prompt.prompt_id, technique, dynamic_label, replicate
                    )
                    midi_path = root / "data" / "midi" / profile / "model" / f"{clip_id}.mid"
                    write_violet_midi(
                        midi_path,
                        prompt.notes,
                        keyswitch,
                        _constant_cc(cc1),
                        config.tempo_bpm,
                    )
                    model_rows.append(
                        {
                            "clip_id": clip_id,
                            "source": "model",
                            "model": "VIOLET",
                            "profile": "constant",
                            "prompt_id": prompt.prompt_id,
                            "pattern": prompt.pattern,
                            "register": prompt.register,
                            "technique": technique,
                            "technique_keyswitch": keyswitch,
                            "dynamic_label": dynamic_label,
                            "cc1_initial": cc1,
                            "cc1_final": cc1,
                            "branch_offset_s": np.nan,
                            "seed": seed,
                            "base_seed": base_seed,
                            "noise_group": noise_group,
                            "replicate": replicate,
                            "w_tech": float(config.raw["model"]["w_tech"]),
                            "w_cc": float(config.raw["model"]["w_cc"]),
                            "reference_midi": prompt.reference_midi,
                            "single_pitch": prompt.single_pitch,
                            "note_onset_s": config.note_onset_seconds,
                            "midi_path": _relative(midi_path, root),
                            "audio_path": f"data/model_audio/{clip_id}.wav",
                            "status": "planned",
                        }
                    )
                for violin_id in violins:
                    for take in range(1, takes + 1):
                        clip_id = _real_clip_id(
                            prompt.prompt_id, technique, dynamic_label, violin_id, take
                        )
                        real_rows.append(
                            {
                                "clip_id": clip_id,
                                "source": "real",
                                "model": "real_violin",
                                "profile": "constant",
                                "prompt_id": prompt.prompt_id,
                                "pattern": prompt.pattern,
                                "register": prompt.register,
                                "technique": technique,
                                "technique_keyswitch": keyswitch,
                                "dynamic_label": dynamic_label,
                                "cc1_initial": cc1,
                                "cc1_final": cc1,
                                "branch_offset_s": np.nan,
                                "violin_id": violin_id,
                                "performer_id": str(config.raw["real"]["performer_id"]),
                                "take": take,
                                "replicate": take,
                                "reference_midi": prompt.reference_midi,
                                "single_pitch": prompt.single_pitch,
                                "note_onset_s": config.note_onset_seconds,
                                "midi_path": _relative(reference_path, root),
                                "audio_path": f"data/real_48k/{clip_id}.wav",
                                "raw_audio_path": f"data/real_raw/{clip_id}.wav",
                                "status": "planned",
                            }
                        )

    delayed_prompt = Prompt(
        prompt_id="delayed_A4",
        pattern="delayed_long",
        register="mid",
        notes=(NoteEvent(69, config.note_onset_seconds, 4.0),),
        reference_midi=69,
        single_pitch=True,
    )
    branch_offset_s = 0.25
    delayed_model: list[dict[str, object]] = []
    delayed_real: list[dict[str, object]] = []
    delayed_techniques = {name: techniques[name] for name in ("sustain", "pizzicato")}
    delayed_takes = 2 if profile == "pilot" else 4
    for technique, keyswitch in delayed_techniques.items():
        for dynamic_label, final_cc in dynamics.items():
            cc_events = _delayed_cc(config, final_cc, branch_offset_s)
            reference_path = _write_reference_midi(
                root,
                profile,
                delayed_prompt,
                technique,
                keyswitch,
                dynamic_label,
                cc_events,
                config.tempo_bpm,
                suffix="delayed",
            )
            for replicate in range(1, model_replicates + 1):
                noise_group = f"{delayed_prompt.prompt_id}-rep{replicate:02d}"
                seed = stable_group_seed(base_seed, noise_group)
                clip_id = _model_clip_id(
                    delayed_prompt.prompt_id,
                    technique,
                    dynamic_label,
                    replicate,
                    profile="delayed",
                )
                midi_path = root / "data" / "midi" / profile / "model" / f"{clip_id}.mid"
                write_violet_midi(
                    midi_path,
                    delayed_prompt.notes,
                    keyswitch,
                    cc_events,
                    config.tempo_bpm,
                )
                delayed_model.append(
                    {
                        "clip_id": clip_id,
                        "source": "model",
                        "model": "VIOLET",
                        "profile": "delayed",
                        "prompt_id": delayed_prompt.prompt_id,
                        "pattern": delayed_prompt.pattern,
                        "register": delayed_prompt.register,
                        "technique": technique,
                        "technique_keyswitch": keyswitch,
                        "dynamic_label": dynamic_label,
                        "cc1_initial": 64,
                        "cc1_final": final_cc,
                        "branch_offset_s": branch_offset_s,
                        "seed": seed,
                        "base_seed": base_seed,
                        "noise_group": noise_group,
                        "replicate": replicate,
                        "w_tech": float(config.raw["model"]["w_tech"]),
                        "w_cc": float(config.raw["model"]["w_cc"]),
                        "reference_midi": 69,
                        "single_pitch": True,
                        "note_onset_s": config.note_onset_seconds,
                        "midi_path": _relative(midi_path, root),
                        "audio_path": f"data/model_audio/{clip_id}.wav",
                        "status": "planned",
                    }
                )
            for violin_id in violins:
                for take in range(1, delayed_takes + 1):
                    clip_id = _real_clip_id(
                        delayed_prompt.prompt_id,
                        technique,
                        dynamic_label,
                        violin_id,
                        take,
                        profile="delayed",
                    )
                    delayed_real.append(
                        {
                            "clip_id": clip_id,
                            "source": "real",
                            "model": "real_violin",
                            "profile": "delayed",
                            "prompt_id": delayed_prompt.prompt_id,
                            "pattern": delayed_prompt.pattern,
                            "register": delayed_prompt.register,
                            "technique": technique,
                            "technique_keyswitch": keyswitch,
                            "dynamic_label": dynamic_label,
                            "cc1_initial": 64,
                            "cc1_final": final_cc,
                            "branch_offset_s": branch_offset_s,
                            "violin_id": violin_id,
                            "performer_id": str(config.raw["real"]["performer_id"]),
                            "take": take,
                            "replicate": take,
                            "reference_midi": 69,
                            "single_pitch": True,
                            "note_onset_s": config.note_onset_seconds,
                            "midi_path": _relative(reference_path, root),
                            "audio_path": f"data/real_48k/{clip_id}.wav",
                            "raw_audio_path": f"data/real_raw/{clip_id}.wav",
                            "status": "planned",
                        }
                    )

    rng = np.random.default_rng(int(config.raw["analysis"]["random_seed"]))
    real_df = pd.DataFrame(real_rows)
    delayed_real_df = pd.DataFrame(delayed_real)
    for frame in (real_df, delayed_real_df):
        frame["recording_order"] = rng.permutation(len(frame)) + 1
        frame.sort_values("recording_order", inplace=True)

    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model": manifest_dir / f"{profile}_model.csv",
        "real": manifest_dir / f"{profile}_real.csv",
        "delayed_model": manifest_dir / f"{profile}_delayed_model.csv",
        "delayed_real": manifest_dir / f"{profile}_delayed_real.csv",
        "summary": manifest_dir / f"{profile}_design_summary.json",
    }
    pd.DataFrame(model_rows).to_csv(outputs["model"], index=False)
    real_df.to_csv(outputs["real"], index=False)
    pd.DataFrame(delayed_model).to_csv(outputs["delayed_model"], index=False)
    delayed_real_df.to_csv(outputs["delayed_real"], index=False)

    all_midi = list((root / "data" / "midi" / profile).rglob("*.mid"))
    validation = [inspect_violet_midi(path) for path in all_midi]
    summary = {
        "profile": profile,
        "prompt_count": len(prompts),
        "model_clips": len(model_rows),
        "real_takes": len(real_rows),
        "delayed_model_clips": len(delayed_model),
        "delayed_real_takes": len(delayed_real),
        "midi_files": len(all_midi),
        "invalid_midi": [row for row in validation if not row["valid"]],
        "critical_note": "Do not record the full design until the pilot and VIOLET smoke test pass.",
    }
    outputs["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def merge_manifests(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("At least one manifest is required")
    merged = pd.concat(frames, ignore_index=True, sort=False)
    if merged["clip_id"].duplicated().any():
        duplicates = merged.loc[merged["clip_id"].duplicated(), "clip_id"].tolist()
        raise ValueError(f"Duplicate clip_id values: {duplicates[:5]}")
    return merged


def create_smoke_design(config: ExperimentConfig) -> Path:
    """Create a two-clip, same-noise VIOLET smoke test from the pilot design."""
    root = project_root_from_config(config)
    pilot_manifest = root / "manifests" / "pilot_model.csv"
    if not pilot_manifest.exists():
        create_design(config, profile="pilot")
    pilot = pd.read_csv(pilot_manifest)
    selected = pilot.loc[
        pilot["prompt_id"].eq("long_mid")
        & pilot["technique"].eq("sustain")
        & pilot["dynamic_label"].isin(["p", "f"])
        & pilot["replicate"].eq(1)
    ].copy()
    selected.sort_values("cc1_final", inplace=True)
    if len(selected) != 2 or selected["noise_group"].nunique() != 1:
        raise RuntimeError("Pilot manifest does not contain the expected paired smoke cells")

    smoke_dir = root / "data" / "midi" / "smoke" / "model"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    updated_rows: list[dict[str, object]] = []
    for row in selected.to_dict(orient="records"):
        source = root / str(row["midi_path"])
        destination = smoke_dir / source.name
        expected_names.add(destination.name)
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise FileExistsError(f"Refusing to overwrite a different smoke MIDI: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        updated = dict(row)
        updated["midi_path"] = _relative(destination, root)
        updated["status"] = "planned_smoke"
        updated_rows.append(updated)

    extra = sorted(path.name for path in smoke_dir.glob("*.mid") if path.name not in expected_names)
    if extra:
        raise RuntimeError(f"Smoke directory contains unexpected MIDI files: {extra}")
    output = root / "manifests" / "smoke_model.csv"
    pd.DataFrame(updated_rows).to_csv(output, index=False)
    return output
