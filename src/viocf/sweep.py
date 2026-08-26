from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig, project_root_from_config
from .design import Prompt, build_prompts, stable_group_seed
from .midi import CCEvent, write_violet_midi

DENSE_CC1_LEVELS = (8, 24, 40, 56, 64, 72, 88, 104, 120)
GUIDANCE_LEVELS = (0.0, 0.5, 1.0, 2.0)
DEFAULT_DENSE_REPLICATES = 8
DEFAULT_GUIDANCE_REPLICATES = 4


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _weight_tag(w_tech: float, w_cc: float) -> str:
    def encode(value: float) -> str:
        return f"{round(value * 10):02d}"

    return f"wt{encode(w_tech)}_wc{encode(w_cc)}"


def _scaled_prompt(prompt: Prompt, anchor_s: float, scale: float, suffix: str) -> Prompt:
    notes = tuple(
        replace(
            note,
            onset_s=anchor_s + (note.onset_s - anchor_s) * scale,
            duration_s=note.duration_s * scale,
        )
        for note in prompt.notes
    )
    return replace(prompt, prompt_id=f"{prompt.prompt_id}_{suffix}", notes=notes)


def expanded_prompts(config: ExperimentConfig) -> list[Prompt]:
    """Return the 12 base prompts at short and long timing scales."""
    prompts: list[Prompt] = []
    for prompt in build_prompts(config, profile="full"):
        prompts.append(_scaled_prompt(prompt, config.note_onset_seconds, 0.70, "short"))
        prompts.append(_scaled_prompt(prompt, config.note_onset_seconds, 1.25, "long"))
    return prompts


def planned_sweep_counts(
    dense_replicates: int = DEFAULT_DENSE_REPLICATES,
    guidance_replicates: int = DEFAULT_GUIDANCE_REPLICATES,
) -> dict[str, int]:
    """Return design counts without writing thousands of MIDI files."""
    if dense_replicates < 1 or guidance_replicates < 1:
        raise ValueError("Sweep replicate counts must be positive")
    dense = 24 * len(DENSE_CC1_LEVELS) * 4 * dense_replicates
    guidance = 6 * 4 * 3 * guidance_replicates * len(GUIDANCE_LEVELS) ** 2
    return {"dense_clips": dense, "guidance_clips": guidance, "total_clips": dense + guidance}


def _manifest_row(
    *,
    root: Path,
    config: ExperimentConfig,
    sweep_kind: str,
    prompt: Prompt,
    technique: str,
    keyswitch: int,
    dynamic_label: str,
    cc1: int,
    replicate: int,
    w_tech: float,
    w_cc: float,
    midi_path: Path,
    clip_id: str,
) -> dict[str, Any]:
    noise_group = f"{prompt.prompt_id}-rep{replicate:02d}"
    base_seed = int(config.raw["model"]["base_seed"])
    return {
        "clip_id": clip_id,
        "source": "model",
        "model": "VIOLET",
        "profile": f"sweep_{sweep_kind}",
        "sweep_kind": sweep_kind,
        "prompt_id": prompt.prompt_id,
        "pattern": prompt.pattern,
        "register": prompt.register,
        "timing_variant": prompt.prompt_id.rsplit("_", maxsplit=1)[-1],
        "technique": technique,
        "technique_keyswitch": keyswitch,
        "dynamic_label": dynamic_label,
        "cc1_initial": cc1,
        "cc1_final": cc1,
        "branch_offset_s": float("nan"),
        "seed": stable_group_seed(base_seed, noise_group),
        "base_seed": base_seed,
        "noise_group": noise_group,
        "replicate": replicate,
        "w_tech": w_tech,
        "w_cc": w_cc,
        "reference_midi": prompt.reference_midi,
        "single_pitch": prompt.single_pitch,
        "note_onset_s": config.note_onset_seconds,
        "midi_path": _relative(midi_path, root),
        "audio_path": f"data/model_audio/sweep/{clip_id}.wav",
        "status": "planned",
    }


def _write_sweep_cell(
    *,
    root: Path,
    config: ExperimentConfig,
    midi_dir: Path,
    sweep_kind: str,
    prompt: Prompt,
    technique: str,
    keyswitch: int,
    dynamic_label: str,
    cc1: int,
    replicate: int,
    w_tech: float,
    w_cc: float,
    weight_tag: str,
) -> dict[str, Any]:
    noise_group = f"{prompt.prompt_id}-rep{replicate:02d}"
    clip_id = (
        f"{noise_group}__t-{technique}__d-{dynamic_label}"
        f"__s-{sweep_kind}__w-{weight_tag}"
    )
    midi_path = midi_dir / f"{clip_id}.mid"
    write_violet_midi(
        midi_path,
        prompt.notes,
        keyswitch,
        [CCEvent(0.0, cc1)],
        config.tempo_bpm,
    )
    return _manifest_row(
        root=root,
        config=config,
        sweep_kind=sweep_kind,
        prompt=prompt,
        technique=technique,
        keyswitch=keyswitch,
        dynamic_label=dynamic_label,
        cc1=cc1,
        replicate=replicate,
        w_tech=w_tech,
        w_cc=w_cc,
        midi_path=midi_path,
        clip_id=clip_id,
    )


def create_compute_sweep(
    config: ExperimentConfig,
    dense_replicates: int = DEFAULT_DENSE_REPLICATES,
    guidance_replicates: int = DEFAULT_GUIDANCE_REPLICATES,
) -> dict[str, Path]:
    """Create the post-pilot dense-CC1 and guidance-weight compute sweeps.

    The default design contains 6,912 dense response clips and 4,608
    guidance-ablation clips. Every technique/dynamics cell in a
    prompt/replicate block shares the filename prefix and deterministic seed.
    """
    planned_sweep_counts(dense_replicates, guidance_replicates)
    root = project_root_from_config(config)
    techniques = config.techniques
    prompts = expanded_prompts(config)
    default_w_tech = float(config.raw["model"]["w_tech"])
    default_w_cc = float(config.raw["model"]["w_cc"])
    manifest_dir = root / "manifests" / "sweep"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    dense_rows: list[dict[str, Any]] = []
    dense_midi_dir = root / "data" / "midi" / "sweep" / "dense"
    dense_weight_tag = _weight_tag(default_w_tech, default_w_cc)
    for prompt in prompts:
        for technique, keyswitch in techniques.items():
            for cc1 in DENSE_CC1_LEVELS:
                for replicate in range(1, dense_replicates + 1):
                    dense_rows.append(
                        _write_sweep_cell(
                            root=root,
                            config=config,
                            midi_dir=dense_midi_dir,
                            sweep_kind="dense",
                            prompt=prompt,
                            technique=technique,
                            keyswitch=keyswitch,
                            dynamic_label=f"cc{cc1:03d}",
                            cc1=cc1,
                            replicate=replicate,
                            w_tech=default_w_tech,
                            w_cc=default_w_cc,
                            weight_tag=dense_weight_tag,
                        )
                    )
    dense_manifest = manifest_dir / "dense.csv"
    pd.DataFrame(dense_rows).to_csv(dense_manifest, index=False)

    prompt_by_id = {prompt.prompt_id: prompt for prompt in prompts}
    anchor_ids = (
        "long_low_short",
        "long_high_long",
        "scale_mid_short",
        "repeat_high_short",
        "leap_low_long",
        "leap_mid_short",
    )
    missing_anchors = sorted(set(anchor_ids) - prompt_by_id.keys())
    if missing_anchors:
        raise RuntimeError(f"Guidance anchor prompts are missing: {missing_anchors}")

    guidance_rows: list[dict[str, Any]] = []
    guidance_manifests: dict[str, Path] = {}
    for w_tech in GUIDANCE_LEVELS:
        for w_cc in GUIDANCE_LEVELS:
            weight_tag = _weight_tag(w_tech, w_cc)
            pair_rows: list[dict[str, Any]] = []
            pair_midi_dir = root / "data" / "midi" / "sweep" / "guidance" / weight_tag
            for prompt_id in anchor_ids:
                prompt = prompt_by_id[prompt_id]
                for technique, keyswitch in techniques.items():
                    for dynamic_label, cc1 in config.dynamics.items():
                        for replicate in range(1, guidance_replicates + 1):
                            pair_rows.append(
                                _write_sweep_cell(
                                    root=root,
                                    config=config,
                                    midi_dir=pair_midi_dir,
                                    sweep_kind="guidance",
                                    prompt=prompt,
                                    technique=technique,
                                    keyswitch=keyswitch,
                                    dynamic_label=dynamic_label,
                                    cc1=cc1,
                                    replicate=replicate,
                                    w_tech=w_tech,
                                    w_cc=w_cc,
                                    weight_tag=weight_tag,
                                )
                            )
            pair_manifest = manifest_dir / f"guidance_{weight_tag}.csv"
            pd.DataFrame(pair_rows).to_csv(pair_manifest, index=False)
            guidance_manifests[f"guidance_{weight_tag}"] = pair_manifest
            guidance_rows.extend(pair_rows)

    guidance_manifest = manifest_dir / "guidance_all.csv"
    pd.DataFrame(guidance_rows).to_csv(guidance_manifest, index=False)
    summary_path = manifest_dir / "sweep_summary.json"
    counts = {
        "dense_clips": len(dense_rows),
        "guidance_clips": len(guidance_rows),
        "total_clips": len(dense_rows) + len(guidance_rows),
    }
    summary = {
        **counts,
        "expanded_prompt_count": len(prompts),
        "dense_cc1_levels": list(DENSE_CC1_LEVELS),
        "dense_replicates": dense_replicates,
        "guidance_anchor_prompts": list(anchor_ids),
        "guidance_weights": list(GUIDANCE_LEVELS),
        "guidance_replicates": guidance_replicates,
        "guidance_pair_manifests": len(guidance_manifests),
        "critical_note": "Run only after the pilot gates and official VIOLET smoke test pass.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "dense": dense_manifest,
        "guidance_all": guidance_manifest,
        "summary": summary_path,
        **guidance_manifests,
    }
