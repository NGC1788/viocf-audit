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
# 확산 스텝 수 축. 기본은 30이다.
#
# 스텝 수에 따른 **오디오 품질**(FAD 등)은 다들 재지만, 스텝 수에 따른
# **제어 충실도**를 잰 연구는 없다. 스텝을 줄이면 소리는 그럭저럭인데 강약 추종이
# 먼저 무너질 수도 있고, 반대로 30스텝이 이미 포화라 120은 낭비일 수도 있다.
# 어느 쪽이든 실무자에게 바로 쓸모 있는 결과이고, MIDI 를 새로 만들 필요가 없어 싸다.
SAMPLING_STEP_LEVELS = (8, 16, 30, 50, 80, 120)
DEFAULT_DENSE_REPLICATES = 8
DEFAULT_GUIDANCE_REPLICATES = 4
DEFAULT_STEPS_REPLICATES = 8
# guidance/steps 스윕이 쓰는 고정 프롬프트 6개 (전체를 다 돌리면 조합이 폭발한다)
ANCHOR_PROMPT_IDS = (
    "long_low_short",
    "long_high_long",
    "scale_mid_short",
    "repeat_high_short",
    "leap_low_long",
    "leap_mid_short",
)


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
    steps_replicates: int = DEFAULT_STEPS_REPLICATES,
    technique_count: int = 4,
    dynamics_count: int = 3,
) -> dict[str, int]:
    """Return design counts without writing thousands of MIDI files.

    ⚠ 주법 개수를 하드코딩하지 않는다. configs 에서 주법을 8개로 늘렸는데 여기가
    4로 고정돼 있으면 계획 수치와 실제 생성량이 2배 어긋난다.
    """
    if min(dense_replicates, guidance_replicates, steps_replicates) < 1:
        raise ValueError("Sweep replicate counts must be positive")
    dense = 24 * len(DENSE_CC1_LEVELS) * technique_count * dense_replicates
    guidance = (
        len(ANCHOR_PROMPT_IDS) * technique_count * dynamics_count
        * guidance_replicates * len(GUIDANCE_LEVELS) ** 2
    )
    steps = (
        len(ANCHOR_PROMPT_IDS) * technique_count * dynamics_count
        * steps_replicates * len(SAMPLING_STEP_LEVELS)
    )
    return {
        "dense_clips": dense,
        "guidance_clips": guidance,
        "steps_clips": steps,
        "total_clips": dense + guidance + steps,
    }


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
    sampling_steps: int | None = None,
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
        "analysis_tier": (
            "real_counterfactual_primary"
            if technique in config.real_techniques
            else "generator_only_exploratory"
        ),
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
        "sampling_steps": (
            int(sampling_steps) if sampling_steps is not None
            # 설정에 없으면 VIOLET 기본값. 스텝 축을 쓰지 않는 스윕도 이 열을 갖게 해서
            # 나중에 여러 스윕을 합칠 때 열이 어긋나지 않게 한다.
            else int(config.raw["model"].get("sampling_steps", 30))
        ),
        "reference_midi": (
            prompt.reference_midi
            if technique not in {"trill_major", "trill_minor"}
            else float("nan")
        ),
        "single_pitch": bool(
            prompt.single_pitch and technique not in {"trill_major", "trill_minor"}
        ),
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
    sampling_steps: int | None = None,
) -> dict[str, Any]:
    noise_group = f"{prompt.prompt_id}-rep{replicate:02d}"
    clip_id = (
        f"{noise_group}__t-{technique}__d-{dynamic_label}"
        f"__s-{sweep_kind}__w-{weight_tag}"
    )
    if sampling_steps is not None:
        clip_id = f"{clip_id}__n-{int(sampling_steps):03d}"
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
        sampling_steps=sampling_steps,
    )


def create_compute_sweep(
    config: ExperimentConfig,
    dense_replicates: int = DEFAULT_DENSE_REPLICATES,
    guidance_replicates: int = DEFAULT_GUIDANCE_REPLICATES,
    steps_replicates: int = DEFAULT_STEPS_REPLICATES,
    include_exploratory_techniques: bool = False,
) -> dict[str, Path]:
    """Create post-pilot CC1, guidance, and sampler-step sweeps.

    By default only techniques with a real-instrument counterfactual baseline
    are included. The other configured techniques can be added explicitly as
    generator-only exploratory analyses; they must not enter primary claims.
    """
    techniques = (
        config.techniques if include_exploratory_techniques else config.real_techniques
    )
    planned_sweep_counts(
        dense_replicates, guidance_replicates, steps_replicates,
        technique_count=len(techniques), dynamics_count=len(config.dynamics),
    )
    root = project_root_from_config(config)
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
    anchor_ids = ANCHOR_PROMPT_IDS
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

    # ---------------- 확산 스텝 수 스윕 ----------------
    # MIDI 내용은 스텝 수와 무관하지만(샘플러 파라미터다), 실행기가 manifest 하나를
    # 원자적 작업으로 다루므로 스텝 수마다 manifest 를 따로 낸다. 한 run 안에서
    # 설정이 섞이면 나중에 manifest 만 보고 감사할 수 없게 된다.
    steps_rows: list[dict[str, Any]] = []
    steps_manifests: dict[str, Path] = {}
    steps_weight_tag = _weight_tag(default_w_tech, default_w_cc)
    for n_steps in SAMPLING_STEP_LEVELS:
        level_rows: list[dict[str, Any]] = []
        level_midi_dir = root / "data" / "midi" / "sweep" / "steps" / f"n{n_steps:03d}"
        for prompt_id in anchor_ids:
            prompt = prompt_by_id[prompt_id]
            for technique, keyswitch in techniques.items():
                for dynamic_label, cc1 in config.dynamics.items():
                    for replicate in range(1, steps_replicates + 1):
                        level_rows.append(
                            _write_sweep_cell(
                                root=root,
                                config=config,
                                midi_dir=level_midi_dir,
                                sweep_kind="steps",
                                prompt=prompt,
                                technique=technique,
                                keyswitch=keyswitch,
                                dynamic_label=dynamic_label,
                                cc1=cc1,
                                replicate=replicate,
                                w_tech=default_w_tech,
                                w_cc=default_w_cc,
                                weight_tag=steps_weight_tag,
                                sampling_steps=n_steps,
                            )
                        )
        level_manifest = manifest_dir / f"steps_n{n_steps:03d}.csv"
        pd.DataFrame(level_rows).to_csv(level_manifest, index=False)
        steps_manifests[f"steps_n{n_steps:03d}"] = level_manifest
        steps_rows.extend(level_rows)
    steps_manifest = manifest_dir / "steps_all.csv"
    pd.DataFrame(steps_rows).to_csv(steps_manifest, index=False)

    guidance_manifest = manifest_dir / "guidance_all.csv"
    pd.DataFrame(guidance_rows).to_csv(guidance_manifest, index=False)
    summary_path = manifest_dir / "sweep_summary.json"
    counts = {
        "dense_clips": len(dense_rows),
        "guidance_clips": len(guidance_rows),
        "steps_clips": len(steps_rows),
        "total_clips": len(dense_rows) + len(guidance_rows) + len(steps_rows),
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
        "sampling_step_levels": list(SAMPLING_STEP_LEVELS),
        "steps_replicates": steps_replicates,
        "techniques": list(techniques),
        "include_exploratory_techniques": bool(include_exploratory_techniques),
        "primary_techniques": list(config.real_techniques),
        "critical_note": "Run only after the pilot gates and official VIOLET smoke test pass.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "dense": dense_manifest,
        "guidance_all": guidance_manifest,
        "steps_all": steps_manifest,
        "summary": summary_path,
        **guidance_manifests,
        **steps_manifests,
    }
