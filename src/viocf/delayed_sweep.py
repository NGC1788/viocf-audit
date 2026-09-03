"""지연 분기 확장 설계 — 세 축으로 넓힌다.

## 왜 넓히는가

기존 지연 실험은 **단 한 점**이었다: 분기 오프셋 0.25 s, 주법 2개, w_cc 1.0.
거기서 나온 결과(51/51 그룹 누출, 피치카토 누출비 0.618 vs 지속음 0.059)는
강하지만 세 가지 질문에 답하지 못한다.

### 1. 거리에 따라 줄어드는가 (분기 오프셋)

인과적 생성기라면 어떤 거리에서도 분기 전 차이가 0 이다. 전 구간 확산이라면
거리와 무관하게 샐 수 있다. **점 하나가 아니라 곡선**이 있어야 그 구분이 선다.
"3 초 뒤에 일어날 일이 지금 소리에 이미 들어 있다"는 훨씬 강한 진술이다.

⚠ 창 길이는 오프셋과 무관하게 고정해야 한다(features.REFERENCE_BRANCH_OFFSET_S).
   안 그러면 오프셋이 클수록 긴 구간을 재게 돼 비교가 성립하지 않는다.

### 2. '줄을 놓았는가'가 정말 원인인가 (주법)

지금은 지속음 1개 vs 피치카토 1개다. n=1 대 n=1 로 물리적 해석을 붙였다.
활이 줄에 남는 주법 3개와 줄이 풀리는 주법 3개로 넓히면 **부류 안에서 반복**된다.

  활이 남는다   sustain, legato_slur, tremolo
  줄이 풀린다   pizzicato, spiccato, staccato

### 3. 손잡이를 세게 돌리면 나아지는가, 나빠지는가 (w_cc)

"CC 유도를 약하게 줘서 그런 것 아니냐"는 당연한 반론이다. 그런데 반대 가능성이
더 흥미롭다 — 유도를 세게 주면 조건이 **창 전체**에 더 강하게 박혀서 과거로 새는
양이 늘어날 수 있다. 그렇다면 "손잡이를 세게 돌릴수록 덜 인과적이 된다"가 된다.

w_cc 는 샘플러 설정이라 클립마다 바꿀 수 없다. 값마다 별도 실행이 필요하므로
manifest 를 w_cc 별로 나눠 쓴다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig, project_root_from_config
from .design import (
    NoteEvent,
    Prompt,
    _delayed_cc,
    _model_clip_id,
    _relative,
    stable_group_seed,
)
from .midi import write_violet_midi

# 창(0.19 s)이 항상 분기 앞에 오도록 최소 오프셋은 REFERENCE_BRANCH_OFFSET_S 이상.
BRANCH_OFFSETS_S = (0.25, 0.50, 1.00, 1.75, 3.00)

# 활이 줄에 남는 쪽 / 줄이 풀리는 쪽. 분기 시점의 세기 변화가 물리적으로
# 가능한가가 갈리는 지점이다.
SUSTAINED_TECHNIQUES = ("sustain", "legato_slur", "tremolo")
RELEASED_TECHNIQUES = ("pizzicato", "spiccato", "staccato")

# w_cc 는 샘플러 설정이라 실행을 나눠야 한다. 0.0 은 유도를 끈 경우다.
CC_GUIDANCE_LEVELS = (0.0, 1.0, 2.0)


def _weight_tag(value: float) -> str:
    return f"wc{str(value).replace('.', 'p')}"


def plan_size(replicates: int) -> dict[str, int]:
    techniques = len(SUSTAINED_TECHNIQUES) + len(RELEASED_TECHNIQUES)
    per_level = techniques * 3 * len(BRANCH_OFFSETS_S) * replicates
    return {
        "techniques": techniques,
        "offsets": len(BRANCH_OFFSETS_S),
        "cc_levels": len(CC_GUIDANCE_LEVELS),
        "clips_per_cc_level": per_level,
        "clips_total": per_level * len(CC_GUIDANCE_LEVELS),
    }


def create_delayed_sweep(
    config: ExperimentConfig,
    replicates: int = 32,
    profile: str = "delayed_sweep",
) -> dict[str, Path]:
    """w_cc 값마다 하나씩 manifest 를 쓴다. 반환값은 {가중치태그: 경로}."""
    root = project_root_from_config(config)
    base_seed = int(config.raw["model"]["base_seed"])
    keyswitches = dict(config.techniques)
    techniques = [
        name for name in (*SUSTAINED_TECHNIQUES, *RELEASED_TECHNIQUES)
        if name in keyswitches
    ]
    missing = sorted(
        set(SUSTAINED_TECHNIQUES + RELEASED_TECHNIQUES) - set(techniques)
    )
    if missing:
        raise RuntimeError(f"키스위치가 없는 주법: {missing}")

    prompt = Prompt(
        prompt_id="delayed_A4",
        pattern="delayed_long",
        register="mid",
        # 분기 3.0 s + 창을 담으려면 음이 그만큼 이어져야 한다.
        notes=(NoteEvent(69, config.note_onset_seconds, 6.0),),
        reference_midi=69,
        single_pitch=True,
    )

    manifest_dir = root / "manifests" / "delayed_sweep"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    for w_cc in CC_GUIDANCE_LEVELS:
        tag = _weight_tag(w_cc)
        rows: list[dict[str, Any]] = []
        midi_dir = root / "data" / "midi" / profile / tag
        for technique in techniques:
            keyswitch = keyswitches[technique]
            released = technique in RELEASED_TECHNIQUES
            for branch_offset in BRANCH_OFFSETS_S:
                for dynamic_label, final_cc in config.dynamics.items():
                    cc_events = _delayed_cc(config, final_cc, branch_offset)
                    for replicate in range(1, replicates + 1):
                        # ⚠ noise_group 에 오프셋을 넣어야 한다. 안 넣으면 오프셋이
                        # 다른 클립이 같은 latent 를 공유해, 짝 비교의 단위가
                        # 흐려진다. 강약은 넣지 않는다 — 같은 latent 에서 강약만
                        # 바꾸는 것이 이 실험의 핵심 대비다.
                        noise_group = (
                            f"{prompt.prompt_id}-{tag}"
                            f"-off{int(branch_offset * 100):04d}"
                            f"-{technique}-rep{replicate:02d}"
                        )
                        seed = stable_group_seed(base_seed, noise_group)
                        clip_id = _model_clip_id(
                            prompt.prompt_id, technique, dynamic_label, replicate,
                            profile=f"{tag}_off{int(branch_offset * 100):04d}",
                        )
                        midi_path = midi_dir / f"{clip_id}.mid"
                        write_violet_midi(
                            midi_path, prompt.notes, keyswitch, cc_events,
                            config.tempo_bpm,
                        )
                        rows.append({
                            "clip_id": clip_id,
                            "source": "model",
                            "model": "VIOLET",
                            "profile": "delayed",
                            "sweep_kind": "delayed_sweep",
                            "prompt_id": prompt.prompt_id,
                            "pattern": prompt.pattern,
                            "register": prompt.register,
                            "technique": technique,
                            "technique_class": "released" if released else "sustained",
                            "technique_keyswitch": keyswitch,
                            "dynamic_label": dynamic_label,
                            "cc1_initial": 64,
                            "cc1_final": final_cc,
                            "branch_offset_s": branch_offset,
                            "note_onset_s": config.note_onset_seconds,
                            "seed": seed,
                            "base_seed": base_seed,
                            "noise_group": noise_group,
                            "replicate": replicate,
                            "w_tech": float(config.raw["model"]["w_tech"]),
                            "w_cc": w_cc,
                            "sampling_steps": int(config.raw["model"]["sampling_steps"]),
                            "midi_path": _relative(midi_path, root),
                            "audio_path": _relative(
                                root / "data" / "model_audio" / profile / tag
                                / f"{clip_id}.wav",
                                root,
                            ),
                            # 실연주 짝이 없다. 헤드라인 지표에 섞이면 안 된다.
                            "analysis_tier": "generator_only_exploratory",
                        })
        manifest = manifest_dir / f"{tag}.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        outputs[tag] = manifest
    return outputs


# ─────────────────────────────────────────────────────────────────────────
# 설정 강건성 — "설정을 바꾸면 누출이 사라지나"
# ─────────────────────────────────────────────────────────────────────────
#
# 개정 31 에서 두 가지가 약점으로 남았다.
#
#   1. w_cc 누출비 0.104 -> 0.203 이 **점 두 개짜리 추세**다.
#      "손잡이를 세게 돌릴수록 덜 인과적" 은 지금 결과 중 가장 인상적인데
#      근거가 가장 약하다.
#   2. sampling_steps 를 30 으로 고정하고 한 번도 안 흔들었다.
#      "샘플링이 부족해서 아니냐" 는 반론이 그대로 열려 있다.
#
# 둘을 채우면 "설정 어디를 만져도 안 없어진다" 가 되고, 그게 구조적 문제라는
# 주장의 최종 형태다.
#
# 격자를 줄인다. 축 하나당 점을 늘리는 게 목적이므로 나머지는 양 끝만 남긴다.
#   주법 4개  = 활이 완전히 떨어지는 쪽 2개 + 계속 닿는 쪽 2개 (개정 31 의 순서 양끝)
#   오프셋 2개 = 가장 가까운 것과 먼 편 하나
# w_cc 는 기존 1.0/2.0 도 **다시 돌린다** — 격자가 달라지면 비교가 성립하지 않는다.

CONFIG_TECHNIQUES = ("pizzicato", "spiccato", "sustain", "tremolo")
CONFIG_OFFSETS_S = (0.25, 1.75)
CONFIG_CC_LEVELS = (0.5, 1.0, 2.0, 3.0, 4.0)
# steps 는 w_cc 를 기본값에 고정하고 따로 훑는다(둘을 곱하면 격자가 폭발한다).
CONFIG_STEPS_LEVELS = (10, 30, 60, 120)
CONFIG_STEPS_AT_CC = 1.0


def config_plan_size(replicates: int) -> dict[str, int]:
    cell = len(CONFIG_TECHNIQUES) * 3 * len(CONFIG_OFFSETS_S) * replicates
    cc_clips = cell * len(CONFIG_CC_LEVELS)
    steps_clips = cell * len(CONFIG_STEPS_LEVELS)
    return {
        "clips_per_cell": cell,
        "cc_levels": len(CONFIG_CC_LEVELS),
        "steps_levels": len(CONFIG_STEPS_LEVELS),
        "cc_clips": cc_clips,
        "steps_clips": steps_clips,
        "clips_total": cc_clips + steps_clips,
    }


def create_config_robustness(
    config: ExperimentConfig,
    replicates: int = 32,
    profile: str = "config_robustness",
) -> dict[str, Path]:
    """(w_cc, sampling_steps) 마다 manifest 를 하나씩 쓴다."""
    root = project_root_from_config(config)
    base_seed = int(config.raw["model"]["base_seed"])
    keyswitches = dict(config.techniques)
    default_steps = int(config.raw["model"]["sampling_steps"])
    default_w_tech = float(config.raw["model"]["w_tech"])

    prompt = Prompt(
        prompt_id="delayed_A4",
        pattern="delayed_long",
        register="mid",
        notes=(NoteEvent(69, config.note_onset_seconds, 6.0),),
        reference_midi=69,
        single_pitch=True,
    )

    # (태그, w_cc, steps) 목록. w_cc 축은 steps 를 기본값에, steps 축은 w_cc 를
    # CONFIG_STEPS_AT_CC 에 고정한다. 겹치는 조합은 한 번만 만든다.
    arms: list[tuple[str, float, int]] = []
    for w_cc in CONFIG_CC_LEVELS:
        arms.append((f"cc{_weight_tag(w_cc)[2:]}_n{default_steps:03d}", w_cc, default_steps))
    for steps in CONFIG_STEPS_LEVELS:
        tag = f"cc{_weight_tag(CONFIG_STEPS_AT_CC)[2:]}_n{steps:03d}"
        if all(tag != existing for existing, _, _ in arms):
            arms.append((tag, CONFIG_STEPS_AT_CC, steps))

    manifest_dir = root / "manifests" / profile
    manifest_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    for tag, w_cc, steps in arms:
        rows: list[dict[str, Any]] = []
        midi_dir = root / "data" / "midi" / profile / tag
        for technique in CONFIG_TECHNIQUES:
            keyswitch = keyswitches[technique]
            released = technique in RELEASED_TECHNIQUES
            for branch_offset in CONFIG_OFFSETS_S:
                for dynamic_label, final_cc in config.dynamics.items():
                    cc_events = _delayed_cc(config, final_cc, branch_offset)
                    for replicate in range(1, replicates + 1):
                        noise_group = (
                            f"{prompt.prompt_id}-{tag}"
                            f"-off{int(branch_offset * 100):04d}"
                            f"-{technique}-rep{replicate:02d}"
                        )
                        seed = stable_group_seed(base_seed, noise_group)
                        clip_id = _model_clip_id(
                            prompt.prompt_id, technique, dynamic_label, replicate,
                            profile=f"{tag}_off{int(branch_offset * 100):04d}",
                        )
                        midi_path = midi_dir / f"{clip_id}.mid"
                        write_violet_midi(
                            midi_path, prompt.notes, keyswitch, cc_events,
                            config.tempo_bpm,
                        )
                        rows.append({
                            "clip_id": clip_id,
                            "source": "model",
                            "model": "VIOLET",
                            "profile": "delayed",
                            "sweep_kind": "config_robustness",
                            "prompt_id": prompt.prompt_id,
                            "pattern": prompt.pattern,
                            "register": prompt.register,
                            "technique": technique,
                            "technique_class": "released" if released else "sustained",
                            "technique_keyswitch": keyswitch,
                            "dynamic_label": dynamic_label,
                            "cc1_initial": 64,
                            "cc1_final": final_cc,
                            "branch_offset_s": branch_offset,
                            "note_onset_s": config.note_onset_seconds,
                            "seed": seed,
                            "base_seed": base_seed,
                            "noise_group": noise_group,
                            "replicate": replicate,
                            "w_tech": default_w_tech,
                            "w_cc": w_cc,
                            "sampling_steps": steps,
                            "midi_path": _relative(midi_path, root),
                            "audio_path": _relative(
                                root / "data" / "model_audio" / profile / tag
                                / f"{clip_id}.wav",
                                root,
                            ),
                            "analysis_tier": "generator_only_exploratory",
                        })
        manifest = manifest_dir / f"{tag}.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        outputs[tag] = manifest
    return outputs
