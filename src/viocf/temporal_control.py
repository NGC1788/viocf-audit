"""시간 제어 시험 — 제어가 시간 축을 따르는가, 전역 토큰인가.

## 가설

지연 분기 실험에서 **누출(1.397)이 의도한 효과(0.662)보다 2.11배 컸다.**
비인과성만으로는 이 방향이 설명되지 않는다 — 미래 조건이 과거로 새는 것이라면
과거 쪽이 미래 쪽보다 **작아야** 정상이다.

더 잘 맞는 설명이 있다.

    VIOLET 의 강약 제어는 시간 축 제어가 아니라 **클립 전체에 붙는 전역 토큰**
    이며, 시계열의 모습으로 입력받을 뿐이다.

이 가설은 지금까지 관측한 것을 전부 설명한다.

  누출 > 효과            토큰이 창 전체를 물들이므로
  거리 감쇠하나 0 아님    국소 성분 + 전역 성분의 합
  유도 세면 누출비 증가   토큰이 강해지면 더 넓게 물듦
  단조성 위반 11.3 %     미세한 시간 제어가 없음
  감쇠 주법이 더 샘      소리가 시작 시점에 결정되어 전역 성분이 드러남

## 시험

**평균은 같고 모양만 다른** CC1 궤적을 넣는다. 전부 평균 64다.

    constant     64 유지
    ramp_up      32 → 96 선형
    ramp_down    96 → 32 선형
    step_up      전반 32, 후반 96
    step_down    전반 96, 후반 32
    oscillate    32 ↔ 96 을 4회 왕복

  시간 제어가 실재하면  소리 크기가 궤적을 따라간다 (상관 r ≈ 1),
                        모양이 다르면 소리도 다르다
  전역 토큰이면        여섯이 거의 같은 소리를 낸다 (r ≈ 0)

⚠ 양성 대조가 필수다. 인과 기준 렌더러(causal_reference)는 설계상 CC1 을
   그대로 따르므로 r ≈ 1 이 나와야 한다. 그게 안 나오면 측정이 틀린 것이고
   VIOLET 결과를 해석할 수 없다.

## 왜 이게 중요한가

기존 제어형 오디오 평가는 클립 단위 평균 지표나 사람 선호도에 의존한다.
그 지표들은 **진짜 시간 제어와 전역 토큰을 구분하지 못한다** — 평균이 같으면
둘 다 같은 점수를 받는다. 이 시험은 그 구분을 강제한다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig, project_root_from_config
from .design import (
    CCEvent,
    NoteEvent,
    Prompt,
    _model_clip_id,
    _relative,
    stable_group_seed,
)
from .midi import write_violet_midi

# 궤적의 두 값. 평균이 정확히 64(=mf)가 되도록 대칭으로 잡는다.
CC_LOW, CC_HIGH = 32, 96
CC_MEAN = (CC_LOW + CC_HIGH) // 2

# 궤적을 몇 단계로 쪼갤 것인가. VIOLET 은 CC1 을 보간하지 않고 sample-and-hold
# 로 프레임에 깔므로(개정 3 확인), 촘촘한 계단으로 임의 모양을 만들 수 있다.
TRAJECTORY_STEPS = 48

# 시험할 주법. 활이 계속 닿는 쪽만 쓴다 — 감쇠 주법은 시작 뒤 CC1 에 물리적으로
# 반응할 수 없으므로 '시간 제어가 되는가'를 물을 수 없다.
TEMPORAL_TECHNIQUES = ("sustain", "legato_slur", "tremolo")

NOTE_SECONDS = 6.0


def _trajectory(shape: str, steps: int) -> list[float]:
    """0~1 위치마다의 CC1 값. 어떤 모양이든 평균이 CC_MEAN 이어야 한다."""
    positions = [(i + 0.5) / steps for i in range(steps)]
    if shape == "constant":
        values = [float(CC_MEAN)] * steps
    elif shape == "ramp_up":
        values = [CC_LOW + (CC_HIGH - CC_LOW) * p for p in positions]
    elif shape == "ramp_down":
        values = [CC_HIGH - (CC_HIGH - CC_LOW) * p for p in positions]
    elif shape == "step_up":
        values = [float(CC_LOW if p < 0.5 else CC_HIGH) for p in positions]
    elif shape == "step_down":
        values = [float(CC_HIGH if p < 0.5 else CC_LOW) for p in positions]
    elif shape == "oscillate":
        # 4회 왕복. 사각파로 두어 상승/하강과 평균·분산을 맞춘다.
        values = [
            float(CC_HIGH if math.sin(2 * math.pi * 4 * p) >= 0 else CC_LOW)
            for p in positions
        ]
    else:
        raise ValueError(f"모르는 모양: {shape}")
    return values


TRAJECTORY_SHAPES = (
    "constant", "ramp_up", "ramp_down", "step_up", "step_down", "oscillate",
)


def trajectory_table() -> pd.DataFrame:
    """각 모양의 평균·표준편차를 표로. 평균이 흔들리면 시험이 성립하지 않는다."""
    rows = []
    for shape in TRAJECTORY_SHAPES:
        values = np.array(_trajectory(shape, TRAJECTORY_STEPS))
        rows.append({
            "shape": shape,
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        })
    return pd.DataFrame(rows)


def _cc_events(config: ExperimentConfig, shape: str) -> list[CCEvent]:
    """궤적을 CC 사건 목록으로. 음이 울리는 구간에만 깔린다."""
    onset = config.note_onset_seconds
    values = _trajectory(shape, TRAJECTORY_STEPS)
    events = [CCEvent(0.0, CC_MEAN)]
    for index, value in enumerate(values):
        time_s = onset + NOTE_SECONDS * index / TRAJECTORY_STEPS
        events.append(CCEvent(time_s, round(value)))
    return events


def plan_size(replicates: int) -> dict[str, int]:
    clips = len(TEMPORAL_TECHNIQUES) * len(TRAJECTORY_SHAPES) * replicates
    return {
        "techniques": len(TEMPORAL_TECHNIQUES),
        "shapes": len(TRAJECTORY_SHAPES),
        "replicates": replicates,
        "clips_total": clips,
    }


def create_temporal_control(
    config: ExperimentConfig,
    replicates: int = 32,
    profile: str = "temporal_control",
) -> Path:
    """단일 manifest. 샘플러 설정이 하나뿐이라 나눌 필요가 없다."""
    root = project_root_from_config(config)
    base_seed = int(config.raw["model"]["base_seed"])
    keyswitches = dict(config.techniques)

    prompt = Prompt(
        prompt_id="temporal_A4",
        pattern="temporal_long",
        register="mid",
        notes=(NoteEvent(69, config.note_onset_seconds, NOTE_SECONDS),),
        reference_midi=69,
        single_pitch=True,
    )

    midi_dir = root / "data" / "midi" / profile
    rows: list[dict[str, Any]] = []
    for technique in TEMPORAL_TECHNIQUES:
        keyswitch = keyswitches[technique]
        for shape in TRAJECTORY_SHAPES:
            cc_events = _cc_events(config, shape)
            for replicate in range(1, replicates + 1):
                # ⚠ noise_group 에 모양을 넣지 **않는다.**
                # 같은 초기 난수에서 모양만 바꿔야 "모양 때문에 달라졌다"고
                # 말할 수 있다. 이게 이 실험의 핵심 짝이다.
                noise_group = f"{prompt.prompt_id}-{technique}-rep{replicate:02d}"
                seed = stable_group_seed(base_seed, noise_group)
                clip_id = _model_clip_id(
                    prompt.prompt_id, technique, shape, replicate, profile=profile,
                )
                midi_path = midi_dir / f"{clip_id}.mid"
                write_violet_midi(
                    midi_path, prompt.notes, keyswitch, cc_events, config.tempo_bpm,
                )
                rows.append({
                    "clip_id": clip_id,
                    "source": "model",
                    "model": "VIOLET",
                    "profile": "temporal",
                    "sweep_kind": "temporal_control",
                    "prompt_id": prompt.prompt_id,
                    "pattern": prompt.pattern,
                    "register": prompt.register,
                    "technique": technique,
                    "trajectory_shape": shape,
                    "dynamic_label": shape,     # 짝 도구가 이 열을 쓴다
                    "technique_keyswitch": keyswitch,
                    "cc1_mean": CC_MEAN,
                    "cc1_low": CC_LOW,
                    "cc1_high": CC_HIGH,
                    "trajectory_steps": TRAJECTORY_STEPS,
                    "note_onset_s": config.note_onset_seconds,
                    "note_seconds": NOTE_SECONDS,
                    "seed": seed,
                    "base_seed": base_seed,
                    "noise_group": noise_group,
                    "replicate": replicate,
                    "w_tech": float(config.raw["model"]["w_tech"]),
                    "w_cc": float(config.raw["model"]["w_cc"]),
                    "sampling_steps": int(config.raw["model"]["sampling_steps"]),
                    "midi_path": _relative(midi_path, root),
                    "audio_path": _relative(
                        root / "data" / "model_audio" / profile / f"{clip_id}.wav", root,
                    ),
                    "analysis_tier": "generator_only_exploratory",
                })

    manifest_dir = root / "manifests" / profile
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / "trajectories.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest
