# 주요 manifest 열

| 열 | 의미 |
|---|---|
| `clip_id` | MIDI·오디오의 유일 식별자 |
| `source` | `model` 또는 `real` |
| `profile` | `constant` 또는 `delayed` |
| `prompt_id` | 동일 악보 조건 |
| `technique` | 주법명 |
| `technique_keyswitch` | VIOLET keyswitch MIDI pitch |
| `dynamic_label` | 의도한 p/mf/f |
| `cc1_initial`, `cc1_final` | 실제 MIDI CC1 값 |
| `branch_offset_s` | note onset 이후 CC1 분기 시점 |
| `noise_group` | 모델의 동일 initial latent block |
| `seed` | patched VIOLET에서 기대되는 실제 seed |
| `violin_id` | V1/V2/V3 익명 악기 ID |
| `performer_id` | nuisance metadata; 실력 label 아님 |
| `replicate` | model seed 또는 real 반복 index |
| `midi_path` | 프로젝트 루트 기준 MIDI 경로 |
| `audio_path` | 분석에 사용할 48kHz audio 경로 |
| `raw_audio_path` | 원본 실연주 경로 |
| `recording_order` | 피로·학습 편향을 줄이기 위한 무작위 순서 |

절대경로는 공유 manifest에 저장하지 않고, 실행 시 프로젝트 루트를 기준으로 해석한다.


---

# 특징 열 (features CSV)

## 음정 — 열마다 backend 가 다르다. 절대 섞어 쓰지 말 것

| 열 | 의미 | backend |
|---|---|---|
| `f0_cents_error` | **지표가 쓰는 값.** 기준음 대비 중앙 음정(cents) | yin (연속값) |
| `f0_cents_error_configured` | 사전고정 backend 로 잰 같은 양. 비교·보고용 | `analysis.f0_backend` (SwiftF0) |
| `f0_value_backend` | 위 `f0_cents_error` 를 실제로 낸 backend | — |
| `f0_backend` | voiced 판정에 쓴 backend | — |
| `f0_median_hz`, `f0_std_cents`, `f0_voiced_frames` | 설정 backend 기준 보조값 | `analysis.f0_backend` |

> ⚠ SwiftF0 는 거친 음정 격자에 스냅되어 **같은 음 안의 10 cents 이동을 0.0 으로 읽는다.**
> 그래서 음정 누출 지표에는 쓸 수 없다. 근거 실측표는 `docs/EXPERIMENT_SPEC.md` 개정 1.

## 비브라토 (yin 기반)

| 열 | 의미 |
|---|---|
| `vibrato_rate_hz` | 3–9 Hz 대역 우세 주파수 |
| `vibrato_extent_cents` | 비브라토 폭 (peak-to-peak, 준정현파 가정) |
| `f0_mod_std_cents` | 중앙값 대비 음정 변동 표준편차 |
| `f0_mod_backend`, `f0_mod_voiced_frames` | backend 와 유효 프레임 수 |

> SwiftF0 로 폭을 재면 깊이에 대해 **비단조적**이다(E5: 80 c → 15.3 c, 120 c → 134.3 c).

## 활바꿈 / 재발음

| 열 | 의미 |
|---|---|
| `env_dip_depth_db` | 음 사이 RMS 포락선이 파이는 깊이의 중앙값 |
| `env_dip_rate_hz` | 초당 유의미한 골(≥3 dB) 개수 |

> **sustain(데타셰)과 legato_slur 를 구분하는 유일한 특징이다.** 음이 이어지면
> `active_duration_s` 는 포화돼 두 주법이 같은 값이 된다(실측 3.79 s vs 3.79 s).
> dip 으로는 0.0 dB vs 5.0 dB 로 분리된다.

## delayed-branch 전용

| 열 | 의미 |
|---|---|
| `prebranch_rms_dbfs`, `prebranch_centroid_hz` | 분기 **이전** 구간 — 조건이 동일하므로 모델에서는 같아야 한다 |
| `postbranch_rms_dbfs`, `postbranch_centroid_hz` | 분기 **이후** 구간 |
| `post_minus_pre_rms_db` | 분기 전후 레벨 차 |
| `detected_branch_time_s` | 분기 시각(onset + branch_offset_s) |

> 모델 단독 강한 검정은 `delayed_branch_model_only.csv` 에 따로 나온다.
> 같은 noise group 안에서 분기 전 spread 가 0 이 아니면 미래 조건이 과거로 샌 것이다.
> 단, VIOLET 은 full-window diffusion 이라 **구조상 비인과적**이므로 결론은
> "물리 위반"이 아니라 "실시간/대화형 제어 불가"로 쓴다.
