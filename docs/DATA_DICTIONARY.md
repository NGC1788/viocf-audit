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

