# 새로움 점검과 관련 연구

기준일: 2026-08-26

## 결론부터

`같은 seed에서 조건만 바꾸는 반사실 평가` 자체는 이제 새롭다고 주장하면 안 된다. 2026-08-12 공개된 *Do Text-to-Music Models Really Follow Instructions?*가 이미 shared-seed matched counterfactual evaluation을 key와 beat grouping에 적용했다.

이 연구의 방어 가능한 핵심은 다음 묶음이다.

> **연속적으로 발음되는 실제 악기를 대상으로, 생성기의 주법·시간가변 강약 제어를 같은 latent에서 개입하고, 실제 바이올린의 대응 변화량을 기준선으로 삼아 누출·조합성·미래조건 누출을 감사한 뒤, 보정하고 재생성하여 다시 감사한다.**

한 요소가 아니라 아래 네 요소의 결합을 contribution으로 둔다.

1. **real-instrument counterfactual reference**: 단순 목표 검출 정확도가 아니라 같은 악보를 실제 바이올린에서 주법·강약만 바꿨을 때 생기는 자연스러운 연동 변화와 비교한다.
2. **continuous local-control audit**: 전역 key/박자가 아니라 음표 단위 주법과 프레임 단위 CC1의 방향, 크기, 단조성, 시간 반응을 평가한다.
3. **leakage + compositionality + temporal causality**: 목표가 변했는지만 보지 않고 비목표 pitch/timing 누출, technique×dynamics 상호작용, 미래 CC1의 분기 이전 누출을 함께 측정한다.
4. **closed-loop calibration**: 문제를 보고하는 데서 끝내지 않고 실제 악기 반응으로 CC1 매핑을 보정하고, 동일 noise로 재생성해 개선과 부작용을 재검증한다.

## 가장 가까운 연구와 정확한 차이

| 연구 | 이미 한 것 | 우리가 추가하는 것 | 피해야 할 주장 |
|---|---|---|---|
| [VIOLET (2026)](https://arxiv.org/abs/2608.07944) | DiT + rectified flow, MIDI·주법·연속 dynamics 조건, pitch/timing/technique/dynamics 평가 | 같은 latent의 factorial 개입, 실제 악기 효과 기준선, 누출·조합성·미래 누출, 보정 후 재감사 | “최초의 제어 가능한 바이올린 생성” |
| [Do Text-to-Music Models Really Follow Instructions? (2026)](https://arxiv.org/abs/2608.11899) | neutral–A–B shared-seed 반사실 평가, key/beat instruction attribution | 실제 악기 기준의 효과 크기, local continuous control, 다중 제어 interaction, temporal branch, closed-loop correction | “음악 생성 최초의 shared-seed 반사실 평가” |
| [Evaluating Disentangled Representations for Controllable Music Generation (2026)](https://arxiv.org/abs/2602.10058) | informativeness/equivariance/invariance/disentanglement probing | 생성 오디오의 개입 결과와 물리적 실제 기준선, 재보정 검증 | “최초의 음악 제어 disentanglement 평가” |
| [Adding temporal musical controls on top of pretrained generative models (ISMIR 2025)](https://ismir2025program.ismir.net/poster_274.html) | 사전학습 모델에 시간가변 control을 추가하고 정확도 평가 | 이미 있는 control의 독립성·물리성·인과성 감사 | “최초의 시간가변 음악 제어” |
| [Music ControlNet](https://arxiv.org/abs/2311.07069) | melody·dynamics·rhythm의 다중 시간가변 제어 | 바이올린 주법×강약의 실악기 대응 변화와 미래정보 placebo | “최초의 다중 시간가변 음악 제어” |

## 발표에서 쓸 한 문장

> 기존 평가는 요청한 주법이나 강약이 출력에 나타났는지를 묻는다. 우리는 **그 변화가 같은 연주를 실제 바이올린에서 바꿨을 때의 변화처럼 크고, 국소적이고, 조합 가능하며, 인과적으로 올바른지**를 묻고, 틀리면 보정 후 다시 검증한다.

## 세 가지 킬러 피겨

1. **Real-effect vector vs model-effect vector**: 주법/강약별 cosine alignment와 magnitude ratio.
2. **Control leakage heatmap**: 목표 제어를 바꿀 때 실제 95% 범위를 넘어 변한 비목표 특징.
3. **Before → calibrated → held-out**: V1·V2로 맞춘 보정이 V3와 동일-noise 재생성에서도 response gap을 줄이는지.

네 번째 보조 그림은 delayed-branch의 분기 이전 `future_leak`과 분기 이후 `post_effect`다.

## 반증 가능한 결과

- VIOLET가 모든 검사를 통과해도 연구는 실패가 아니다. 실제 악기 기준으로 어떤 제어가 어느 범위에서 신뢰 가능한지 처음 정량화한 결과가 된다.
- 보정이 V3에서 개선되지 않으면 “악기 불변 보정” 가설을 기각하고 악기별 calibration이 필요하다는 결과로 쓴다.
- pizzicato에서 delayed dynamics가 거의 반응하지 않는다면 모델 결함이 아니라 실제 물리 기준과 일치할 수 있다. 그래서 절대 변화량이 아니라 real–model gap을 본다.

## 범위를 줄여야 할 때의 우선순위

1. sustain/pizzicato × p/mf/f × long/scale, same-noise, V1 실제 녹음
2. delayed-branch 미래 누출
3. 12 prompt와 V1/V2/V3 일반화
4. dense CC1 response와 보정
5. guidance-weight sweep과 response surrogate

1–2만 완료해도 중간발표용 명확한 문제·방법·예비결과가 성립한다. 3–5는 체크포인트와 시간에 따라 확장한다.
