# 사전고정 실험 명세

## 연구질문

1. 동일 MIDI와 동일 초기 latent에서 주법·강약 개입이 의도한 출력 변화를 만드는가?
2. 비목표 특징 변화가 실제 바이올린의 자연 변동범위를 초과하는가?
3. 주법과 강약의 interaction이 실악기의 interaction과 일치하는가?
4. technique-aware CC1 보정이 held-out 악기에서도 response gap을 줄이는가?
5. 미래의 CC1 조건이 분기 이전 오디오로 새거나, pizzicato에 물리적으로 불가능한 후기 에너지를 만드는가?

## 가설

- H1: p<mf<f RMS 단조성은 평균적으로 나타나지만 technique별 포화와 크기 오차가 존재한다.
- H2: duration/register OOD에서 pitch·timing leakage가 실제 기준보다 커진다.
- H3: technique×dynamics interaction의 모델 분포는 실연주 분포와 다르다.
- H4: technique-aware monotonic calibration은 response alignment를 개선하면서 pitch/onset을 보존한다.
- H5: full-window 생성기는 delayed branch의 미래 CC1을 분기 이전 오디오에 누출할 수 있다.

## 반사실 효과

특징 벡터를 `z`라고 하고 기준을 sustain/mf로 둔다.

```text
Delta_T = z(t, mf) - z(sustain, mf)
Delta_D = z(t, d)  - z(t, mf)
I(t,d)  = z(t,d) - z(t,mf) - z(sustain,d) + z(sustain,mf)
```

모든 delta는 model에서는 prompt×initial-noise 내부, real에서는 prompt×performer×violin×take 내부에서 계산한다.

## Primary endpoint

1. Effect Alignment cosine 및 magnitude ratio
2. Human-calibrated Excess Leakage
3. Human-calibrated Compositionality Gap

RMS–CC1 Spearman, FAD, VioPTT accuracy는 secondary endpoint다.

## 통계 단위

- frame이나 note를 독립 표본으로 세지 않는다.
- primary generalization unit은 prompt다.
- model seed와 real take/violin은 prompt 내부 반복이다.
- prompt-level hierarchical bootstrap을 사용한다.
- 세 global endpoint만 confirmatory로 두고 세부 비교는 Holm 보정한다.

## Calibration split

- real V1/V2: 보정기 적합
- real V3: held-out 검증
- generator base output과 calibrated output은 동일 initial-noise group 사용
- 보정 후 pitch/onset 성능 저하도 함께 보고



---

## 개정 이력 (Preregistration amendments)

사전고정 명세는 **데이터 수집 시작 전에만** 고칠 수 있고, 고칠 때는 이유와 근거를 남긴다.
아래 개정은 전부 **실측 데이터가 한 건도 수집되기 전**에 이루어졌다(오디오 0개 상태).

### 2026-08-26 개정 1 — F0 주 backend 를 SwiftF0 에서 yin 으로

**원래**: "논문 분석의 F0 backend 는 SwiftF0 0.1.2 로 고정한다."

**문제**: primary endpoint 중 하나가 "강약 개입 시 음정이 얼마나 새는가(cents)"인데,
SwiftF0 는 **같은 음 안에서의 음정 차이를 측정하지 못한다.** 합성 신호(2.5 s)로 실측:

| 음 | 주입한 차이 | SwiftF0 측정 | yin 측정 |
|---|---|---|---|
| G3 | 10 c | **0.0 c** | 9.8 c |
| A4 | 50 c | 66.1 c | 50.1 c |
| G5 | 25 c | **−0.3 c** | 25.7 c |
| G5 | 50 c | 21.9 c | 49.5 c |

SwiftF0 는 거친 음정 격자에 스냅되어 10 cents 이동을 0 으로 읽는다. 이 상태로 실험했다면
조건 간 음정 차이가 항상 0 근처로 나와 **"VIOLET 은 음정 누출이 없다"는 거짓 음성**이
headline 결론이 됐을 것이다.

**개정**: 역할을 분리한다.
- SwiftF0: voiced 판정 + 사전고정 비교값 `f0_cents_error_configured` (버리지 않고 함께 보고)
- yin(연속값, 격자 없음): 지표가 사용하는 `f0_cents_error`, `f0_mod_std_cents`, 비브라토
- 두 backend 값을 같은 물리량 열에 **절대 섞지 않는다.** 행마다 backend 를 기록한다.

**논문 보고 방식**: 두 backend 결과를 모두 싣고, SwiftF0 의 격자 한계를 분석기 검증
절(section)에 명시한다. 회귀 방지 테스트는 `tests/test_analyzer_ground_truth.py`.

### 2026-08-26 개정 2 — F0 탐색 하한에 여유 추가

**문제**: `fmin` 이 악보 최저음 G3(195.998 Hz)와 **정확히 같았다.** 비브라토는 기준음
아래로도 흔들리므로 아래쪽 절반이 통째로 버려졌다. G3 + 80 cents 비브라토 실측:
유효 프레임 57%, 중앙값이 **+29.6 cents** 로 부풀고 폭은 35 c 로 축소.
저음역에서 **존재하지 않는 음정 누출**이 관측될 상태였다.

**개정**: `fmin` 을 최저음보다 4반음 아래로 내린다(backend 모델 한계로 클램프).
수정 후 같은 신호에서 유효 100%, 중앙 +2.3 c, 폭 65.8 c.

### 2026-08-26 개정 3 — 특징 3종 추가

- `env_dip_depth_db`, `env_dip_rate_hz`: 활바꿈/재발음 깊이.
  없으면 **sustain(데타셰)과 legato_slur 를 구분할 수 없다.** 실측으로 두 주법의
  `active_duration_s` 가 3.79 s 로 동일하게 나왔고(구분 불가), dip 은 0.0 vs 5.0 dB 로 분리됐다.
  4개 주법 중 하나가 측정 불가면 technique contrast 와 interaction 이 무의미해진다.
- `vibrato_extent_cents`, `vibrato_rate_hz`: 비브라토. yin 기반.
  SwiftF0 로 재면 깊이에 대해 **비단조적**이다(E5: 80 c 주입 → 15.3 c, 120 c 주입 → 134.3 c).

### 2026-08-26 개정 4 — z 단위의 정의를 명확히

`1 z` = **같은 조건을 다시 연주했을 때의 흔들림(조건 내 IQR 의 중앙값)**.
기존 구현은 조건 전체를 뭉친 IQR 을 썼는데, 그건 "p 부터 f 까지 전체 폭"이라 훨씬 크고
모델의 이탈을 실제보다 작아 보이게 만든다. 명세가 선언한 "실연주 자연 변동" 과도 다르다.
조건 내 반복이 없어 후퇴한 특징은 `robust_feature_scales` 에 기록되어 감사 가능하다.
