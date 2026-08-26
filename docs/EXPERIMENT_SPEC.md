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

