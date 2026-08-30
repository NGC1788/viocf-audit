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

RMS–CC1 Spearman 은 secondary endpoint 다.

**미구현 항목의 지위** — 아래 둘은 이 저장소에 구현돼 있지 않다. 구현 전까지
결과·발표에서 언급하지 않는다(약속만 하고 없는 지표는 그 자체가 결함이다).
- **FAD**: 오디오 품질 지표. 이 연구의 질문(제어 충실도)과 직교하며, 참조 분포 선택이
  결과를 좌우해 논쟁을 부른다. 넣으려면 참조 코퍼스를 명세에 먼저 고정해야 한다.
- **VioPTT accuracy**: [VioPTT](https://arxiv.org/abs/2509.23759) 는 저자 일부가
  VIOLET 과 겹치고 둘 다 합성 학습자료를 쓴다. 그대로 쓰면 순환 평가다. 쓴다면
  **독립적인 주법 검출기**(예: MERTech, arXiv 2310.09853)와 병행해 일치할 때만 인용한다.

## 통계 단위

- frame이나 note를 독립 표본으로 세지 않는다.
- primary generalization unit은 prompt다.
- model seed와 real take/violin은 prompt 내부 반복이다.
- prompt-level **2단계(hierarchical) 부트스트랩**을 사용한다: prompt 를 복원추출한 뒤
  뽑힌 prompt 안에서 관측치를 다시 복원추출한다 (`metrics.bootstrap_mean_ci`).
  클러스터만 재표집하는 1단계 부트스트랩은 클러스터 내부 변동을 무시해 CI 를 좁게 낸다.
- 세 global endpoint(CEA/HCEL/CG)만 confirmatory 로 두고, 특징별 누출 비교는
  **Holm-Bonferroni 보정**한다 (`metrics.holm_adjust`). p 값은 prompt 단위
  부호뒤집기 순열검정으로 낸다 (`metrics.cluster_permutation_pvalue`) — prompt 내부
  상관을 깨지 않기 위해서다. 결과는 `metrics_summary.json` 의
  `leakage_per_feature_tests` 에 원시 p 와 보정 p 가 함께 기록된다.

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


### 2026-08-27 개정 5 — 설계 확장 (GPU 여유를 통계력으로 환산)

A5000 을 한 달 쓸 수 있으나 **추론은 아무리 키워도 며칠**이라, 남는 시간을 클립 수가
아니라 **검정력과 축의 개수**로 바꾼다. 늘린 것과 그 이유:

| 축 | 이전 | 이후 | 왜 |
|---|---|---|---|
| 프롬프트 | 12 | **24** | 일반화 단위를 prompt 로 선언해 놓고 12개만 쓰면 부트스트랩 CI 가 불안정하다(20개 미만이면 경고가 뜬다). 24개로 그 문턱을 넘긴다. |
| 단일음 프롬프트 | 6 | **12** | 음정 특징은 `single_pitch` 프롬프트에서만 나온다. headline 지표 하나인 음정 누출의 표본이 그대로 2배가 된다. |
| 시드 | 5 | **32** | 모델 쪽 표준오차가 1/√n. 검출 가능한 최소 누출이 약 2.5배 작아지고, "시드 하나 잘못 걸린 것 아니냐"가 막힌다. |
| 모델 주법 | 4 | **8** (expanded 한정) | VIOLET 은 12기법을 지원한다. 늘린 4개(spiccato·tremolo·trill_major·harmonic)는 **실악기 기준선이 없다**. 아래 개정 6 참조. |
| 확산 스텝 | 고정 30 | **8/16/30/50/80/120** | 스텝 수 대 *오디오 품질*은 흔히 재지만 스텝 수 대 **제어 충실도**는 아무도 재지 않았다. MIDI 를 새로 만들 필요가 없어 싸고, 실무자에게 바로 쓸모 있다. |

**모델 축과 실연주 축을 분리했다.** `real.techniques` 로 사람이 녹음할 주법을 4개로
묶고, `real.full_violin_prompts` 로 앞 12개 프롬프트만 바이올린 3대 전부, 나머지 12개는
V1 한 대로 녹음한다. 반복 2회는 **양쪽 모두 유지한다** — HCEL 의 기준선이라 줄일 수 없다.

규모: 모델 클립 48,576(보정 재생성 포함 103,872), 실연주 1,224 테이크.
실연주는 이전 936 대비 **+31%** 이며, 이것이 이 확장의 실질 비용이다
(GPU 시간이 아니라 사람의 녹음 시간이 병목이다).


### 2026-08-27 개정 6 — 추가 4주법의 지위를 "탐색"으로 정정

**개정 5 의 오류**: 늘린 4개 주법을 "모델 vs 가상악기 축으로 감사한다"고 적었으나,
**가상악기 기준선은 이 저장소에 구현된 적이 없다.** 근거 없는 비교를 명세에 적어 둔 것이고,
그대로 발표하면 "그 가상악기 데이터는 어디 있느냐"는 질문에 답할 수 없다.

**정정**: 추가 4주법은 **generator-only 탐색 축**이다.
- manifest 의 `analysis_tier` 열로 구분한다
  (`real_counterfactual_primary` / `generator_only_exploratory`).
- `run_metric_suite` 와 `fit_technique_calibration` 은 primary 행만 사용한다.
  탐색 행은 CEA/HCEL/CG 어느 headline 에도 들어갈 수 없다.
- `pilot`·`full` 프로파일은 실악기 대응 4주법만 생성한다(`model_techniques_for_profile`).
  탐색 주법은 `expanded` 와, 명시적으로 요청한 sweep(`--include-exploratory-techniques`)에서만 나온다.
- 탐색 행의 용도는 **대리모델(surrogate) 학습과 실패 영역 탐색**이다.
  "VIOLET 이 이 주법에서 제어를 흘린다"는 주장은 실악기 기준선이 생기기 전까지 하지 않는다.

가상악기 기준선을 정말 넣고 싶다면 별도 개정으로 **데이터 출처·라이선스·렌더 절차**를
먼저 명세에 적고 구현한 뒤에 한다.

### 2026-08-27 개정 7 — 샘플링 스텝 축을 실행 경로에 연결

개정 5 에서 스텝 축은 manifest 와 MIDI 까지만 만들어졌고 **실행기가 그 축을 무시**했다.
그대로 돌렸으면 6개 스텝 조건이 전부 기본값 30 으로 렌더돼 축이 통째로 무의미해진다.
지금은 manifest → `run_compute_sweep.sh` → `run_violet.sh`(Hydra `sampler_steps=`) →
`verify_violet_run.sh` → `collect-violet` 까지 이어지고, 렌더 로그의
`effective_sampling_steps` 를 manifest 기대값과 대조해 어긋나면 `pairing_pass` 가 실패한다.


## 개정 8 — 렌더 실패 판정을 활성 구간 검출기로 (데이터 수집 후, 지표 산출 전)

expanded 전수 QC(18,432 클립)를 돌린 뒤의 정정이다.

**무엇이 틀렸나.** 무음 판정 경계를 peak −35 dBFS 로 잡으면서 "표본에서 정상 셀과
실패 셀이 완전히 분리되고 그 사이가 비어 있다"고 근거를 달았다. 전수 분포는
−65 부근의 얕은 최소를 지나 −30 까지 단조 증가하는 연속 꼬리이고, 검출기가
무음이라 한 클립과 소리가 있다고 한 클립의 peak 범위가 겹친다. **어떤 단일 peak
경계로도 검출기 판정을 재현할 수 없다.** 표본 384개로 본 "분리"는 실재하지 않았다.

**어떻게 고쳤나.** 지표의 제외 기준 1순위를 활성 구간 검출기(`qc_reasons` 의
`near_silence`/`missing_audio`/`non_finite_samples`)로 바꿨다. 이 기준은 조정
상수가 없다 — 어떤 프레임 RMS 도 −60 dBFS 를 넘지 못하면 소리가 없는 것이다.
peak 경계는 QC 미실행 시의 대용으로만 남기고 `PEAK_FALLBACK_DBFS` 로 명시했다.

**전수 결과.** 생성 실패 1,157 / 18,432 = **6.28 %**. 독립적으로 뽑은 표본
추정치(24/384 = 6.25 %)와 일치한다.

## 개정 9 — low_snr 은 품질 게이트가 아니다 (같은 시점)

전수 QC 에서 실패 5,795건 전부에 `low_snr` 이 붙었고, `clipping` 은 0건이었다.
`min_snr_db: 30` 은 실연주 녹음의 방 소음을 전제로 정한 값인데, snr_db 분포는
+5~+45 dB 에 몰려 봉우리가 +35 다. **기준선이 분포 한가운데를 자른다** — 이상치
검출이 아니라 임의 절단이다.

주법별 실패율도 pizzicato 52.0 % vs trill_major 14.2 % 로 갈리는데, 이것이
디코더 노이즈의 성질인지 짧은 주법을 벌주는 측정 인공물인지는 SNR 을 신호항과
노이즈항으로 분해해야 판정된다(`scripts/audit_snr_criterion.py`).

`qc_pass` 는 지표에서 행을 거르지 않으므로 결과가 오염되지는 않았다. 다만
"31.4 % 실패"라는 요약 숫자는 그대로 인용하면 안 된다 — 생성 실패 6.28 % 와
임계값 절단 25.16 % 는 완전히 다른 것이다.
