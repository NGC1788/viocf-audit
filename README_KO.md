# VioCF-Audit

제어 가능한 바이올린 생성기가 실제로 주법과 강약을 제어하는지, 동일 악보·동일 생성 노이즈와 실제 바이올린의 대응 변화량으로 감사하고 보정하는 재현 가능한 연구 저장소다.

연구 제목:

> **제어 가능한 바이올린 생성모델은 실제로 표현을 제어하는가? — 실악기 반사실 기준의 제어 누출·조합성 감사와 폐루프 보정**

영문:

> **Do Controllable Violin Generators Really Control Expression? A Real-Instrument Counterfactual Audit of Leakage, Compositionality, and Closed-Loop Calibration**

먼저 읽을 문서:

- 오늘 바로 할 일: `TODAY.md`
- 9월 말까지 일정: `docs/ROADMAP_TO_PRESENTATION.md`
- 가장 가까운 최신 연구와 차별점: `docs/NOVELTY_AND_RELATED_WORK.md`
- 녹음 절차: `docs/RECORDING_PROTOCOL.md`
- 마이크·세 악기의 연구 내 역할: `docs/MICROPHONE_AND_INSTRUMENT_USE.md`
- 사전고정 분석: `docs/EXPERIMENT_SPEC.md`

## 지금 반드시 알아야 할 상태

- 기준 VIOLET commit: `cf0975a752a7ee3cc6e11bb573f9e47c64a0ef97`
- **체크포인트는 공개돼 있다.** VIOLET 원본 README 가 체크포인트를 `datasets/` 주소로
  링크하는데 그건 401 이라 막힌 것처럼 보인다. 실제 가중치는 `models/` 레포에 게이팅 없이 있다.
  (2026-08-26 확인: `api/datasets/User-tian/VIOLET` → 401 / `api/models/User-tian/VIOLET` → 200, gated=False)
  ```bash
  huggingface-cli download User-tian/VIOLET --local-dir checkpoints/
  #  pretrained_checkpoint/ema_snapshots/ema_prof_99515   581 MB
  #  dacvae_ft/weights.pth                                431 MB
  ```
  → 저자에게 접근 요청 메일을 보낼 필요가 없다. `docs/VIOLET_ACCESS_REQUEST.txt` 는
  레포가 실제로 비공개로 바뀌었을 때만 쓴다.
- 체크포인트와 DACVAE를 실제로 내려받기 전에는 VIOLET 실험을 성공했다고 간주하지 않는다.
- 공식 inference 설정의 기본값은 `sampler_w_cc: 0.0`이다. 이 저장소의 실행 스크립트는 반드시 `w_cc=1`, `w_tech=1`로 덮어쓴다.
- 원본은 파일 순서로 seed를 만들고 조건별 무음 재시도를 한다. `patches/violet_counterfactual_noise.patch`가 이를 same-noise block으로 고친다.
- 2026-08-12 공개된 counterfactual text-to-music 평가가 이미 shared-seed 설계를 사용한다. 따라서 새로움은 shared seed 자체가 아니라 **실악기 효과 기준선 + local continuous control + leakage/interaction/temporal audit + 재보정**의 결합에 둔다.
- **F0 backend 는 역할이 나뉜다.** SwiftF0 0.1.2 는 voiced 판정과 사전고정 비교값
  (`f0_cents_error_configured`)에 쓰고, 지표가 사용하는 `f0_cents_error`·비브라토는
  yin(연속값)으로 낸다. SwiftF0 는 **같은 음 안에서의 음정 차이를 측정하지 못하기 때문이다**
  (10 cents 이동을 0.0 으로 읽는다). 근거와 실측표는 `docs/EXPERIMENT_SPEC.md` 개정 1.
  각 feature 행에 backend 가 기록되며 서로 다른 backend 값을 같은 열에 섞지 않는다.

## 빠른 시작

Ubuntu 서버에서 저장소 루트로 이동한 뒤:

```bash
bash scripts/bootstrap_analysis.sh
```

이 명령은 분석용 가상환경과 SwiftF0를 설치하고, pilot MIDI/manifest를 생성하고, MIDI의 keyswitch와 CC1을 검사한다.

생성되는 pilot 규모:

- 일반 factorial: 모델 48개, 실연주 48개
- delayed-branch: 모델 12개, 실연주 12개

파일은 다음 위치에 생긴다.

```text
manifests/pilot_model.csv
manifests/pilot_real.csv
manifests/pilot_delayed_model.csv
manifests/pilot_delayed_real.csv
data/midi/pilot/
```

## VIOLET 준비

접근 상태 확인:

```bash
bash scripts/check_violet_access.sh
```

저장소 clone과 same-noise patch 적용:

```bash
bash scripts/prepare_violet_repo.sh
```

`results/preflight.json`에서 A5000과 `ready_for_violet_hardware=true`를 확인한 다음:

```bash
bash scripts/setup_violet_env.sh
```

분석용 `.venv`와 VIOLET용 `.venv-violet`은 의존성 충돌을 피하려고 분리한다. CUDA runtime의
최종 판정은 `setup_violet_env.sh` 마지막 출력의 `CUDA available True`다.

체크포인트는 README에 안내된 위치에 직접 놓거나 환경변수로 지정한다.

```bash
export VIOCF_VIOLET_EMA=/absolute/path/to/ema_prof_99515
export VIOCF_DACVAE_CKPT=/absolute/path/to/weights.pth
```

먼저 같은 noise group의 p/f 두 개만 smoke 추론한다.

```bash
bash scripts/launch_violet_tmux.sh smoke
tmux attach -t viocf_smoke
```

수집 결과가 `all_pass=true`이면 pilot 추론을 실행한다.

```bash
viocf collect-violet \
  --run-dir logs/violet/smoke/RUN_ID \
  --manifest manifests/smoke_model.csv \
  --output results/smoke_violet_collect.csv
bash scripts/launch_violet_tmux.sh pilot
```

실행이 끝나면 실제 run directory를 사용해 baseline과 delayed manifest를 각각 수집·검증한다.

```bash
viocf collect-violet \
  --run-dir logs/violet/pilot/RUN_ID \
  --manifest manifests/pilot_model.csv \
  --output results/pilot_violet_collect.csv

viocf collect-violet \
  --run-dir logs/violet/pilot/RUN_ID \
  --manifest manifests/pilot_delayed_model.csv \
  --output results/pilot_delayed_violet_collect.csv
```

`*.summary.json`에서 다음이 모두 참이어야 한다.

- 모든 출력 발견
- noise group마다 `render_seed` 한 개
- `render_attempt=1`
- `effective_w_tech=effective_w_cc=1`

## 실연주 녹음

`manifests/pilot_real.csv`의 `recording_order` 순서로 한 명이 V1만 연주한다. 파일명은 CSV의 `raw_audio_path`와 정확히 일치시킨다.

```text
data/real_raw/r__...__v-V1__take01.wav
```

전체화면 무음 시각 cue와 중단 복구 로그를 사용한다. 이 도구는 DAW를 대신 녹음하지 않고,
조건·파일명·delayed 전환 시점을 보여주며 채택/재촬영/건너뛰기를 기록한다.

```bash
python scripts/recording_cue.py manifests/pilot_real.csv --windowed
python scripts/recording_cue.py manifests/pilot_delayed_real.csv --windowed
```

GUI가 없는 서버에서는 `--terminal`을 사용한다. 녹음은 마이크가 연결된 컴퓨터에서 실행하는
편이 낫다. 로그는 `data/recording_sessions/*.jsonl`에 append-only로 저장되어 비정상 종료 후에도
이어갈 수 있다.

원본은 48k 또는 96kHz, 24-bit mono로 받아도 되며 분석본은 다음처럼 통일한다.

```bash
viocf convert-real --manifest manifests/pilot_real.csv
viocf convert-real --manifest manifests/pilot_delayed_real.csv
```

녹음 세부 규칙은 `docs/RECORDING_PROTOCOL.md`를 따른다.

## QC, 특징, 지표

```bash
viocf qc \
  --manifest manifests/pilot_real.csv \
  --output results/pilot_real_qc.csv

viocf features \
  --manifest manifests/pilot_real.csv \
  --output results/pilot_real_features.csv

viocf features \
  --manifest manifests/pilot_model.csv \
  --output results/pilot_model_features.csv

viocf metrics \
  --features results/pilot_real_features.csv results/pilot_model_features.csv \
  --output-dir results/pilot_metrics

viocf figures \
  --features results/pilot_real_features.csv results/pilot_model_features.csv \
  --metrics-dir results/pilot_metrics \
  --output-dir results/pilot_figures
```

주요 산출물:

- `effect_alignment.csv`: 실제 효과와 모델 효과의 방향·크기 정렬
- `excess_leakage.csv`: 실제 95% 변동범위를 넘은 비목표 변화
- `compositionality_gap.csv`: 주법×강약 interaction 차이
- `delayed_branch.csv`: 미래 CC1 누출과 sustain/pizzicato 물리성
- `fig*.png`: 발표용 핵심 그림

## 폐루프 보정

전체 녹음과 생성이 끝난 뒤 V1·V2로 CC1 보정기를 맞추고 V3는 held-out으로 둔다.

```bash
viocf calibrate \
  --features results/full_real_features.csv results/full_model_features.csv \
  --fit-violins V1,V2 --heldout-violin V3 \
  --output-dir results/calibration

viocf apply-calibration \
  --manifest manifests/full_model.csv \
  --mapping results/calibration/cc1_calibration_curve.csv \
  --output manifests/full_model_calibrated.csv
```

새 MIDI를 다시 VIOLET으로 생성한 후 같은 QC·features·metrics를 반복한다. 보정기의 CSV에 기록된 `surrogate_after_error`는 예측값일 뿐이며, 실제 개선 주장은 반드시 재생성 결과로 해야 한다.

## 전체 실험 생성

pilot의 Go/No-Go를 통과한 뒤에만 실행한다.

```bash
viocf make-design --profile full
```

- 일반 factorial: 모델 720개, 실연주 864개
- delayed-branch: 모델 30개, 실연주 72개

## A5000 장기 compute sweep

pilot의 모든 gate를 통과한 뒤에만 만든다. 기본값은 dense CC1 response 6,912개와
guidance-weight ablation 4,608개, 합계 11,520개다.

```bash
viocf make-sweep
VIOCF_SWEEP_DRY_RUN=true bash scripts/run_compute_sweep.sh
```

계획표가 정확하면 tmux에서 실행한다.

```bash
tmux new-session -s viocf_sweep \
  "cd '$PWD' && VIOCF_SWEEP_RUN_ID=2026sep bash scripts/run_compute_sweep.sh 2>&1 | tee logs/viocf_sweep.log"
```

완료 후 동일한 run ID로 수집·분석한다.

```bash
VIOCF_SWEEP_RUN_ID=2026sep bash scripts/collect_compute_sweep.sh
bash scripts/analyze_compute_sweep.sh
```

`results/sweep/surrogate/`에는 prompt-grouped cross-validation 지표, prediction, feature
importance와 ExtraTrees response surrogate가 생성된다. feature importance는 예측 설명이지
인과효과가 아니므로 최종 주장은 실제 재생성 개입으로 확인한다.

`w_tech=0, w_cc>0`은 VIOLET의 compositional guidance 식에서 순수 dynamics-only branch가
아니다. 실행 스크립트도 이를 diagnostic mixture라고 경고한다.

## 연구에서 주장하지 않는 것

- V1/V2/V3 가격대가 음질이나 성능을 결정한다는 주장
- 세 사람의 실력 비교
- 청취자 없이 감정·자연스러움·선호도가 향상됐다는 주장
- 모든 비목표 특징 변화가 나쁘다는 주장
- VioPTT 하나만으로 주법 성공 여부를 판정하는 것
- CSV-TD나 VIOLET 체크포인트의 재배포

코드에는 MIT 라이선스를 적용했지만, VIOLET·체크포인트·데이터는 각 원 저작자의 이용조건을 별도로 따른다.
