# 오늘부터 실행할 일

## 오늘의 완료 조건

오늘은 전체 녹음을 하지 않는다. 다음 여섯 가지를 끝내면 성공이다.

1. 서버 preflight 저장
2. VIOLET 체크포인트 접근 요청
3. pilot MIDI 60개와 녹음 manifest 확인
4. VIOLET 저장소 clone 및 same-noise patch 적용
5. 체크포인트가 있으면 2개 MIDI smoke inference
6. V1으로 실연주 약 12~20개 파일을 녹음해 QC/특징 추출

## 0~1시간: 서버

```bash
cd /서버/프로젝트/viocf-audit-starter
bash scripts/bootstrap_analysis.sh
cat results/preflight.json
```

직접 확인할 값:

- `commands.nvidia_smi.stdout`의 GPU가 `NVIDIA RTX A5000`
- VRAM 약 24GB
- 여유 디스크 최소 50GB, 권장 150GB
- `ready_for_violet_hardware`가 true
- `data/midi/pilot` 검사 결과 invalid 0개

분석용 `.venv`에는 PyTorch를 설치하지 않으므로 이 단계의 `torch.cuda_available` 또는
`ready_for_violet_runtime`이 false여도 정상이다. `scripts/setup_violet_env.sh`가 만드는 별도
`vendor/VIOLET/.venv-violet` 안에서 CUDA available가 true여야 한다.

## 1~2시간: 접근권한

```bash
bash scripts/check_violet_access.sh
```

401이면 `docs/VIOLET_ACCESS_REQUEST.txt` 내용을 GitHub issue 또는 저자 이메일로 보낸다. 답을 기다리면서 아래 작업은 계속한다.

## 2~3시간: VIOLET 코드

```bash
bash scripts/prepare_violet_repo.sh
git -C vendor/VIOLET status --short
```

다음 두 파일만 patch로 수정되어야 한다.

```text
src/inference/render_utils.py
src/models/violin_diffusion_module.py
```

원본 연구 코드의 다른 파일은 임의로 수정하지 않는다.

## 3~4시간: 실험표 검토

다음 CSV를 VS Code로 연다.

```text
manifests/pilot_model.csv
manifests/pilot_real.csv
manifests/pilot_delayed_model.csv
manifests/pilot_delayed_real.csv
```

확인 사항:

- 일반 prompt가 `long_mid`, `scale_mid` 두 개
- 주법이 sustain, staccato, pizzicato, legato_slur 네 개
- 강약이 p/mf/f 세 개
- model의 같은 `noise_group` 안에서 seed가 정확히 하나
- delayed prompt 첫 250ms의 CC1 초기값이 모두 64

## 4~6시간: 마이크 파일럿

V1 한 대와 가장 반복적으로 연주 가능한 한 명만 사용한다.

1. 마이크를 bridge에서 약 70cm, 15cm 위에 설치
2. 가장 큰 f에서 peak -12dBFS 부근으로 gain 설정
3. gain·의자·마이크 위치를 테이프로 고정
4. room tone 30초
5. long_mid의 sustain/pizzicato × p/mf/f를 각 2회 녹음
6. 가능하면 delayed sustain/pizzicato도 녹음

파일명은 manifest의 `raw_audio_path`를 복사해 사용한다. 녹음이 예쁘지 않다는 이유로 삭제하지 말고, clipping·누락음·잘못된 조건만 다시 녹음한다.

```bash
viocf convert-real --manifest manifests/pilot_real.csv
viocf qc --manifest manifests/pilot_real.csv --output results/pilot_real_qc.csv
viocf features --manifest manifests/pilot_real.csv --output results/pilot_real_features.csv
```

오늘의 파일럿 통과 기준:

- clipping 0
- 누락음 0
- 대부분 SNR 30dB 이상
- sustain에서 평균 RMS가 대체로 p < mf < f
- staccato/pizzicato의 active duration 또는 decay가 sustain과 구분
- 파일 처리 오류 0

## 체크포인트가 오늘 확보된 경우

```bash
bash scripts/setup_violet_env.sh
bash scripts/launch_violet_tmux.sh smoke
tmux attach -t viocf_smoke
```

2개 파일이 끝나면 실제 run ID를 넣어 수집한다.

```bash
viocf collect-violet \
  --run-dir logs/violet/smoke/RUN_ID \
  --manifest manifests/smoke_model.csv \
  --output results/smoke_violet_collect.csv
cat results/smoke_violet_collect.summary.json
```

`all_pass=true`일 때만 60개 pilot을 시작한다.

```bash
bash scripts/launch_violet_tmux.sh pilot
```

끝난 뒤 collect 결과에서 모든 noise group의 `pairing_pass=true`를 확인한다. 이 검증 전에는 생성 결과를 통계에 넣지 않는다.

## 오늘 하지 않을 일

- VIOLET 전체 재학습
- full 864-take 녹음
- 12개 주법 전부 시도
- 음원 normalization/denoise/compression
- 바이올린을 저가·중가·고가 성능으로 분석
- 감정 인식 정확도 주장

## 48시간 Go/No-Go

GO:

- 모델 추론이 반복 가능
- 동일 noise group의 실제 render seed가 동일
- 실제 녹음 QC와 특징 추출이 자동으로 동작
- delayed-branch MIDI가 의도한 CC1 분기를 가짐

HOLD:

- 체크포인트만 기다리는 상태. 실악기 pilot과 분석 코드는 계속 진행

NO-GO/주제 조정:

- 4일 내 VIOLET inference 자산을 받지 못함
- same-noise를 재현할 수 없음
- 주법 지시를 현재 연주 수준으로 반복할 수 없음
