# 오늘부터 실행할 일

## 오늘의 완료 조건

오늘은 전체 녹음을 하지 않는다. 다음 여섯 가지를 끝내면 성공이다.

1. 서버 preflight 저장
2. VIOLET 체크포인트 **다운로드**(접근 요청 불필요 — 아래 참조)
3. pilot MIDI 60개와 녹음 manifest 확인
4. VIOLET 저장소 clone 및 same-noise patch 적용
5. 체크포인트가 있으면 2개 MIDI smoke inference
6. V1으로 실연주 약 12~20개 파일을 녹음해 QC/특징 추출
7. `pytest -q` 로 분석기 검증 34개 통과 확인 (특히 test_analyzer_ground_truth.py)

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

## 1~2시간: 체크포인트 내려받기

```bash
bash scripts/check_violet_access.sh
bash scripts/download_violet_checkpoints.sh
```

이 스크립트는 두 파일을 재개 가능하게 받은 뒤 **크기와 SHA-256 을 대조**하고,
runner 가 실제로 읽는 위치(`vendor/VIOLET/checkpoints`)에 놓는다.
`huggingface-cli download ... --local-dir checkpoints/` 는 저장소 루트에 받으므로
runner 가 못 찾는다 — 쓰지 말 것.

`200 (public, ungated)` 이 나오면 바로 받으면 된다.

> VIOLET 원본 README 는 체크포인트를 `huggingface.co/datasets/User-tian/VIOLET` 로 링크하는데
> 그 주소는 401 이다. 실제 가중치는 `huggingface.co/User-tian/VIOLET`(models 레포)에
> **게이팅 없이 공개**돼 있다. 저자에게 메일 보낼 필요 없다.
> `docs/VIOLET_ACCESS_REQUEST.txt` 는 레포가 실제로 비공개가 됐을 때만 쓴다.

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

## 체크포인트를 받은 뒤 (오늘 바로 가능)

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

- 서버 접속이 안 되는 상태. 실악기 pilot 과 분석 코드는 계속 진행
  (체크포인트 대기는 더 이상 HOLD 사유가 아니다 — 공개 확인됨)

NO-GO/주제 조정:

- 4일 내 VIOLET inference 를 한 번도 돌리지 못함 (자산은 이미 공개이므로 이건 환경 문제다)
- same-noise를 재현할 수 없음
- 주법 지시를 현재 연주 수준으로 반복할 수 없음
