# 9월 말 중간발표까지 실행 로드맵

기준일은 2026-08-26이며, 발표일을 9월 30일로 가정한다. 실제 발표가 더 빠르면 각 단계의 `최소 산출물`까지만 수행한다.

## 최종 목표

중간발표에서 완성 모델을 약속하는 것이 아니라 다음 네 가지를 증명한다.

1. 최근 연구와 겹치지 않는 연구질문이 사전고정되어 있다.
2. same-noise 반사실 생성과 실악기 녹음이 실제로 작동한다.
3. 최소 한 개의 제어 실패 또는 통과 사례를 정량 그림으로 보인다.
4. full experiment와 closed-loop 보정까지 갈 재현 가능한 파이프라인이 있다.

## 8월 26–28일: 인프라와 12-cell pilot

완료 조건:

- 서버 preflight JSON
- VIOLET 저장소와 patch 검증
- checkpoint/DACVAE 다운로드 완료 (models 레포에 공개, 접근 요청 불필요)
- V1에서 `long_mid × sustain/pizzicato × p/mf/f × 2회` 12개 녹음
- 같은 12개 조건의 모델 생성
- QC와 첫 response plot

Go/No-Go:

- 서버 접속이 막히면 실연주·분석은 계속한다.
- patch 적용 또는 same-noise 검증이 실패하면 model 결과를 통계에 넣지 않는다.

## 8월 29일–9월 4일: 전체 pilot 60개와 첫 킬러 피겨

작업:

- `pilot_real.csv` 48개와 `pilot_delayed_real.csv` 12개 녹음
- 모델 60개 생성·수집
- effect alignment, excess leakage, compositionality, future leak 계산
- 각 조건 3개 파일을 수동 청취하여 자동 특징 오류 기록

최소 산출물:

- 같은 seed에서 p/mf/f 파형 또는 RMS 곡선 한 장
- real vs model effect-vector 그림 한 장
- delayed prefix/post-branch 그림 한 장

## 9월 5–11일: 설계 동결과 재현성

작업:

- pilot에서 실패한 prompt·주법만 수정하고 manifest v1 동결
- `full`의 12 prompt가 현재 연주 수준에서 반복 가능한지 dry run
- 마이크 위치·gain·room tone 규칙 동결
- 분석 특징의 오류율과 제외 기준 동결
- seed pairing manifest를 보존

결정:

- 네 주법 모두 불안정하면 sustain/staccato/pizzicato 세 개로 줄인다.
- delayed 연주가 재현 불가능하면 delayed real은 sustain/pizzicato의 방향성 기준만 사용하고 정밀 시간 비교 주장을 낮춘다.

## 9월 12–18일: 본 실험과 서버 장기 실행

우선순위:

1. full base grid
2. dense CC1 sweep
3. V1/V2/V3 실제 녹음
4. guidance sweep

A5000은 생성 작업에 사용하고, CPU 분석은 생성 완료 파일에 증분 실행한다. 모든 장기 실행은 tmux 로그와 run manifest를 남긴다. GPU를 한 달 쓸 수 있어도 full retraining은 하지 않는다. 현재 병목은 학습량도 체크포인트 접근도 아니다(둘 다 해소). **서버 접속과 실악기 녹음**이다.

최소 산출물:

- prompt를 cluster로 한 bootstrap CI
- 주법별 CC1 response curve
- leakage heatmap

## 9월 19–23일: closed-loop 보정

작업:

- V1/V2 response로 technique-aware isotonic mapping 적합
- 보정 MIDI 생성
- 원본과 같은 noise group으로 재생성
- V3 held-out에서 real–model gap, pitch/onset 부작용 비교

성공 기준은 `예측상 개선`이 아니라 재생성된 오디오에서의 개선이다.

## 9월 24–27일: 통계와 발표자료

슬라이드 순서:

1. 왜 controllability의 기존 정확도만으로 부족한가
2. 가장 가까운 2026 counterfactual 연구와 우리의 차이
3. VioCF-Audit 설계: same score + same noise + real counterpart
4. 세 지표와 delayed placebo
5. 데이터 수집 사진/마이크/세 바이올린의 역할
6. 예비결과 3개
7. closed-loop 보정 결과 또는 실행 계획
8. 한계와 다음 단계

그래프마다 `n`, prompt 수, seed/take 수, CI 단위를 적는다. 프레임 수를 표본 수처럼 쓰지 않는다.

## 9월 28–30일: 동결과 리허설

- 발표 48시간 전부터 새 모델·새 지표를 추가하지 않는다.
- 결과 CSV, 그림, 실행 config, commit hash를 한 폴더에 동결한다.
- “체크포인트가 없으면?”, “사람 세 명이 일반화가 되나?”, “악기 가격 연구인가?”, “반사실 평가는 이미 있지 않나?”에 20초 답변을 준비한다.

## 매일 남길 것

```text
date / operator / git commit / command / config / run_id
input manifest hash / expected N / completed N / excluded N + reason
GPU hours / observations / next decision
```

연구 노트에는 좋은 결과뿐 아니라 실패한 run과 설계 변경 이유도 기록한다.

## 중간발표 최소 성공선

다음만 있어도 발표가 성립한다.

- V1 12개 실연주와 대응 모델 12개
- same-noise 검증 로그
- 강약 response real-vs-model 그림
- delayed future-leak 한 그림
- full 설계와 보정 계획

반대로 많은 오디오를 만들었지만 seed pairing과 실제 기준선이 검증되지 않았다면 발표 자료로 쓰지 않는다.
