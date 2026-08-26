# 실악기 녹음 프로토콜

## 역할

- 가장 반복적으로 연주할 수 있는 한 명이 core matrix를 담당한다.
- 연주자 실력은 연구변수가 아니다.
- 세 악기는 V1, V2, V3라는 물리적 domain replication이다.
- 가격과 품질의 인과관계를 주장하지 않는다.

## 고정 조건

- 48k 또는 96kHz, 24-bit mono raw master
- cardioid microphone을 bridge 약 70cm, bridge보다 약 15cm 위에 설치
- 실제 각도와 거리는 파일럿 후 고정하고 사진·줄자·바닥 테이프로 기록
- 가장 큰 f에서 peak 약 -12dBFS가 되도록 한 번만 gain 설정
- 세션 도중 gain, mic pad, interface setting 변경 금지
- EQ, HPF, compressor, limiter, AGC, normalization, denoise, reverb 금지
- A=440, 같은 활·송진·어깨받침과 동일 운지/보잉 사용
- 메트로놈은 이어폰으로만 듣고 녹음에 bleed되지 않게 함
- HVAC·창문·가구 배치를 유지하고 온도·습도를 기록

절대 SPL calibrator가 없다면 결과는 상대 dBFS 변화량으로만 보고한다.

## 세션 순서

1. 세션 metadata 작성
2. room tone 30초
3. 가장 큰 f 조건으로 clipping 검사
4. manifest의 `recording_order` 순서로 녹음
5. 20~25분마다 휴식
6. 세션 종료 room tone 30초
7. 녹음 직후 checksum/백업

## 연주 규칙

- 가능한 한 무비브라토
- sustain/détaché: 음 길이를 유지하고 활을 계속 접촉
- legato_slur: 같은 음열을 지정된 slur 단위로 연결
- staccato: on-string 짧은 분리
- pizzicato: 오른손 pizzicato, 재발음 없이 자연 감쇠
- p/mf/f는 상대적 지시이며 실제 달성 RMS도 함께 측정

spiccato는 파일럿에서 영상으로 off-string 상태를 안정적으로 확인할 수 있을 때만 추가한다.

## Delayed-branch

- 첫 250ms는 모든 branch를 mf로 유지
- 이후 p/mf/f로 분기
- sustain은 이후 bow energy를 변화시킬 수 있음
- single pizzicato는 재-pluck하지 않음
- pizzicato가 250ms 뒤 다시 커지는 모델 결과는 물리적 오류 후보

연주자가 미리 branch를 알고 onset을 다르게 만들 수 있으므로, 최종 실험에서는 가능한 경우 250ms 시점에 시각 cue를 무작위 제시한다. 실제 반응지연은 metadata에 남기며 모델과 사람의 즉시 반응시간을 동일하다고 주장하지 않는다.

## QC 실패 기준

- clipping sample 1개 이상
- dropout 또는 손상 파일
- 잘못된 주법/강약/악기
- 음 누락·추가
- 눈에 띄는 외부 소음
- 마이크·gain 변경

단순히 음색이 마음에 들지 않거나 연주가 덜 예쁘다는 이유로 제외하지 않는다. 원본 실패 파일도 별도 보존한다.

