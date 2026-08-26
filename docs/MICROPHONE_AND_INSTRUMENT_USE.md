# 마이크와 세 바이올린을 연구 자산으로 쓰는 법

## 연구 안에서의 역할

- 약 70만 원 마이크는 실연주 기준선의 측정 장비다. 모델 오디오와 비교할 response feature, noise floor, onset, decay, spectrum을 같은 고정 체계로 수집한다.
- V1/V2/V3는 가격 비교 표본이 아니라 서로 다른 실제 악기의 **domain replication**이다.
- 한 연주자가 core matrix를 담당해 연주자 차이를 줄인다. 다른 두 사람은 파일럿·장비 점검·일부 robustness 반복에 참여할 수 있지만 실력 라벨을 만들지 않는다.

## 첫 세션 전에 남길 것

1. 마이크·오디오 인터페이스의 정확한 제조사/모델/serial
2. cardioid 방향, bridge까지 거리, 높이, azimuth
3. gain knob 사진 또는 디지털 gain 값
4. 연주자 시점과 측면의 setup 사진
5. V1/V2/V3 익명 매핑과 각 악기의 줄·활 상태
6. 30초 room tone

`manifests/session_metadata_template.csv`를 세션마다 복사해 한 줄을 작성한다. 개인정보나 실제 악기 가격은 공개본에서 제거한다.

## gain 결정

1. 실제 본 실험과 같은 거리에서 가장 큰 `f sustain`을 연주한다.
2. peak가 약 -12 dBFS가 되도록 한 번 조정한다.
3. p가 작다고 gain을 다시 올리지 않는다.
4. 세 악기 사이에도 gain을 유지한다. 악기별 절대 level 차이가 생기지만 분석은 같은 셀 내부의 변화량을 우선 사용한다.

SPL calibrator가 없으므로 dBFS를 물리적 음압 레벨로 쓰지 않는다. 이 연구가 비교하는 것은 고정 녹음 체계에서의 상대 반응이다.

## 세 악기의 최소 사용량

- pilot: V1만 사용해 전체 파이프라인을 검증
- full fit: V1·V2 response curve로 보정기 적합
- held-out: V3는 보정기 적합에 넣지 않고 마지막 일반화 검증에만 사용

이 구조라면 세 악기를 모두 필수적으로 사용하면서도 “가격이 비싸면 더 좋다”처럼 현재 표본으로 답할 수 없는 질문을 피한다.

## 녹음 파일 보존

```text
data/real_raw/       원본 WAV; 수정 금지
data/real_48k/       48 kHz mono PCM24 분석본
manifests/           조건과 파일의 대응표
results/*_qc.csv     checksum과 QC 결과
```

원본, manifest, session metadata, room tone, setup photo를 함께 백업한다. 논문 공개 시에는 라이선스와 개인정보 동의를 별도로 확인한다.
