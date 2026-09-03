#!/usr/bin/env bash
# 맥에서 오디오 인터페이스로 처리 없이 녹음한다 (ffmpeg + avfoundation).
#
# 왜 ffmpeg 인가: 대부분의 녹음 앱은 자동 이득·노이즈 억제·음성 강조를 기본으로
# 켠다. 이 연구의 주축이 세기(dB)라 그런 처리가 하나라도 걸리면 p 와 f 가 같은
# 레벨로 눌려 전부 못 쓰게 된다. ffmpeg 는 장치가 주는 표본을 그대로 받는다.
#
# 사용:
#   scripts/record_mac.sh --list                     장치 목록
#   scripts/record_mac.sh --test                     30초 점검용 녹음 + 자동 검사
#   scripts/record_mac.sh --out 세션.wav --seconds 0  본 녹음 (Ctrl+C 로 종료)
set -euo pipefail

VIOCF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIOCF_DEVICE="${VIOCF_AUDIO_DEVICE:-}"
VIOCF_RATE=48000
VIOCF_SECONDS=0
VIOCF_OUT=""
VIOCF_MODE="record"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) VIOCF_MODE="list"; shift ;;
    --test) VIOCF_MODE="test"; shift ;;
    --device) VIOCF_DEVICE="$2"; shift 2 ;;
    --out) VIOCF_OUT="$2"; shift 2 ;;
    --seconds) VIOCF_SECONDS="$2"; shift 2 ;;
    *) echo "알 수 없는 인자: $1"; exit 2 ;;
  esac
done

if [[ "${VIOCF_MODE}" == "list" ]]; then
  echo "오디오 입력 장치 (AVFoundation index):"
  # ffmpeg 는 장치 목록을 stderr 로 낸다.
  ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 \
    | sed -n '/AVFoundation audio devices/,$p'
  echo
  echo "Scarlett 이 보이면 그 번호를 --device 로 넘기거나"
  echo "  export VIOCF_AUDIO_DEVICE=<번호>"
  exit 0
fi

if [[ -z "${VIOCF_DEVICE}" ]]; then
  echo "장치를 지정해야 한다. 먼저: scripts/record_mac.sh --list"
  exit 2
fi

if [[ "${VIOCF_MODE}" == "test" ]]; then
  VIOCF_OUT="${VIOCF_OUT:-${VIOCF_ROOT}/data/real_raw/setup_test.wav}"
  VIOCF_SECONDS=21
  echo "=============================================================="
  echo "점검용 녹음 21초"
  echo "=============================================================="
  echo "  3초  가만히      (무음)"
  echo "  5초  아주 여리게  (p)   같은 음 하나를 계속"
  echo "  5초  아주 세게    (f)"
  echo "  5초  다시 여리게  (p)"
  echo "  3초  가만히      (무음)"
  echo
  echo "  ※ 세기 차이를 **최대한 크게** 낼 것. 그래야 눌렸는지 알 수 있다."
  echo
  for i in 3 2 1; do printf "\r  시작까지 %d " "$i"; sleep 1; done
  printf "\r  시작!        \n"
fi

[[ -n "${VIOCF_OUT}" ]] || { echo "--out 이 필요하다"; exit 2; }
mkdir -p "$(dirname "${VIOCF_OUT}")"

# -ac 1 : NTG-3 는 인터페이스 1번 입력 하나뿐이므로 모노로 받는다.
# pcm_s24le : 24비트. 연구 설정(48 kHz / PCM24)과 맞춘다.
# -af 없음 : 필터를 하나도 걸지 않는다. 이게 핵심이다.
VIOCF_ARGS=(-hide_banner -loglevel warning -f avfoundation
            -ar "${VIOCF_RATE}" -ac 1 -i ":${VIOCF_DEVICE}"
            -c:a pcm_s24le -ar "${VIOCF_RATE}" -ac 1)
if [[ "${VIOCF_SECONDS}" != "0" ]]; then
  VIOCF_ARGS+=(-t "${VIOCF_SECONDS}")
else
  echo "  녹음 중 — 끝나면 Ctrl+C"
fi

ffmpeg "${VIOCF_ARGS[@]}" -y "${VIOCF_OUT}" || true
echo
echo "저장: ${VIOCF_OUT}"

if [[ "${VIOCF_MODE}" == "test" ]]; then
  echo
  if [[ -x "${VIOCF_ROOT}/.venv-check/bin/python" ]]; then
    PYTHONPATH="${VIOCF_ROOT}/src" "${VIOCF_ROOT}/.venv-check/bin/python" \
      "${VIOCF_ROOT}/scripts/check_recording_setup.py" "${VIOCF_OUT}"
  else
    echo "점검을 직접 돌릴 것:"
    echo "  python scripts/check_recording_setup.py ${VIOCF_OUT}"
  fi
fi
