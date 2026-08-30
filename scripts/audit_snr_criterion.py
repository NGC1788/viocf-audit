#!/usr/bin/env python3
"""low_snr 실패 4,638건이 모델 결함인지 측정 인공물인지 가른다.

## 전수 QC 가 보여준 것

  low_snr      5,795  31.44 %   (실패 전체와 같다 — 모든 실패에 이 이유가 붙었다)
  near_silence 1,157   6.28 %   (low_snr 의 부분집합)
  clipping         0            (clip_threshold 0.999 는 한 번도 안 걸렸다)

snr_db 분포는 +5 ~ +45 dB 에 몰려 있고 봉우리가 +35 다. 기준선 30 dB 는
**분포 한가운데를 자른다.** 골짜기가 아니다. 즉 이 기준은 "비정상 클립"을
고르는 게 아니라 그냥 분포를 둘로 쪼개고 있다.

## 무엇을 가려야 하는가

snr_db = active_rms_dbfs - noise_dbfs 다. 두 항 중 어느 쪽이 움직여서
주법별 실패율이 pizzicato 52% vs trill 14% 로 갈렸는지가 핵심이다.

  noise_dbfs 가 주법에 따라 변한다
      -> 디코더가 내용에 따라 다른 양의 노이즈를 낸다 = **모델의 성질(결과)**

  noise_dbfs 는 일정한데 active_rms_dbfs 가 변한다
      -> 짧은 주법일수록 활성 구간에 감쇠 꼬리가 길게 포함돼 평균 레벨이 낮다
         = **측정 인공물**. 피치카토는 원래 그런 소리지 실패가 아니다.

이 스크립트는 그 분해를 직접 한다. 추측하지 않는다.

## 그리고 내 판단 오류 하나

render 등급 경계(-35 dBFS)를 표본 384개로 보고 "분포의 골짜기"라고 적었다.
전수 18,432개는 그렇지 않다 — peak_dbfs 는 -65 부근의 얕은 최소를 지나
-30 까지 단조 증가하는 연속 꼬리다. 골짜기가 없다.

그래서 임의 상수 대신 **활성 구간 검출기**를 쓰는 게 옳다. 그쪽은 조정값이
아니라 절대 기준이다(어떤 프레임도 -60 dBFS RMS 를 못 넘으면 소리가 없다).
이 스크립트는 두 기준이 얼마나 다르게 판정하는지, 그리고 경계를 옮기면
결과가 흔들리는지를 함께 낸다.

사용:
  python scripts/audit_snr_criterion.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CONFIGURED_MIN_SNR_DB = 30.0
# 활성 구간 검출기의 절대 바닥. audio.detect_active_region 이 쓰는 값이며
# 조정 대상이 아니다 — 어떤 프레임 RMS 도 이걸 못 넘으면 소리가 없는 것이다.
ACTIVE_FLOOR_DBFS = -60.0


def decompose(frame: pd.DataFrame) -> None:
    """SNR 을 신호항과 노이즈항으로 나눠 어느 쪽이 변동을 만드는지 본다."""
    print("=" * 74)
    print("SNR 분해 — snr_db = active_rms_dbfs - noise_dbfs")
    print("=" * 74)
    for column in ("technique", "dynamic_label", "pattern"):
        if column not in frame.columns:
            continue
        table = frame.groupby(column).agg(
            snr=("snr_db", "mean"),
            signal=("active_rms_dbfs", "mean"),
            noise=("noise_dbfs", "mean"),
            n=("snr_db", "size"),
        )
        print()
        print(f"[{column}]")
        print(table.round(2).to_string())
        signal_spread = float(table["signal"].max() - table["signal"].min())
        noise_spread = float(table["noise"].max() - table["noise"].min())
        print(f"  신호항 변동폭 {signal_spread:6.2f} dB")
        print(f"  노이즈항 변동폭 {noise_spread:6.2f} dB")
        if signal_spread > 2 * max(noise_spread, 1e-9):
            print(f"  -> 신호항이 지배한다. {column}별 SNR 차이는 디코더 노이즈가 아니라")
            print("     '얼마나 계속 소리가 나는가'의 차이다 = 측정 인공물.")
        elif noise_spread > 2 * max(signal_spread, 1e-9):
            print("  -> 노이즈항이 지배한다. 디코더가 내용에 따라 다른 노이즈를 낸다")
            print("     = 보고할 모델 성질.")
        else:
            print("  -> 두 항이 비슷하게 기여한다. 단정하지 말 것.")


def duty_cycle_confound(frame: pd.DataFrame) -> None:
    """활성 구간이 짧을수록 SNR 이 낮은가? 인공물 가설의 직접 검정."""
    needed = {"active_start_s", "active_end_s", "duration_s", "snr_db"}
    if not needed.issubset(frame.columns):
        print("\n(활성 구간 열이 없어 듀티비 검정 생략)")
        return
    data = frame.copy()
    data["duty"] = (
        (data["active_end_s"] - data["active_start_s"]) / data["duration_s"]
    )
    data = data.loc[np.isfinite(data["duty"]) & np.isfinite(data["snr_db"])]
    data = data.loc[data["duty"].between(0, 1.001)]
    if len(data) < 50:
        print("\n(듀티비를 계산할 수 있는 행이 부족하다)")
        return
    print()
    print("=" * 74)
    print("듀티비 가설 — '소리 나는 시간 비율'이 SNR 을 만드는가")
    print("=" * 74)
    correlation = float(np.corrcoef(data["duty"], data["snr_db"])[0, 1])
    print(f"  피어슨 상관 r = {correlation:+.3f}   (n = {len(data):,})")
    bins = pd.cut(data["duty"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.001], right=False)
    table = data.groupby(bins, observed=True).agg(
        snr=("snr_db", "mean"),
        fail_pct=("snr_db", lambda values: float((values < CONFIGURED_MIN_SNR_DB).mean() * 100)),
        n=("snr_db", "size"),
    )
    print(table.round(2).to_string())
    if correlation > 0.3:
        print("  -> 듀티비가 높을수록 SNR 이 높다. low_snr 은 상당 부분")
        print("     '짧은 주법'을 벌주고 있다. 품질 게이트로 쓸 수 없다.")


def boundary_disagreement(frame: pd.DataFrame) -> None:
    """활성 구간 검출기와 peak 임계값이 얼마나 다르게 판정하는가."""
    if "qc_reasons" not in frame.columns or "peak_dbfs" not in frame.columns:
        return
    print()
    print("=" * 74)
    print("두 기준의 불일치 — 활성 구간 검출 vs peak 임계값")
    print("=" * 74)
    reasons = frame["qc_reasons"].fillna("")
    detector_silent = reasons.str.contains("near_silence")
    peak = pd.to_numeric(frame["peak_dbfs"], errors="coerce")
    print(f"  검출기가 '소리 없음'으로 판정: {int(detector_silent.sum()):,}")
    print()
    print("  peak 경계를 옮기면 (검출기 판정을 정답으로 두고)")
    print(f"  {'경계':>8}  {'제외 수':>8}  {'검출기와 불일치':>14}")
    for threshold in (-60.0, -55.0, -50.0, -45.0, -40.0, -35.0, -30.0, -25.0):
        peak_silent = peak < threshold
        disagree = int((peak_silent ^ detector_silent).sum())
        print(f"  {threshold:8.0f}  {int(peak_silent.sum()):8,d}  {disagree:14,d}")
    print()
    print("  불일치가 최소가 되는 경계가 있더라도 그건 검출기를 흉내 낸 값일 뿐이다.")
    print("  검출기를 직접 쓰면 조정할 상수가 사라진다 — 그게 옳은 방향이다.")

    # 검출기가 무음이라 한 클립의 peak 분포. 하나의 peak 경계로 재현 가능한가?
    silent_peaks = peak.loc[detector_silent].dropna()
    if len(silent_peaks):
        print()
        print("  검출기가 무음이라 한 클립들의 peak_dbfs 분위수")
        for q in (0.05, 0.25, 0.5, 0.75, 0.95, 1.0):
            print(f"    {q * 100:5.0f} %  {float(silent_peaks.quantile(q)):7.1f} dBFS")
        overlap = peak.loc[~detector_silent]
        print(f"  소리가 있다고 판정된 클립의 최저 peak: "
              f"{float(overlap.min()):.1f} dBFS")
        if float(silent_peaks.max()) > float(overlap.min()):
            print("  -> 두 집단의 peak 범위가 겹친다. **어떤 단일 peak 경계로도**")
            print("     검출기 판정을 재현할 수 없다. peak 임계값을 쓰면 안 된다.")


def detector_contamination(frame: pd.DataFrame) -> None:
    """검출기의 '무음' 판정이 오염됐는지 본다.

    detect_active_region 의 문턱은

        threshold = max(noise_rms * 10^(12/20),  10^(-60/20))

    이고 noise_rms 는 **클립 앞 0.35 초**의 RMS 다. 설계상 첫 음은 0.75 초에
    들어오므로 그 구간은 비어 있어야 한다. 그런데 VIOLET 이 0.35 초 안에
    무언가를 내면 noise_rms 가 커지고, 문턱이 따라 올라가고, 어떤 프레임도
    그걸 못 넘어서 **소리가 큰 클립이 '무음'으로 판정된다.**

    전수 결과가 정확히 그 증상을 보였다: 검출기가 무음이라 한 클립의
    peak 상위 25 % 가 -27.0 dBFS 이고 최대는 -10.6 dBFS 다. 그건 무음이 아니다.

    그래서 '생성 실패 6.28 %' 는 그대로 쓸 수 없다. 진짜 무음과
    '앞구간 오염으로 오판된 것'을 갈라야 한다.
    """
    if "qc_reasons" not in frame.columns or "noise_dbfs" not in frame.columns:
        return
    print()
    print("=" * 74)
    print("검출기 오염 — '무음' 판정이 앞 0.35 초에 좌우되는가")
    print("=" * 74)
    silent = frame["qc_reasons"].fillna("").str.contains("near_silence")
    subset = frame.loc[silent].copy()
    if subset.empty:
        print("  무음 판정 클립이 없다.")
        return
    noise = pd.to_numeric(subset["noise_dbfs"], errors="coerce")
    peak = pd.to_numeric(subset["peak_dbfs"], errors="coerce")
    # 문턱은 noise + 12 dB 와 -60 dBFS 중 큰 쪽. 앞구간이 조용하면 -60 이 이긴다.
    threshold_dbfs = np.maximum(noise + 12.0, ACTIVE_FLOOR_DBFS)
    contaminated = threshold_dbfs > ACTIVE_FLOOR_DBFS + 0.01
    print(f"  무음 판정 {len(subset):,} 개 중")
    print(f"    앞 0.35 초가 조용해 절대 바닥(-60 dBFS)이 문턱이었다: "
          f"{int((~contaminated).sum()):,}  <- 진짜 무음")
    print(f"    앞 0.35 초에 소리가 있어 문턱이 올라갔다:              "
          f"{int(contaminated.sum()):,}  <- 판정을 믿을 수 없다")
    if int(contaminated.sum()):
        print()
        print("  문턱이 올라간 클립의 peak_dbfs 분위수")
        for q in (0.5, 0.75, 0.95, 1.0):
            print(f"    {q * 100:5.0f} %  {float(peak.loc[contaminated].quantile(q)):7.1f} dBFS")
        loud = int((peak.loc[contaminated] > -30.0).sum())
        print(f"  그 중 peak > -30 dBFS (명백히 소리가 있는 것): {loud:,}")
        if loud:
            print("  -> 이만큼은 **무음이 아닌데 무음으로 세어졌다.** 6.28 % 를 그대로")
            print("     인용하면 안 된다. 절대 기준만 쓰는 재판정이 필요하다:")
            print("     bash scripts/rescore_silence.sh  (프레임 RMS 최대값 vs -60 dBFS)")
    print()
    print("  ※ 앞 0.35 초는 설계상 비어 있어야 한다(note_onset_seconds = 0.75).")
    print("     거기에 소리가 있다는 것 자체가 VIOLET 의 관측 가능한 성질이다.")
    audible_noise = pd.to_numeric(
        frame.loc[~silent, "noise_dbfs"], errors="coerce"
    ).median()
    print(f"     무음 판정 클립의 앞구간 중앙값 {float(noise.median()):.1f} dBFS "
          f"vs 나머지 {float(audible_noise):.1f} dBFS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="expanded")
    parser.add_argument("--qc", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = Path(args.qc) if args.qc else root / "results" / args.profile / "model_qc.csv"
    if not path.is_file():
        print(f"QC 표가 없다: {path}", file=sys.stderr)
        return 2

    frame = pd.read_csv(path)
    for column in ("snr_db", "active_rms_dbfs", "noise_dbfs", "peak_dbfs"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # 무음 클립은 SNR 이 -240 으로 찍혀 평균을 완전히 망가뜨린다. 분해에서는 뺀다
    # (그 클립들은 '소리가 없다'가 이미 결론이라 SNR 을 물을 이유가 없다).
    audible = frame.loc[np.isfinite(frame["snr_db"]) & frame["snr_db"].gt(-100)]
    print(f"전체 {len(frame):,} 클립 중 소리가 있는 {len(audible):,} 개로 SNR 을 분해한다")
    print(f"(무음 {len(frame) - len(audible):,} 개는 SNR 이 정의되지 않아 제외)")
    print()
    print(f"소리가 있는 클립의 snr_db: 중앙값 {audible['snr_db'].median():.1f} dB, "
          f"5~95 % {audible['snr_db'].quantile(0.05):.1f} ~ "
          f"{audible['snr_db'].quantile(0.95):.1f} dB")
    print(f"설정된 기준 {CONFIGURED_MIN_SNR_DB:.0f} dB 는 이 분포의 "
          f"{float((audible['snr_db'] < CONFIGURED_MIN_SNR_DB).mean() * 100):.1f} 분위에 있다")
    print("  -> 기준이 분포 한가운데면 그건 이상치 검출이 아니라 임의 절단이다.")
    print()
    print(f"VIOLET 디코더 바닥 노이즈: 중앙값 "
          f"{audible['noise_dbfs'].median():.1f} dBFS "
          f"(이건 보고할 모델 성질이다 — 실연주 녹음의 방 소음과 다르다)")
    print()

    decompose(audible)
    duty_cycle_confound(audible)
    boundary_disagreement(frame)
    detector_contamination(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
