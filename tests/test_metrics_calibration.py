from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from viocf.calibration import fit_technique_calibration
from viocf.metrics import run_metric_suite


def synthetic_feature_table() -> pd.DataFrame:
    rows = []
    techniques = ["sustain", "staccato", "pizzicato", "legato_slur"]
    dynamics = {"p": (32, -6.0), "mf": (64, 0.0), "f": (96, 6.0)}
    technique_rms = {"sustain": 0.0, "staccato": -1.5, "pizzicato": -2.0, "legato_slur": 0.5}
    for source in ["real", "model"]:
        for prompt_index in range(4):
            prompt = f"p{prompt_index}"
            units = ["V1:1", "V2:1", "V3:1"] if source == "real" else ["seed1", "seed2"]
            for unit_index, unit in enumerate(units):
                for technique in techniques:
                    for dynamic_label, (cc1, real_delta) in dynamics.items():
                        dyn_delta = real_delta if source == "real" else 0.75 * real_delta
                        interaction = 0.4 if technique == "pizzicato" and dynamic_label == "f" else 0.0
                        rows.append(
                            {
                                "clip_id": f"{source}-{prompt}-{unit}-{technique}-{dynamic_label}",
                                "source": source,
                                "profile": "constant",
                                "prompt_id": prompt,
                                "technique": technique,
                                "dynamic_label": dynamic_label,
                                "cc1_final": cc1,
                                "replicate": unit_index + 1,
                                "noise_group": unit if source == "model" else np.nan,
                                "performer_id": "P1" if source == "real" else np.nan,
                                "violin_id": unit.split(":")[0] if source == "real" else np.nan,
                                "take": 1 if source == "real" else np.nan,
                                "rms_dbfs": -30 + technique_rms[technique] + dyn_delta + interaction,
                                "attack_time_s": 0.12 - 0.02 * (technique in {"staccato", "pizzicato"}),
                                "spectral_centroid_hz": 1800 + 80 * real_delta + 100 * (technique == "pizzicato"),
                                "spectral_rolloff_hz": 4200 + 120 * real_delta,
                                "hnr_db": 12 - 0.1 * real_delta,
                                "f0_cents_error": (1.0 if source == "real" else 1.5) * (cc1 - 64) / 64,
                                "onset_time_s": 0.5 + 0.005 * (technique == "staccato"),
                                "onset_count": 1,
                                "active_duration_s": 2.0 - 1.2 * (technique in {"staccato", "pizzicato"}),
                            }
                        )
    return pd.DataFrame(rows)


CONFIG = {
    "analysis": {
        "baseline_technique": "sustain",
        "baseline_dynamic": "mf",
        "bootstrap_iterations": 30,
        "random_seed": 123,
        "response_features": [
            "rms_dbfs",
            "attack_time_s",
            "spectral_centroid_hz",
            "spectral_rolloff_hz",
            "hnr_db",
            "f0_cents_error",
            "onset_time_s",
            "active_duration_s",
        ],
        "pitch_timing_features": [
            "f0_cents_error",
            "onset_time_s",
            "onset_count",
            "active_duration_s",
        ],
    }
}


def test_metric_suite_outputs(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    synthetic_feature_table().to_csv(features, index=False)
    outputs = run_metric_suite([features], tmp_path / "metrics", CONFIG)
    alignment = pd.read_csv(outputs["alignment"])
    composition = pd.read_csv(outputs["composition"])
    monotonicity = pd.read_csv(outputs["monotonicity"])
    assert not alignment.empty
    assert alignment["effect_alignment_cosine"].mean() > 0.5
    assert not composition.empty
    assert monotonicity["monotonic"].all()
    assert outputs["summary"].exists()


def test_metric_suite_excludes_generator_only_rows_from_headlines(tmp_path: Path) -> None:
    frame = synthetic_feature_table()
    frame["analysis_tier"] = "real_counterfactual_primary"
    exploratory = frame.loc[frame["source"].eq("model")].copy()
    exploratory["clip_id"] = "extra-" + exploratory["clip_id"]
    exploratory["technique"] = "trill_major"
    exploratory["analysis_tier"] = "generator_only_exploratory"
    combined = pd.concat([frame, exploratory], ignore_index=True)
    features = tmp_path / "features.csv"
    combined.to_csv(features, index=False)
    outputs = run_metric_suite([features], tmp_path / "metrics", CONFIG)
    monotonicity = pd.read_csv(outputs["monotonicity"])
    assert "trill_major" not in set(monotonicity["technique"])


def test_calibration_mapping_is_monotonic(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    synthetic_feature_table().to_csv(features, index=False)
    outputs = fit_technique_calibration([features], tmp_path / "calibration")
    mapping = pd.read_csv(outputs["mapping"])
    assert set(mapping["technique"]) == {"sustain", "staccato", "pizzicato", "legato_slur"}
    for _, group in mapping.groupby("technique"):
        assert np.all(np.diff(group.sort_values("desired_cc1")["corrected_cc1"]) >= 0)
    assert outputs["figure"].exists()


# ---------------------------------------------------------------- 통계 도구 검증
def test_holm_adjust_matches_known_values() -> None:
    """Holm-Bonferroni 를 손으로 계산한 값과 대조한다.

    p = 0.01, 0.02, 0.03 (m=3) 이면
      0.01*3 = 0.03
      0.02*2 = 0.04
      0.03*1 = 0.03 -> 단조 강제로 0.04
    """
    from viocf.metrics import holm_adjust

    out = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.03})
    assert out["a"] == pytest.approx(0.03, abs=1e-9)
    assert out["b"] == pytest.approx(0.04, abs=1e-9)
    assert out["c"] == pytest.approx(0.04, abs=1e-9)  # 단조 강제


def test_holm_adjust_is_monotone_and_bounded() -> None:
    from viocf.metrics import holm_adjust

    out = holm_adjust({f"f{i}": p for i, p in enumerate([0.5, 0.001, 0.9, 0.04])})
    values = sorted(out.values())
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values == sorted(values)


def test_permutation_pvalue_separates_signal_from_noise() -> None:
    """진짜 효과에는 작은 p, 순수 잡음에는 큰 p 가 나와야 한다."""
    from viocf.metrics import cluster_permutation_pvalue

    rng = np.random.default_rng(0)
    prompts = [f"p{i}" for i in range(24)]
    signal = pd.DataFrame(
        {"prompt_id": np.repeat(prompts, 4),
         "excess_leakage": rng.normal(1.2, 0.3, 96)}
    )
    noise = pd.DataFrame(
        {"prompt_id": np.repeat(prompts, 4),
         "excess_leakage": rng.normal(0.0, 0.3, 96)}
    )
    assert cluster_permutation_pvalue(signal, "excess_leakage") < 0.01
    assert cluster_permutation_pvalue(noise, "excess_leakage") > 0.10


def test_hierarchical_bootstrap_is_wider_than_naive_cluster_bootstrap() -> None:
    """2단계 부트스트랩은 클러스터 내부 변동까지 반영하므로 CI 가 더 넓어야 한다.

    이게 뒤집히면 'hierarchical' 이라는 이름이 거짓이 된다.
    """
    from viocf.metrics import bootstrap_mean_ci

    rng = np.random.default_rng(1)
    prompts = [f"p{i}" for i in range(24)]
    frame = pd.DataFrame(
        {"prompt_id": np.repeat(prompts, 8),
         "value": np.repeat(rng.normal(0, 0.5, 24), 8) + rng.normal(0, 1.5, 192)}
    )
    out = bootstrap_mean_ci(frame, "value", iterations=800, seed=3)
    assert out["bootstrap"] == "hierarchical"
    assert out["n_clusters"] == 24

    # 같은 데이터로 1단계(클러스터만 재표집) CI 를 직접 계산해 비교한다.
    rng2 = np.random.default_rng(3)
    by = {k: g["value"].to_numpy() for k, g in frame.groupby("prompt_id")}
    keys = list(by)
    naive = []
    for _ in range(800):
        picks = rng2.integers(0, len(keys), size=len(keys))
        naive.append(float(np.mean(np.concatenate([by[keys[i]] for i in picks]))))
    naive_width = float(np.quantile(naive, 0.975) - np.quantile(naive, 0.025))
    assert (out["ci_high"] - out["ci_low"]) > naive_width


def test_c2st_detects_difference_and_accepts_identical() -> None:
    """C2ST 검증: 같은 분포는 0.5 근처, 다른 분포는 높은 정확도.

    이게 뒤집히면 임베딩 공간 비교 결과를 믿을 수 없다.
    """
    from viocf.metrics import classifier_two_sample_test

    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (300, 8))
    same = rng.normal(0, 1, (300, 8))
    shifted = rng.normal(1.5, 1, (300, 8))

    null = classifier_two_sample_test(a, same)
    signal = classifier_two_sample_test(a, shifted)

    assert 0.40 < null["accuracy"] < 0.62, "같은 분포인데 구별해버림"
    assert null["p_value"] > 0.05
    assert signal["accuracy"] > 0.80, "명백히 다른 분포를 못 구별함"
    assert signal["p_value"] < 1e-6


def test_c2st_handles_degenerate_input() -> None:
    from viocf.metrics import classifier_two_sample_test

    rng = np.random.default_rng(1)
    tiny = rng.normal(0, 1, (2, 4))
    out = classifier_two_sample_test(tiny, rng.normal(0, 1, (2, 4)))
    assert np.isnan(out["accuracy"])


def test_metrics_excludes_failed_renders(tmp_path: Path) -> None:
    """무음/약한 렌더가 지표에 섞이면 안 된다.

    VIOLET 은 near-silent 렌더를 seed 재시도로 감추는데, 우리는 짝 실험을 위해
    재시도를 껐다. 그래서 실패 클립이 그대로 남는다. 실패율은 별도 결과로 보고하되,
    지표 계산에는 절대 들어가면 안 된다 — 평균과 분산을 통째로 오염시킨다.
    """
    from viocf.metrics import run_metric_suite

    frame = synthetic_feature_table().copy()
    frame["render_grade"] = "ok"
    poisoned = frame.index[: len(frame) // 3]
    frame.loc[poisoned, "render_grade"] = "silent"
    for column in ("rms_dbfs", "spectral_centroid_hz"):
        if column in frame:
            frame.loc[poisoned, column] = -999.0

    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    with pytest.warns(UserWarning, match="렌더 실패"):
        paths = run_metric_suite([path], tmp_path / "out", CONFIG)

    poisoned_scales = json.loads(
        paths["summary"].read_text(encoding="utf-8")
    ).get("robust_feature_scales", {})

    # 오염 행을 아예 넣지 않은 경우와 비교한다. 제외가 제대로 됐다면 두 스케일이 같아야 한다.
    # (특징마다 단위가 달라서 절대값으로는 판정할 수 없다 — centroid 는 Hz 라 수백이 정상이다)
    clean = frame.loc[frame["render_grade"].eq("ok")].copy()
    clean_path = tmp_path / "clean.csv"
    clean.to_csv(clean_path, index=False)
    clean_scales = json.loads(
        run_metric_suite([clean_path], tmp_path / "out_clean", CONFIG)["summary"]
        .read_text(encoding="utf-8")
    ).get("robust_feature_scales", {})

    for name, value in clean_scales.items():
        assert name in poisoned_scales
        assert poisoned_scales[name] == pytest.approx(value, rel=1e-9), (
            f"{name}: 무음 행이 스케일을 오염시켰다 "
            f"({poisoned_scales[name]} != {value})"
        )


def test_metrics_falls_back_to_peak_when_grade_missing(tmp_path: Path) -> None:
    """render_grade 열이 없어도 peak 로 걸러야 한다."""
    from viocf.metrics import run_metric_suite

    frame = synthetic_feature_table().copy()
    frame["peak_dbfs"] = -12.0
    frame.loc[frame.index[:4], "peak_dbfs"] = -70.0

    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    with pytest.warns(UserWarning, match="peak"):
        run_metric_suite([path], tmp_path / "out", CONFIG)


def test_delayed_branch_does_not_count_unmeasurable_groups_as_leaks():
    """무음 탓에 값이 없는 그룹을 '누출 있음'으로 세면 안 된다.

    이 검정의 주장은 "N개 그룹 전부에서 분기 전 구간이 다르다"이다. 분모에
    측정 불가 그룹이 섞이면 그 주장이 부풀려진다. 예전 코드는 spread 가 전부
    NaN 일 때 nanmax(...) = NaN 이 되고 `NaN < 1e-6` 이 False 라 누출로 셌다.

    세 종류를 섞어 넣고 각각이 제대로 분류되는지 본다.
      causal      분기 전이 완전히 동일 -> identical
      leaky       분기 전이 다름        -> 누출
      unmeasurable 무음이라 값이 NaN    -> 판정 제외
    """
    import numpy as np

    from viocf.metrics import delayed_branch_strict_model_leak

    rows = []
    for group_name, kind in (
        ("g_causal", "causal"),
        ("g_leaky", "leaky"),
        ("g_nan", "unmeasurable"),
    ):
        for index, dynamic in enumerate(("p", "mf", "f")):
            if kind == "causal":
                rms_value, centroid = -20.0, 900.0
            elif kind == "leaky":
                rms_value, centroid = -20.0 + index * 4.0, 900.0 + index * 50.0
            else:
                rms_value, centroid = np.nan, np.nan
            rows.append({
                "profile": "delayed",
                "source": "model",
                "prompt_id": "delayed_A4",
                "technique": "pizzicato",
                "noise_group": group_name,
                "dynamic_label": dynamic,
                "prebranch_rms_dbfs": rms_value,
                "prebranch_centroid_hz": centroid,
            })
    frame = pd.DataFrame(rows)

    result = delayed_branch_strict_model_leak(frame)
    assert len(result) == 3
    by_group = result.set_index("noise_group")

    assert by_group.loc["g_causal", "measurable"]
    assert by_group.loc["g_causal", "prebranch_identical"], "동일한 그룹을 누출로 셌다"

    assert by_group.loc["g_leaky", "measurable"]
    assert not by_group.loc["g_leaky", "prebranch_identical"]

    assert not by_group.loc["g_nan", "measurable"], (
        "값이 전부 NaN 인 그룹이 측정 가능으로 분류됐다"
    )

    measured = result.loc[result["measurable"]]
    assert len(measured) == 2, "측정 불가 그룹이 분모에 남았다"
    assert int(measured["prebranch_identical"].sum()) == 1


def test_delayed_branch_metrics_uses_notated_window():
    """delayed_branch_metrics 도 악보 시각 기준 창을 써야 한다.

    strict 검정만 고치고 이쪽을 빠뜨리면 같은 요약 안에 서로 다른 정의의
    '분기 전'이 섞인다. 두 열을 모순되게 채워 넣고, 악보 시각 쪽 값이 쓰이는지
    본다(검출 기준 열에는 일부러 큰 차이를 심어 둔다 — 그쪽이 쓰이면 티가 난다).
    """
    from viocf.metrics import delayed_branch_metrics

    rows = []
    for replicate in range(4):
        for dynamic, abs_rms, detected_rms in (("p", -20.0, -20.0), ("f", -20.0, -5.0)):
            rows.append({
                "profile": "delayed",
                "source": "model",
                "prompt_id": "delayed_A4",
                "technique": "pizzicato",
                "replicate": replicate,
                "noise_group": f"g{replicate}",
                "dynamic_label": dynamic,
                # 악보 시각 기준: p 와 f 가 같다 -> future_leak 은 0 이어야 한다
                "prebranch_abs_rms_dbfs": abs_rms,
                "prebranch_abs_centroid_hz": 900.0,
                # 검출 onset 기준: 큰 차이를 심어 둔다 -> 이게 쓰이면 0 이 안 나온다
                "prebranch_rms_dbfs": detected_rms,
                "prebranch_centroid_hz": 900.0 + (0.0 if dynamic == "p" else 400.0),
                "postbranch_rms_dbfs": -20.0 if dynamic == "p" else -12.0,
                "postbranch_centroid_hz": 1000.0,
            })
    frame = pd.DataFrame(rows)

    result = delayed_branch_metrics(frame)
    assert not result.empty
    per_unit = result.loc[result["technique"].eq("pizzicato")]
    assert (per_unit["future_leak"] < 1e-9).all(), (
        "악보 시각 기준으로는 분기 전이 동일한데 future_leak 이 0 이 아니다 "
        "— 검출 onset 기준 열이 쓰이고 있다"
    )
    # 대조: 실제 효과는 잡혀야 한다. 안 잡히면 이 테스트가 아무것도 검증 못 한다.
    assert (per_unit["post_effect"] > 0).all()
