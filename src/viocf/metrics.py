from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

EPS = 1e-9


def _unit_id(frame: pd.DataFrame) -> pd.Series:
    source = frame["source"].astype(str)
    model_id = frame.get("noise_group", frame.get("replicate", pd.Series(index=frame.index))).astype(str)
    performer = frame.get("performer_id", pd.Series("P1", index=frame.index)).fillna("P1").astype(str)
    violin = frame.get("violin_id", pd.Series("V0", index=frame.index)).fillna("V0").astype(str)
    replicate = frame.get("replicate", frame.get("take", pd.Series(1, index=frame.index))).astype(str)
    real_id = performer + ":" + violin + ":" + replicate
    return pd.Series(np.where(source.eq("model"), model_id, real_id), index=frame.index)


_CONDITION_KEYS = ["prompt_id", "technique", "dynamic_label"]


def _real_robust_scales(frame: pd.DataFrame, features: Sequence[str]) -> dict[str, float]:
    """1 z 단위 = **같은 조건을 다시 연주했을 때 실제로 흔들리는 폭**.

    ⚠ 조건 전체(p~f, 전 주법)를 한 덩어리로 뭉쳐 IQR 을 잡으면 그건 '재현성'이 아니라
    '조건 간 전체 폭'이다. 그 값은 훨씬 커서, 모델의 이탈이 실제보다 작아 보이게 만든다.
    이 연구는 "사람이 다시 켰을 때의 흔들림"을 기준으로 삼는다고 선언했으므로
    (docs/EXPERIMENT_SPEC.md) 분모도 그것이어야 한다.

    조건별 IQR 을 구해 중앙값으로 모은다. 반복이 부족해 조건 내 산포를 못 구하면
    전체 IQR 로 후퇴하되, 그 특징은 scales 와 함께 기록돼 감사 가능해야 한다.
    """
    real = frame.loc[frame["source"].eq("real")]
    keys = [key for key in _CONDITION_KEYS if key in real.columns]
    scales: dict[str, float] = {}
    for feature in features:
        scale = math.nan
        if keys and not real.empty:
            within: list[float] = []
            for _, group in real.groupby(keys, dropna=False):
                values = pd.to_numeric(group.get(feature), errors="coerce").dropna().to_numpy(dtype=float)
                if values.size >= 2:
                    q25, q75 = np.quantile(values, [0.25, 0.75])
                    spread = float(q75 - q25)
                    if spread <= EPS:
                        spread = float(np.std(values, ddof=1))
                    if spread > EPS:
                        within.append(spread)
            if within:
                scale = float(np.median(within))
        if not np.isfinite(scale) or scale <= EPS:
            # 후퇴: 조건 내 반복이 없을 때만. 의미가 다르므로 위 값과 섞어 해석하지 말 것.
            values = pd.to_numeric(real.get(feature), errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                values = pd.to_numeric(frame.get(feature), errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                scales[feature] = 1.0
                continue
            q25, q75 = np.quantile(values, [0.25, 0.75])
            scale = float(q75 - q25)
            if scale <= EPS:
                scale = float(np.std(values))
        scales[feature] = scale if scale > EPS else 1.0
    return scales


def standardize_for_response(
    frame: pd.DataFrame, features: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    available = [feature for feature in features if feature in frame.columns]
    if not available:
        raise ValueError("None of the configured response features exist in the feature table")
    scales = _real_robust_scales(frame, available)
    output = frame.copy()
    for feature in available:
        output[feature] = pd.to_numeric(output[feature], errors="coerce") / scales[feature]
    output["unit_id"] = _unit_id(output)
    return output, scales


def build_contrasts(
    frame: pd.DataFrame,
    features: Sequence[str],
    baseline_technique: str = "sustain",
    baseline_dynamic: str = "mf",
) -> pd.DataFrame:
    """Create paired technique and dynamics deltas within prompt/replicate blocks."""
    records: list[dict[str, Any]] = []
    keys = ["source", "prompt_id", "unit_id"]
    for key, group in frame.loc[frame["profile"].eq("constant")].groupby(keys, dropna=False):
        source, prompt_id, unit_id = key
        cells = group.groupby(["technique", "dynamic_label"], dropna=False)[list(features)].mean()
        baseline_key = (baseline_technique, baseline_dynamic)
        if baseline_key not in cells.index:
            continue
        baseline = cells.loc[baseline_key]
        techniques = sorted(set(group["technique"].dropna().astype(str)))
        dynamics = sorted(set(group["dynamic_label"].dropna().astype(str)))

        for technique in techniques:
            target_key = (technique, baseline_dynamic)
            if technique == baseline_technique or target_key not in cells.index:
                continue
            delta = cells.loc[target_key] - baseline
            record: dict[str, Any] = {
                "source": source,
                "prompt_id": prompt_id,
                "unit_id": unit_id,
                "control_type": "technique",
                "technique": technique,
                "dynamic_label": baseline_dynamic,
                "contrast": f"technique:{baseline_technique}->{technique}@{baseline_dynamic}",
            }
            record.update({feature: delta[feature] for feature in features})
            records.append(record)

        for technique in techniques:
            base_key = (technique, baseline_dynamic)
            if base_key not in cells.index:
                continue
            technique_baseline = cells.loc[base_key]
            for dynamic in dynamics:
                target_key = (technique, dynamic)
                if dynamic == baseline_dynamic or target_key not in cells.index:
                    continue
                delta = cells.loc[target_key] - technique_baseline
                record = {
                    "source": source,
                    "prompt_id": prompt_id,
                    "unit_id": unit_id,
                    "control_type": "dynamics",
                    "technique": technique,
                    "dynamic_label": dynamic,
                    "contrast": f"dynamics:{baseline_dynamic}->{dynamic}@{technique}",
                }
                record.update({feature: delta[feature] for feature in features})
                records.append(record)
    return pd.DataFrame(records)


def _cosine_and_magnitude(model: np.ndarray, real: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(model) & np.isfinite(real)
    if np.count_nonzero(valid) == 0:
        return math.nan, math.nan
    model = model[valid]
    real = real[valid]
    denominator = np.linalg.norm(model) * np.linalg.norm(real)
    cosine = float(np.dot(model, real) / denominator) if denominator > EPS else math.nan
    magnitude_ratio = float(np.dot(model, real) / (np.dot(real, real) + EPS))
    return cosine, magnitude_ratio


def effect_alignment(contrasts: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouping = ["prompt_id", "control_type", "technique", "dynamic_label", "contrast"]
    for key, group in contrasts.groupby(grouping, dropna=False):
        means = group.groupby("source")[list(features)].mean()
        if "model" not in means.index or "real" not in means.index:
            continue
        cosine, ratio = _cosine_and_magnitude(
            means.loc["model"].to_numpy(dtype=float), means.loc["real"].to_numpy(dtype=float)
        )
        records.append(
            dict(zip(grouping, key))
            | {
                "effect_alignment_cosine": cosine,
                "magnitude_ratio": ratio,
                "model_replicates": int(group["source"].eq("model").sum()),
                "real_replicates": int(group["source"].eq("real").sum()),
            }
        )
    return pd.DataFrame(records)


def excess_leakage(
    contrasts: pd.DataFrame,
    dynamics_non_target: Sequence[str],
    technique_non_target: Sequence[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouping = ["control_type", "technique", "dynamic_label", "contrast"]
    for key, group in contrasts.groupby(grouping, dropna=False):
        control_type = str(key[0])
        requested = dynamics_non_target if control_type == "dynamics" else technique_non_target
        features = [feature for feature in requested if feature in group.columns]
        real = group.loc[group["source"].eq("real")]
        model = group.loc[group["source"].eq("model")]
        if real.empty or model.empty or not features:
            continue
        thresholds = {
            feature: float(np.nanquantile(np.abs(pd.to_numeric(real[feature], errors="coerce")), 0.95))
            for feature in features
            if pd.to_numeric(real[feature], errors="coerce").notna().any()
        }
        for _, row in model.iterrows():
            components = []
            component_map: dict[str, float] = {}
            for feature, threshold in thresholds.items():
                value = abs(float(row[feature])) if pd.notna(row[feature]) else math.nan
                excess = max(0.0, value - threshold) if np.isfinite(value) else math.nan
                component_map[f"excess_{feature}"] = excess
                if np.isfinite(excess):
                    components.append(excess)
            score = float(np.linalg.norm(components)) if components else math.nan
            record = dict(zip(grouping, key)) | {
                "prompt_id": row["prompt_id"],
                "unit_id": row["unit_id"],
                "excess_leakage": score,
            }
            record.update(component_map)
            records.append(record)
    return pd.DataFrame(records)


def build_interactions(
    frame: pd.DataFrame,
    features: Sequence[str],
    baseline_technique: str = "sustain",
    baseline_dynamic: str = "mf",
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["source", "prompt_id", "unit_id"]
    for key, group in frame.loc[frame["profile"].eq("constant")].groupby(keys, dropna=False):
        source, prompt_id, unit_id = key
        cells = group.groupby(["technique", "dynamic_label"], dropna=False)[list(features)].mean()
        base = (baseline_technique, baseline_dynamic)
        if base not in cells.index:
            continue
        techniques = sorted(set(group["technique"].dropna().astype(str)) - {baseline_technique})
        dynamics = sorted(set(group["dynamic_label"].dropna().astype(str)) - {baseline_dynamic})
        for technique in techniques:
            for dynamic in dynamics:
                required = [
                    (technique, dynamic),
                    (technique, baseline_dynamic),
                    (baseline_technique, dynamic),
                    base,
                ]
                if not all(cell in cells.index for cell in required):
                    continue
                interaction = (
                    cells.loc[(technique, dynamic)]
                    - cells.loc[(technique, baseline_dynamic)]
                    - cells.loc[(baseline_technique, dynamic)]
                    + cells.loc[base]
                )
                record: dict[str, Any] = {
                    "source": source,
                    "prompt_id": prompt_id,
                    "unit_id": unit_id,
                    "technique": technique,
                    "dynamic_label": dynamic,
                    "interaction": f"{technique}x{dynamic}",
                }
                record.update({feature: interaction[feature] for feature in features})
                records.append(record)
    return pd.DataFrame(records)


def multivariate_energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not len(x) or not len(y):
        return math.nan
    return float(
        2.0 * np.mean(cdist(x, y)) - np.mean(cdist(x, x)) - np.mean(cdist(y, y))
    )


def classifier_two_sample_test(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 20260826,
    folds: int = 5,
) -> dict[str, float]:
    """C2ST — 분류기 이표본 검정. 고차원 임베딩에는 energy distance 보다 이쪽이 맞다.

    "두 분포가 다른가?" 를 "분류기가 둘을 구별할 수 있는가?" 로 바꾼다.
    교차검증 정확도가 곧 divergence 다.

      0.5  = 구별 불가 = 두 분포가 같다
      1.0  = 완벽히 구별 = 완전히 다르다

    energy distance 는 단위가 임의라 '얼마나 다른지'를 말하기 어렵지만, C2ST 는
    **정확도**라 해석이 바로 된다("모델 출력과 실연주를 78% 로 구별할 수 있다").
    p 값도 이항검정으로 바로 나온다.

    MERT 같은 768~1024 차원 임베딩에서는 거리 기반 통계량이 차원의 저주로 둔해지는데,
    C2ST 는 분류기가 판별 방향을 스스로 찾으므로 그 영향을 덜 받는다.
    """
    from scipy.stats import binomtest
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        return {"accuracy": math.nan, "p_value": math.nan, "n": 0}
    if len(x) < folds or len(y) < folds:
        return {"accuracy": math.nan, "p_value": math.nan, "n": len(x) + len(y)}

    data = np.vstack([x, y])
    labels = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
    ok = np.isfinite(data).all(axis=1)
    data, labels = data[ok], labels[ok]
    if len(np.unique(labels)) < 2:
        return {"accuracy": math.nan, "p_value": math.nan, "n": len(labels)}

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    predicted = cross_val_predict(
        HistGradientBoostingClassifier(random_state=seed), data, labels, cv=splitter
    )
    correct = int(np.sum(predicted == labels))
    total = len(labels)
    # 귀무가설: 분류기가 동전던지기와 같다(정확도 0.5)
    p = float(binomtest(correct, total, 0.5, alternative="greater").pvalue)
    return {"accuracy": correct / total, "p_value": p, "n": total}


def _split_half_floor(real: np.ndarray, seed: int, repeats: int = 25) -> float:
    """실연주를 무작위로 반 갈라 잰 energy distance 의 중앙값 = 유한표본 바닥."""
    if len(real) < 4:
        return 0.0
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        order = rng.permutation(len(real))
        midpoint = len(real) // 2
        left, right = real[order[:midpoint]], real[order[midpoint:]]
        distance = multivariate_energy_distance(left, right)
        if np.isfinite(distance):
            values.append(distance)
    return float(np.median(values)) if values else 0.0


# QC 를 못 돌렸을 때만 쓰는 대용 경계. 전수 분포에 골짜기가 없으므로 이 값은
# 원리에서 나온 게 아니라 '아무것도 없는 것보다 낫다' 수준이다. QC 의 활성 구간
# 검출을 쓸 수 있으면 언제나 그쪽이 옳다.
PEAK_FALLBACK_DBFS = -35.0


def _stable_key_seed(key: object) -> int:
    """Return a process-independent seed (Python's hash() is randomized)."""
    digest = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def compositionality_gap(interactions: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for key, group in interactions.groupby(["technique", "dynamic_label", "interaction"], dropna=False):
        model = group.loc[group["source"].eq("model"), list(features)].replace([np.inf, -np.inf], np.nan)
        real = group.loc[group["source"].eq("real"), list(features)].replace([np.inf, -np.inf], np.nan)
        usable = [
            feature
            for feature in features
            if model[feature].notna().any() and real[feature].notna().any()
        ]
        if not usable:
            continue
        x = model[usable].dropna().to_numpy(dtype=float)
        y = real[usable].dropna().to_numpy(dtype=float)
        if not len(x) or not len(y):
            continue
        distance = multivariate_energy_distance(x, y)
        # 유한표본 noise floor: 완벽한 모델이라도 표본이 유한하면 거리가 0이 아니다.
        # ⚠ 행 순서를 그대로 반으로 가르면 (예: 앞=V1/V2, 뒤=V3) floor 에 악기 차이가
        # 섞여 들어가 gap 을 과소평가한다. 무작위 분할을 여러 번 해서 중앙값을 쓴다.
        human_floor = _split_half_floor(y, seed=_stable_key_seed(key))
        records.append(
            {
                "technique": key[0],
                "dynamic_label": key[1],
                "interaction": key[2],
                "features_used": ";".join(usable),
                "energy_distance_model_real": distance,
                "human_split_half_floor": human_floor,
                "human_calibrated_gap": distance - human_floor,
                "model_n": len(x),
                "real_n": len(y),
            }
        )
    return pd.DataFrame(records)


def monotonicity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    constant = frame.loc[frame["profile"].eq("constant")].copy()
    if "unit_id" not in constant.columns:
        constant["unit_id"] = _unit_id(constant)
    records: list[dict[str, Any]] = []
    for key, group in constant.groupby(["source", "prompt_id", "technique", "unit_id"], dropna=False):
        cells = group.groupby("cc1_final")["rms_dbfs"].mean().sort_index()
        if len(cells) < 3:
            continue
        differences = np.diff(cells.to_numpy(dtype=float))
        records.append(
            {
                "source": key[0],
                "prompt_id": key[1],
                "technique": key[2],
                "unit_id": key[3],
                "monotonic": bool(np.all(differences > 0)),
                "adjacent_violation_rate": float(np.mean(differences <= 0)),
                "p_to_f_gain_db": float(cells.iloc[-1] - cells.iloc[0]),
                "minimum_adjacent_gain_db": float(np.min(differences)),
            }
        )
    return pd.DataFrame(records)


def delayed_branch_strict_model_leak(frame: pd.DataFrame) -> pd.DataFrame:
    """분기 이전 구간의 **모델 단독** 검정. 실연주를 기준선으로 쓰지 않는다.

    왜 따로 두는가: 사람은 "250 ms 뒤부터 f" 를 연주할 수 없다. 목표 세기를 처음부터
    알고 활을 놓기 때문에 실연주는 분기 전부터 이미 다르다. 그 값을 baseline 으로 쓰면
    null 이 부풀어 모델 검정이 약해진다.

    모델 쪽은 조건이 훨씬 강하다. 같은 noise group 안에서 초기 latent 가 같고 분기 전
    CC1 조건이 **완전히 동일**하다(VIOLET 은 CC1 을 보간하지 않고 sample-and-hold 로
    프레임에 깐다 - src/data/components/midi_processor.py 참조). 따라서 인과적 생성기라면
    분기 전 오디오가 조건에 관계없이 **정확히 같아야** 한다. 0 이 아니면 미래 조건이
    과거로 샌 것이다.

    ⚠ 해석 주의: VIOLET 은 full-window diffusion 이라 **구조상 비인과적**이다. 누출이
    관측되는 것 자체는 결함이 아니라 설계의 귀결이다. 결론은 "물리 법칙 위반"이 아니라
    "이 모델은 실시간/대화형 제어에 쓸 수 없다" 로 쓴다.
    """
    delayed = frame.loc[frame["profile"].eq("delayed") & frame["source"].eq("model")].copy()
    if delayed.empty:
        return pd.DataFrame()
    # ⚠ 악보 시각 기준(prebranch_abs_*)을 우선한다.
    # 검출 onset 기준 창은 onset 검출이 흔들리면 창이 함께 움직여서, 차이가
    # '다른 소리'인지 '다른 구간'인지 구분되지 않는다. 악보 시각은 manifest 값이라
    # 한 noise_group 안에서 정확히 같으므로 그 모호함이 없다.
    absolute = [
        name for name in ("prebranch_abs_rms_dbfs", "prebranch_abs_centroid_hz")
        if name in delayed
    ]
    features = absolute or [
        name for name in ("prebranch_rms_dbfs", "prebranch_centroid_hz") if name in delayed
    ]
    if not features:
        return pd.DataFrame()
    if "noise_group" not in delayed.columns:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for key, group in delayed.groupby(["prompt_id", "technique", "noise_group"], dropna=False):
        cells = group.groupby("dynamic_label")[features].mean()
        if len(cells) < 2:
            continue
        row: dict[str, Any] = {
            "prompt_id": key[0], "technique": key[1], "noise_group": key[2],
            "n_dynamics": len(cells),
        }
        for feature in features:
            values = pd.to_numeric(cells[feature], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"spread_{feature}"] = float(np.ptp(values)) if values.size >= 2 else math.nan

        # ⚠ 무음 클립은 분기 전 특징이 NaN 이다. 강약 라벨이 2개 있어도 **값이**
        # 2개가 아니면 spread 를 낼 수 없다.
        #
        # 예전 코드는 그 경우 nanmax([nan, nan]) = nan 이 되고 nan < 1e-6 이 False 라
        # **자료가 없는 그룹을 '누출 있음'으로 세었다.** 자료 없음과 누출은 다르다.
        # 이 검정의 주장이 '46/46 그룹에서 누출'인 만큼 분모가 오염되면 안 된다.
        spreads = [
            abs(row[f"spread_{feature}"])
            for feature in features
            if np.isfinite(row.get(f"spread_{feature}", math.nan))
        ]
        row["measurable"] = bool(spreads)
        row["prebranch_identical"] = bool(max(spreads) < 1e-6) if spreads else False
        records.append(row)
    return pd.DataFrame(records)


def delayed_branch_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    delayed = frame.loc[frame["profile"].eq("delayed")].copy()
    if delayed.empty:
        return pd.DataFrame()
    # ⚠ 여기도 악보 시각 기준을 우선한다. strict 검정만 고치고 이쪽을 빠뜨리면
    # 같은 표에 서로 다른 정의의 '분기 전'이 섞인다(실제로 그럴 뻔했다).
    # 검출 onset 기준 창은 onset 검출이 흔들리면 함께 움직여 가짜 차이를 만든다 —
    # 실측 6.73 dB, 개정 21 참조.
    prefix_absolute = [
        name for name in ("prebranch_abs_rms_dbfs", "prebranch_abs_centroid_hz")
        if name in delayed
    ]
    prefix_features = prefix_absolute or [
        name for name in ("prebranch_rms_dbfs", "prebranch_centroid_hz") if name in delayed
    ]
    post_features = [name for name in ("postbranch_rms_dbfs", "postbranch_centroid_hz") if name in delayed]
    requested = prefix_features + post_features
    scales = _real_robust_scales(delayed, requested)
    for feature in requested:
        delayed[feature] = pd.to_numeric(delayed[feature], errors="coerce") / scales[feature]
    delayed["unit_id"] = _unit_id(delayed)

    records: list[dict[str, Any]] = []
    for key, group in delayed.groupby(["source", "prompt_id", "technique", "unit_id"], dropna=False):
        cells = group.groupby("dynamic_label")[requested].mean()
        if "p" not in cells.index or "f" not in cells.index:
            continue
        prefix_delta = cells.loc["f", prefix_features] - cells.loc["p", prefix_features]
        post_delta = cells.loc["f", post_features] - cells.loc["p", post_features]
        records.append(
            {
                "source": key[0],
                "prompt_id": key[1],
                "technique": key[2],
                "unit_id": key[3],
                "future_leak": float(np.linalg.norm(prefix_delta.dropna().to_numpy(dtype=float))),
                "post_effect": float(np.linalg.norm(post_delta.dropna().to_numpy(dtype=float))),
                "post_rms_effect_scaled": (
                    float(post_delta.get("postbranch_rms_dbfs", math.nan))
                    if len(post_features)
                    else math.nan
                ),
            }
        )
    result = pd.DataFrame(records)
    if result.empty:
        return result
    means = result.groupby(["source", "technique"])[["future_leak", "post_effect"]].mean()
    gap_rows = []
    for source in sorted(result["source"].unique()):
        if (source, "sustain") in means.index and (source, "pizzicato") in means.index:
            gap_rows.append(
                {
                    "source": source,
                    "prompt_id": "delayed_A4",
                    "technique": "sustain_minus_pizzicato",
                    "unit_id": "aggregate",
                    # future_leak 차이도 낸다. 예전엔 이유 없이 NaN 이었는데,
                    # 이 대비야말로 핵심이다 — 피치카토는 줄을 놓은 뒤라 정당한
                    # 미래 효과가 없으므로 누출이 더 두드러져야 한다.
                    "future_leak": float(
                        means.loc[(source, "sustain"), "future_leak"]
                        - means.loc[(source, "pizzicato"), "future_leak"]
                    ),
                    "post_effect": float(
                        means.loc[(source, "sustain"), "post_effect"]
                        - means.loc[(source, "pizzicato"), "post_effect"]
                    ),
                    "post_rms_effect_scaled": math.nan,
                }
            )
    return pd.concat([result, pd.DataFrame(gap_rows)], ignore_index=True)


def bootstrap_mean_ci(
    frame: pd.DataFrame,
    value: str,
    cluster: str = "prompt_id",
    iterations: int = 2000,
    seed: int = 20260826,
) -> dict[str, float]:
    clean = frame[[cluster, value]].dropna()
    clusters = clean[cluster].drop_duplicates().to_numpy()
    if len(clusters) == 0:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    if len(clusters) < 20:
        # 일반화 단위를 prompt 로 잡았으므로 CI 는 prompt 개수에만 의존한다.
        # 12개(full)면 percentile 부트스트랩 CI 가 상당히 불안정하고, 2개(pilot)면 의미가 없다.
        warnings.warn(
            f"bootstrap cluster '{cluster}' n={len(clusters)} (<20): "
            "신뢰구간이 불안정하다. 파일럿 수치는 확증용으로 쓰지 말 것.",
            stacklevel=2,
        )
    # ---- 2단계(hierarchical) 부트스트랩 --------------------------------------
    # 사전고정 명세가 "prompt-level hierarchical bootstrap" 을 약속했다.
    # 클러스터만 재표집하는 1단계 부트스트랩은 클러스터 **내부**의 표집 변동을 무시해서
    # 신뢰구간을 좁게 낸다. 여기서는 (1) prompt 를 복원추출하고, (2) 뽑힌 prompt 안에서
    # 관측치를 다시 복원추출한다. 이게 명세가 말한 hierarchical 이다.
    rng = np.random.default_rng(seed)
    by_cluster = {
        key: group[value].to_numpy(dtype=float)
        for key, group in clean.groupby(cluster, sort=True)
    }
    order = list(by_cluster)
    index = {name: position for position, name in enumerate(order)}
    arrays = [by_cluster[name] for name in order]

    estimates = []
    for _ in range(iterations):
        picks = rng.integers(0, len(order), size=len(order))
        drawn = []
        for position in picks:
            values = arrays[position]
            if values.size:
                drawn.append(values[rng.integers(0, values.size, size=values.size)])
        if not drawn:
            continue
        estimates.append(float(np.mean(np.concatenate(drawn))))
    if not estimates:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan,
                "n_clusters": len(order), "bootstrap": "hierarchical"}
    del index
    return {
        "mean": float(clean[value].mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "n_clusters": len(order),
        "bootstrap": "hierarchical",
    }


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni 다중비교 보정. 사전고정 명세가 세부 비교에 요구한다.

    headline 3개(CEA/HCEL/CG)는 confirmatory 라 보정하지 않는다. 그 아래의
    특징별·주법별 비교는 개수가 수십 개라 보정 없이는 우연한 '누출 발견'이 쏟아진다.

    반환값은 조정된 p 값이며 원래 순서를 유지한 dict 다.
    """
    items = [(name, float(value)) for name, value in pvalues.items() if np.isfinite(value)]
    if not items:
        return {name: math.nan for name in pvalues}
    items.sort(key=lambda pair: pair[1])
    total = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, raw) in enumerate(items):
        candidate = min(1.0, (total - rank) * raw)
        running = max(running, candidate)  # Holm 은 단조 증가하도록 강제한다
        adjusted[name] = running
    for name in pvalues:
        adjusted.setdefault(name, math.nan)
    return {name: adjusted[name] for name in pvalues}


def cluster_permutation_pvalue(
    frame: pd.DataFrame,
    value: str,
    cluster: str = "prompt_id",
    null_value: float = 0.0,
    iterations: int = 2000,
    seed: int = 20260826,
) -> float:
    """클러스터 단위 부호뒤집기 순열검정으로 p 값을 낸다.

    holm_adjust 에 넣을 p 값이 있어야 명세의 보정 조항이 실체를 갖는다.
    prompt 단위로 부호를 뒤집으므로 prompt 내부 상관을 깨지 않는다.
    """
    clean = frame[[cluster, value]].dropna()
    if clean.empty:
        return math.nan
    means = clean.groupby(cluster)[value].mean().to_numpy(dtype=float) - null_value
    if means.size < 2:
        return math.nan
    observed = abs(float(np.mean(means)))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(iterations, means.size))
    null = np.abs((signs * means).mean(axis=1))
    return float((1.0 + np.sum(null >= observed)) / (1.0 + iterations))


def run_metric_suite(
    feature_paths: Iterable[str | Path],
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Path]:
    frames = [pd.read_csv(path) for path in feature_paths]
    if not frames:
        raise ValueError("At least one feature CSV is required")
    data = pd.concat(frames, ignore_index=True, sort=False)
    data = data.loc[data.get("feature_error", pd.Series(index=data.index, dtype=object)).isna()]
    # ⚠ 렌더가 실패한 클립(무음/비정상적으로 약함)은 특징값이 무의미하다.
    # 그대로 두면 평균과 분산을 오염시켜 CEA/HCEL/CG 가 전부 틀어진다.
    #
    # 이 실패는 버그가 아니라 관측 대상이다 — VIOLET 원본은 near-silent 렌더를
    # 다른 seed 로 재시도해 감추지만, 우리는 짝 실험을 지키려고 재시도를 껐다.
    # 그래서 실패율 자체가 결과가 된다(scripts/analyze_render_failures.py).
    # 다만 **지표 계산에서는 빼야 한다.** 실패율은 따로 보고한다.
    #
    # ⚠ 판정 기준의 우선순위가 중요하다. 전수 QC(18,432 클립)가 보여준 바로는
    # peak_dbfs 분포에 골짜기가 없다 — -65 부근의 얕은 최소를 지나 -30 까지
    # 단조 증가하는 연속 꼬리다. 표본 384개로 "골짜기"라고 본 것은 틀렸다.
    # 게다가 검출기가 무음이라 한 클립과 소리가 있다고 한 클립의 peak 범위가
    # 겹쳐서, **어떤 단일 peak 경계로도** 검출기 판정을 재현할 수 없다.
    #
    # 그래서 활성 구간 검출기(qc_reasons 의 near_silence)를 1순위로 쓴다.
    # 그쪽은 조정 상수가 아니라 절대 기준이다 — 어떤 프레임 RMS 도
    # -60 dBFS 를 못 넘으면 소리가 없는 것이다(audio.detect_active_region).
    # peak 경계는 QC 를 아직 안 돌린 경우의 마지막 수단으로만 남긴다.
    dropped_by = None
    if "silent_absolute" in data.columns:
        # 1순위. 특징 추출이 직접 낸 절대 기준이다(features.SILENCE_FLOOR_DBFS).
        # 앞구간을 보지 않으므로 검출기의 노이즈 추정 오염에 영향받지 않는다.
        before = len(data)
        data = data.loc[~data["silent_absolute"].astype(bool)].copy()
        dropped_by = ("절대 기준 (프레임 RMS 최대 < -60 dBFS)", before - len(data), before)
    elif "qc_reasons" in data.columns:
        reasons = data["qc_reasons"].fillna("")
        broken = reasons.str.contains(
            "near_silence|missing_audio|non_finite_samples", regex=True
        )
        before = len(data)
        data = data.loc[~broken].copy()
        dropped_by = ("활성 구간 검출기", before - len(data), before)
    elif "render_grade" in data.columns:
        before = len(data)
        data = data.loc[data["render_grade"].eq("ok")].copy()
        dropped_by = ("render_grade", before - len(data), before)
    elif "peak_dbfs" in data.columns:
        before = len(data)
        data = data.loc[
            pd.to_numeric(data["peak_dbfs"], errors="coerce") >= PEAK_FALLBACK_DBFS
        ].copy()
        dropped_by = (
            f"peak < {PEAK_FALLBACK_DBFS:.0f} dBFS (대용 기준 — QC 를 돌리는 편이 낫다)",
            before - len(data),
            before,
        )

    if dropped_by and dropped_by[1]:
        criterion, dropped, before = dropped_by
        warnings.warn(
            f"렌더 실패 클립 {dropped}개를 지표에서 제외했다 "
            f"({dropped / before * 100:.1f}%, 기준: {criterion}). "
            "실패율은 별도로 보고할 것 — 이건 결측이 아니라 관측 결과다.",
            stacklevel=2,
        )

    # ⚠ 지연 분기 검정은 **모델 단독**이라 tier 필터 앞의 자료를 써야 한다.
    #
    # analysis_tier 필터는 실연주 짝이 없는 행이 CEA/HCEL/CG 에 몰래 섞이는 것을
    # 막으려고 있다. 그런데 delayed_branch_* 는 설계상 실연주 기준선을 쓰지 않는다
    # (사람은 "250 ms 뒤부터 f" 를 연주할 수 없다 — delayed_branch_strict 주석 참조).
    # 거기까지 필터를 걸면 generator_only_exploratory 로 표시된 확장 스윕
    # 8,640 클립이 통째로 사라진다(실제로 겪음 — 요약에 지연 블록이 아예 없었다).
    model_scope = data.copy()

    if "analysis_tier" in data.columns:
        # Real-instrument comparisons are only identified for conditions with
        # a matched real counterfactual. Generator-only rows remain available
        # to the sweep surrogate, but cannot silently enter headline metrics.
        data = data.loc[data["analysis_tier"].eq("real_counterfactual_primary")].copy()
    requested = [str(value) for value in config["analysis"]["response_features"]]
    features = [name for name in requested if name in data.columns]
    standardized, scales = standardize_for_response(data, features)
    baseline_technique = str(config["analysis"]["baseline_technique"])
    baseline_dynamic = str(config["analysis"]["baseline_dynamic"])
    contrasts = build_contrasts(
        standardized, features, baseline_technique, baseline_dynamic
    )
    alignment = effect_alignment(contrasts, features) if not contrasts.empty else pd.DataFrame()
    pitch_timing = [str(value) for value in config["analysis"]["pitch_timing_features"]]
    leakage = (
        excess_leakage(
            contrasts,
            dynamics_non_target=pitch_timing,
            technique_non_target=[name for name in pitch_timing if name != "active_duration_s"],
        )
        if not contrasts.empty
        else pd.DataFrame()
    )
    interactions = build_interactions(
        standardized, features, baseline_technique, baseline_dynamic
    )
    composition = (
        compositionality_gap(interactions, features) if not interactions.empty else pd.DataFrame()
    )
    monotonicity = monotonicity_summary(data)
    delayed = delayed_branch_metrics(model_scope)
    delayed_strict = delayed_branch_strict_model_leak(model_scope)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "contrasts": output / "contrasts.csv",
        "alignment": output / "effect_alignment.csv",
        "leakage": output / "excess_leakage.csv",
        "interactions": output / "interactions.csv",
        "composition": output / "compositionality_gap.csv",
        "monotonicity": output / "monotonicity.csv",
        "delayed": output / "delayed_branch.csv",
        "delayed_strict": output / "delayed_branch_model_only.csv",
        "summary": output / "metrics_summary.json",
    }
    for key, frame in (
        ("contrasts", contrasts),
        ("alignment", alignment),
        ("leakage", leakage),
        ("interactions", interactions),
        ("composition", composition),
        ("monotonicity", monotonicity),
        ("delayed", delayed),
        ("delayed_strict", delayed_strict),
    ):
        frame.to_csv(paths[key], index=False)

    iterations = int(config["analysis"]["bootstrap_iterations"])
    random_seed = int(config["analysis"]["random_seed"])
    headline: dict[str, Any] = {"robust_feature_scales": scales}
    if not alignment.empty:
        headline["effect_alignment"] = bootstrap_mean_ci(
            alignment, "effect_alignment_cosine", iterations=iterations, seed=random_seed
        )
    if not leakage.empty:
        headline["excess_leakage"] = bootstrap_mean_ci(
            leakage, "excess_leakage", iterations=iterations, seed=random_seed + 1
        )
    if not composition.empty:
        headline["compositionality_gap_mean"] = float(
            composition["human_calibrated_gap"].mean()
        )
    if not monotonicity.empty:
        headline["monotonic_violation_rate"] = (
            monotonicity.groupby("source")["adjacent_violation_rate"].mean().to_dict()
        )
    if not delayed.empty:
        headline["delayed_branch_means"] = (
            delayed.groupby(["source", "technique"])[["future_leak", "post_effect"]]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )
    # ---- 세부 비교의 Holm 보정 -------------------------------------------------
    # headline 3개는 confirmatory 라 보정하지 않는다. 아래 특징별 누출 비교는
    # 개수가 많아 보정 없이는 우연한 '발견'이 쏟아진다(명세 통계 단위 절).
    if not leakage.empty and "feature" in leakage.columns:
        raw_p: dict[str, float] = {}
        for feature, group in leakage.groupby("feature"):
            if "prompt_id" not in group.columns:
                continue
            raw_p[str(feature)] = cluster_permutation_pvalue(
                group, "excess_leakage", cluster="prompt_id",
                iterations=iterations, seed=random_seed + 7,
            )
        if raw_p:
            adjusted = holm_adjust(raw_p)
            headline["leakage_per_feature_tests"] = {
                name: {
                    "p_raw": raw_p[name],
                    "p_holm": adjusted[name],
                    "significant_after_holm": bool(
                        np.isfinite(adjusted[name]) and adjusted[name] < 0.05
                    ),
                }
                for name in sorted(raw_p)
            }
            headline["multiple_comparison_correction"] = "holm-bonferroni"
            headline["confirmatory_endpoints"] = [
                "effect_alignment", "excess_leakage", "compositionality_gap_mean",
            ]

    if not delayed_strict.empty:
        spread_cols = [c for c in delayed_strict.columns if c.startswith("spread_")]
        # 측정 가능한 그룹만 분모에 넣는다. 무음 탓에 값이 없는 그룹을 '누출'로
        # 세면 결론이 부풀려진다.
        measured = (
            delayed_strict.loc[delayed_strict["measurable"]]
            if "measurable" in delayed_strict.columns
            else delayed_strict
        )
        headline["delayed_model_only_prebranch"] = {
            "noise_groups_measured": len(measured),
            "noise_groups_unmeasurable": int(len(delayed_strict) - len(measured)),
            "identical_groups": int(measured["prebranch_identical"].sum()),
            "max_spread": {
                c: (float(measured[c].max()) if measured[c].notna().any() else None)
                for c in spread_cols
            },
            "median_spread": {
                c: (float(measured[c].median()) if measured[c].notna().any() else None)
                for c in spread_cols
            },
            "note": (
                "인과적 생성기라면 분기 전 spread 가 0 이어야 한다. "
                "unmeasurable 은 무음 탓에 값이 없어 판정에서 제외한 그룹이다."
            ),
        }
    paths["summary"].write_text(
        json.dumps(headline, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return paths
