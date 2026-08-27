"""대리모델(surrogate) 실사 — ExtraTrees 가 정말 뒤처지는가.

대리모델의 일: (prompt, 주법, CC1, guidance, 스텝수) -> VIOLET 의 반응 특징 예측.
그래야 생성기를 매번 안 돌리고도 '제어가 깨질 것 같은 영역'을 찾을 수 있다.

과제 구조를 실제와 똑같이 만든다:
  - 혼합형 예측변수(범주형 4 + 수치형 4)
  - **prompt 단위 GroupKFold** — 처음 보는 프롬프트로 일반화해야 의미가 있다
  - 비선형 + 상호작용 + 포화(플래토) — 실제 제어 응답의 특징
  - 이질적 잡음

공정성: 전부 기본 하이퍼파라미터. 튜닝하면 튜닝한 쪽이 이기는 게 당연하므로
'상자에서 꺼내 바로 쓸 때' 기준으로 겨룬다.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

CATEGORICAL = ["technique", "pattern", "register", "timing_variant"]
NUMERIC = ["cc1_final", "w_tech", "w_cc", "sampling_steps"]


def make_data(n_prompts: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    techniques = ["sustain", "staccato", "pizzicato", "legato_slur"]
    patterns = ["long", "scale", "repeat", "leap"]
    registers = ["low", "mid", "high"]
    variants = ["short", "long"]
    rows = []
    for p in range(n_prompts):
        pattern = patterns[p % len(patterns)]
        register = registers[p % len(registers)]
        variant = variants[p % len(variants)]
        prompt_offset = rng.normal(0, 2.0)
        for tech in techniques:
            for cc1 in (8, 24, 40, 56, 64, 72, 88, 104, 120):
                for w_cc in (0.0, 0.5, 1.0, 2.0):
                    for steps in (8, 30, 120):
                        # 실제 제어 응답을 흉내: 포화 + 주법별 이득 + guidance 상호작용
                        gain = {"sustain": 1.0, "staccato": 0.6,
                                "pizzicato": 0.25, "legato_slur": 0.9}[tech]
                        sat = np.tanh((cc1 - 64) / 45.0)
                        step_pen = -3.0 * np.exp(-steps / 20.0)
                        rms = (-18 + 9 * gain * sat * (0.3 + 0.7 * min(w_cc, 1.5))
                               + prompt_offset + step_pen
                               + rng.normal(0, 0.6 + 0.4 * (tech == "pizzicato")))
                        rows.append({
                            "prompt_id": f"p{p:02d}", "technique": tech, "pattern": pattern,
                            "register": register, "timing_variant": variant,
                            "cc1_final": cc1, "w_tech": 1.0, "w_cc": w_cc,
                            "sampling_steps": steps, "rms_dbfs": rms})
    return pd.DataFrame(rows)


def build(model):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", SimpleImputer(strategy="median"), NUMERIC),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(name, model, data):
    X, y, g = data[CATEGORICAL + NUMERIC], data["rms_dbfs"].to_numpy(), data["prompt_id"]
    maes, r2s = [], []
    t0 = time.time()
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
        pipe = build(model)
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[te])
        maes.append(mean_absolute_error(y[te], pred))
        r2s.append(r2_score(y[te], pred))
    return name, float(np.mean(maes)), float(np.mean(r2s)), time.time() - t0


def main():
    data = make_data()
    print(f"표본 {len(data):,}행 · prompt {data.prompt_id.nunique()}개 · "
          f"prompt 단위 GroupKFold 5겹 · 전부 기본 하이퍼파라미터")
    print("=" * 78)
    print(f"{'모델':34s}{'MAE(dB)':>10s}{'R2':>9s}{'학습+평가':>12s}")
    print("-" * 78)

    candidates = [
        ("ExtraTrees (현재)", ExtraTreesRegressor(n_estimators=300, random_state=0, n_jobs=-1)),
        ("HistGradientBoosting (sklearn)", HistGradientBoostingRegressor(random_state=0)),
    ]
    try:
        from lightgbm import LGBMRegressor
        candidates.append(("LightGBM", LGBMRegressor(random_state=0, verbose=-1)))
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor
        candidates.append(("CatBoost", CatBoostRegressor(random_state=0, verbose=0)))
    except ImportError:
        pass

    results = []
    for name, model in candidates:
        try:
            results.append(evaluate(name, model, data))
        except Exception as e:  # noqa: BLE001
            print(f"{name:34s}  실패: {type(e).__name__}: {str(e)[:40]}")
    for name, mae, r2, sec in results:
        print(f"{name:34s}{mae:10.3f}{r2:9.3f}{sec:11.1f}s")

    if results:
        best = min(results, key=lambda r: r[1])
        base = next((r for r in results if r[0].startswith("ExtraTrees")), None)
        print("-" * 78)
        print(f"최저 MAE: {best[0]} ({best[1]:.3f} dB)")
        if base and best[0] != base[0]:
            print(f"현재 ExtraTrees 대비 MAE {(1 - best[1] / base[1]) * 100:.1f}% 개선")
        elif base:
            print("현재 ExtraTrees 가 최저 — 교체 근거 없음")


if __name__ == "__main__":
    main()
