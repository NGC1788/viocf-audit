"""학습된 음악 표현(MERT)으로 해석가능 특징을 교차확인한다.

## 왜 넣는가

이 연구의 지표는 전부 **해석가능 특징**(RMS, 어택, centroid, cents…) 위에서 계산된다.
해석 가능하다는 게 장점이지만, 뒤집으면 **우리가 고른 21개 특징이 놓친 변화는
영원히 안 보인다**는 뜻이다. "누출이 없다"는 결론이 실제로는 "우리 특징으로는 안
보인다"일 수 있다.

MERT 는 음악 오디오로 자기지도 학습된 표현이다. 우리가 무엇을 재야 할지 미리 정하지
않았으므로, 여기서 차이가 보이는데 해석가능 특징에서 안 보이면 **특징 설계의 구멍**이
드러난다. 반대로 둘 다 보이면 결론이 훨씬 튼튼해진다.

즉 MERT 는 headline 지표를 **대체하지 않는다.** 감사의 감사다.

## 왜 C2ST 로 비교하는가

임베딩은 768~1024 차원이라 energy distance 같은 거리 기반 통계량이 차원의 저주로
둔해진다. 분류기 이표본 검정(C2ST)은 분류기가 판별 방향을 스스로 찾으므로 그 영향을
덜 받고, 결과가 **정확도**라 해석이 바로 된다
("모델 출력과 실연주를 78% 로 구별할 수 있다"). -> metrics.classifier_two_sample_test

## 환경 요구사항 (서버에서 반드시 확인)

- `transformers` + `torch >= 2.6`. MERT 저장소는 safetensors 를 제공하지 않고
  `pytorch_model.bin` 만 올라와 있는데, transformers 는 CVE-2025-32434 때문에
  torch < 2.6 에서 `torch.load` 를 거부한다. **분석용 venv 의 torch 를 올려야 한다.**
- 첫 실행 때 모델을 내려받는다(95M 은 약 0.4 GB). 오프라인 서버면 미리 캐시할 것.
- `trust_remote_code=True` 가 필요하다. MERT 는 저장소에 커스텀 모델 코드를 함께 둔다.
- **`transformers` 는 4.x 로 상한을 건다.** MERT 의 원격 코드는 4.x 시절 것이라
  5.x 에서는 forward 가 `output_hidden_states` 를 받기만 하고 채우지 않아
  `hidden_states` 가 None 으로 온다(실측: 4.57.6 정상 / 5.16.1 실패).
  이 모듈에 forward hook 대비책이 있지만 정상 경로를 쓰는 편이 안전하다.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# MERT-v1-95M: 95M 파라미터, 24 kHz 입력, 13 레이어 x 768 차원.
# 330M 도 있으나 95M 이 추론 비용 대비 충분하다는 것이 원 논문의 보고다.
DEFAULT_MODEL_ID = "m-a-p/MERT-v1-95M"

# 어느 레이어를 쓸 것인가.
# 음악 SSL 모델은 **하위 레이어일수록 음향적(음색·피치), 상위일수록 의미적(장르·무드)**
# 이라는 것이 층별 분석 연구들의 일관된 보고다. 우리가 감사하는 건 연주 표현이므로
# 중하위 레이어가 맞다. 전 레이어 평균은 성격이 다른 층을 섞어 신호를 흐린다.
# 기본값은 중하위 대역이며, 층별 민감도는 결과에 함께 기록해 사후 확인할 수 있게 한다.
DEFAULT_LAYERS = (3, 4, 5, 6)


def _require_backend():
    try:
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise RuntimeError(
            "MERT 임베딩에는 transformers 와 torch 가 필요하다.\n"
            "  pip install 'transformers>=4.40' 'torch>=2.6'\n"
            "torch 2.6 이상이어야 하는 이유는 이 파일 상단 주석 참조."
        ) from exc
    version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if version < (2, 6):
        raise RuntimeError(
            f"torch {torch.__version__} 로는 MERT 를 로드할 수 없다(>=2.6 필요). "
            "MERT 는 safetensors 를 제공하지 않고, transformers 가 CVE-2025-32434 때문에 "
            "구버전 torch 의 torch.load 를 거부한다."
        )
    return torch, AutoModel, Wav2Vec2FeatureExtractor


class MertEmbedder:
    """클립 하나 -> 고정 길이 임베딩 벡터.

    시간축은 평균으로 접는다. 우리가 비교하는 단위가 '클립'이기 때문이다
    (프레임 단위 비교는 온셋 정렬 문제를 다시 끌어들인다).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        layers: Sequence[int] = DEFAULT_LAYERS,
        device: str | None = None,
    ) -> None:
        torch, AutoModel, Wav2Vec2FeatureExtractor = _require_backend()
        self._torch = torch
        self.model_id = model_id
        self.layers = tuple(int(layer) for layer in layers)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        # ⚠ transformers 5.x + MERT 의 커스텀 원격코드 조합에서는 forward 의
        # output_hidden_states=True 가 전달되지 않아 hidden_states 가 None 으로 온다.
        # 설정에 직접 박아야 한다(실측으로 확인).
        self.model.config.output_hidden_states = True
        self.model.eval().to(device)
        self._captured: list[Any] = []
        self._install_layer_hooks()
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.sample_rate = int(self.extractor.sampling_rate)

    def _install_layer_hooks(self) -> None:
        """중간 레이어를 forward hook 으로 직접 낚아챈다.

        왜 필요한가: MERT 의 원격 모델 코드는 transformers 4.x 시절에 쓰였다.
        transformers 5.x 에서는 forward 가 `output_hidden_states` 인자를 **받기는 하지만
        채우지 않아서** `outputs.hidden_states` 가 None 으로 온다(실측 확인).
        훅은 모델이 무엇을 반환하든 무관하게 동작하므로 버전 변화에 견딘다.

        transformers 4.x 를 쓰면 정상 경로가 동작하고 훅 결과는 무시된다.
        """
        encoder = getattr(self.model, "encoder", None)
        layers = getattr(encoder, "layers", None) if encoder is not None else None
        if layers is None:
            return

        def make_hook(index: int):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                self._captured.append((index, tensor.detach()))
            return hook

        for index, layer in enumerate(layers):
            layer.register_forward_hook(make_hook(index))

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        torch = self._torch
        self._captured = []
        if sample_rate != self.sample_rate:
            import librosa

            samples = librosa.resample(
                np.asarray(samples, dtype=np.float32),
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
            )
        inputs = self.extractor(
            np.asarray(samples, dtype=np.float32),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden = getattr(outputs, "hidden_states", None)
        if hidden is None:
            # 훅으로 직접 낚아챈 것을 쓴다 (아래 _install_layer_hooks 주석 참조)
            hidden = self._captured or None
        if hidden is None:
            raise RuntimeError(
                "MERT 의 중간 레이어를 얻지 못했다. transformers 버전을 확인할 것 "
                "(4.x 권장). 훅 경로도 실패했다면 모델 구조가 바뀐 것이다."
            )
        if isinstance(hidden, list) and hidden and isinstance(hidden[0], tuple):
            # 훅 경로: (레이어번호, 텐서) 목록 -> 레이어 순으로 정렬
            hidden = [tensor for _, tensor in sorted(hidden, key=lambda pair: pair[0])]
        # hidden_states: (레이어, 배치, 프레임, 차원)
        stacked = torch.stack(list(hidden)).squeeze(1)
        n_layers = stacked.shape[0]
        chosen = [layer for layer in self.layers if 0 <= layer < n_layers]
        if not chosen:
            raise ValueError(
                f"레이어 {self.layers} 가 모델의 레이어 수 {n_layers} 범위 밖이다"
            )
        # 선택한 레이어를 시간축 평균 후 이어붙인다(레이어별 정보를 섞지 않는다).
        pooled = [stacked[layer].mean(dim=0) for layer in chosen]
        return torch.cat(pooled).float().cpu().numpy()

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "layers": list(self.layers),
            "device": self.device,
            "sample_rate": self.sample_rate,
        }


def embed_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    project_root: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    layers: Sequence[int] = DEFAULT_LAYERS,
    device: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """manifest 의 오디오를 전부 임베딩해 parquet/csv 로 저장한다.

    재개 가능: 출력 파일이 이미 있으면 거기 있는 clip_id 는 건너뛴다.
    20만 클립 규모라 중간에 죽는 것을 전제로 짠다.
    """
    from .audio import read_audio

    root = Path(project_root).resolve()
    manifest = pd.read_csv(manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if output.exists():
        try:
            done = set(pd.read_csv(output, usecols=["clip_id"])["clip_id"].astype(str))
        except Exception:  # noqa: BLE001 - 손상된 파일이면 처음부터 다시
            done = set()

    embedder = MertEmbedder(model_id=model_id, layers=layers, device=device)
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(embedder.describe(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    processed = 0
    for record in manifest.to_dict(orient="records"):
        clip_id = str(record.get("clip_id"))
        if clip_id in done:
            continue
        audio_path = Path(str(record.get("audio_path", "")))
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        if not audio_path.exists():
            continue
        try:
            audio = read_audio(audio_path, mono=True)
            vector = embedder.embed(audio.samples, audio.sample_rate)
        except Exception as exc:  # noqa: BLE001 - 실패한 행도 남긴다
            rows.append({"clip_id": clip_id, "embed_error": f"{type(exc).__name__}: {exc}"})
            continue
        row: dict[str, Any] = {
            "clip_id": clip_id,
            "source": record.get("source"),
            "prompt_id": record.get("prompt_id"),
            "technique": record.get("technique"),
            "dynamic_label": record.get("dynamic_label"),
            "analysis_tier": record.get("analysis_tier"),
            "noise_group": record.get("noise_group"),
        }
        row.update({f"e{index:04d}": float(value) for index, value in enumerate(vector)})
        rows.append(row)
        processed += 1
        if limit is not None and processed >= limit:
            break

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    header = not output.exists()
    frame.to_csv(output, mode="a" if output.exists() else "w", header=header, index=False)
    return frame


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("e") and column[1:].isdigit()]


def embedding_contrast_c2st(
    frame: pd.DataFrame,
    baseline_dynamic: str = "mf",
    target_dynamic: str = "f",
    seed: int = 20260826,
) -> pd.DataFrame:
    """주법별로 '모델의 강약 변화'와 '실연주의 강약 변화'를 임베딩 공간에서 비교한다.

    각 클립의 (target - baseline) 차이 벡터를 만들고, 모델 쪽 차이벡터 집합과
    실연주 쪽 차이벡터 집합을 C2ST 로 구별해 본다.

      정확도 0.5 -> 모델의 변화가 실연주의 변화와 구별되지 않는다 (좋음)
      정확도 1.0 -> 완전히 다른 방식으로 변한다 (제어가 실악기와 다르게 작동)
    """
    from .metrics import classifier_two_sample_test

    columns = embedding_columns(frame)
    if not columns or "analysis_tier" not in frame.columns:
        return pd.DataFrame()
    data = frame.loc[frame["analysis_tier"].eq("real_counterfactual_primary")]

    records: list[dict[str, Any]] = []
    for technique, group in data.groupby("technique"):
        deltas: dict[str, list[np.ndarray]] = {"model": [], "real": []}
        for source in ("model", "real"):
            side = group.loc[group["source"].eq(source)]
            for _, cell in side.groupby("prompt_id"):
                base = cell.loc[cell["dynamic_label"].eq(baseline_dynamic), columns]
                target = cell.loc[cell["dynamic_label"].eq(target_dynamic), columns]
                if base.empty or target.empty:
                    continue
                deltas[source].append(
                    target.to_numpy(dtype=float).mean(axis=0)
                    - base.to_numpy(dtype=float).mean(axis=0)
                )
        if len(deltas["model"]) < 5 or len(deltas["real"]) < 5:
            continue
        result = classifier_two_sample_test(
            np.vstack(deltas["model"]), np.vstack(deltas["real"]), seed=seed
        )
        records.append({
            "technique": str(technique),
            "contrast": f"dyn {baseline_dynamic}->{target_dynamic}",
            "c2st_accuracy": result["accuracy"],
            "c2st_p_value": result["p_value"],
            "n": result["n"],
            "distinguishable": bool(
                np.isfinite(result["p_value"]) and result["p_value"] < 0.05
            ),
        })
    return pd.DataFrame(records)


def summarise(frames: Iterable[pd.DataFrame]) -> dict[str, Any]:
    merged = [frame for frame in frames if frame is not None and not frame.empty]
    if not merged:
        return {"techniques": 0}
    combined = pd.concat(merged, ignore_index=True)
    return {
        "techniques": int(combined["technique"].nunique()),
        "mean_c2st_accuracy": float(combined["c2st_accuracy"].mean()),
        "distinguishable_fraction": float(combined["distinguishable"].mean()),
        "note": (
            "정확도가 0.5 에 가까울수록 모델의 제어 변화가 실연주의 변화와 "
            "임베딩 공간에서 구별되지 않는다는 뜻이다(좋음)."
        ),
    }


__all__ = [
    "DEFAULT_LAYERS",
    "DEFAULT_MODEL_ID",
    "MertEmbedder",
    "embed_manifest",
    "embedding_columns",
    "embedding_contrast_c2st",
    "summarise",
]

assert math
