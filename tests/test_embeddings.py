"""MERT 임베딩 축의 성질을 못박는 테스트.

MERT 가 없는 환경에서는 건너뛴다(분석용 venv 에는 torch 를 넣지 않는 방침).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="MERT 임베딩은 .venv-embed 에서만 검사한다")
pytest.importorskip("transformers", reason="MERT 임베딩은 .venv-embed 에서만 검사한다")

SAMPLE_RATE = 48000


def _tone(frequency: float = 440.0, amplitude: float = 0.2, brightness: float = 1.0):
    time = np.arange(int(5.0 * SAMPLE_RATE)) / SAMPLE_RATE
    wave = sum(
        harmonic ** (-2.0 / brightness) * np.sin(2 * np.pi * frequency * harmonic * time)
        for harmonic in range(1, 13)
    )
    return (wave / np.max(np.abs(wave)) * amplitude).astype(np.float32)


def _cosine(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture(scope="module")
def embedder():
    from viocf.embeddings import MertEmbedder

    try:
        return MertEmbedder(device="cpu")
    except Exception as exc:  # noqa: BLE001 - 모델 다운로드 불가 등
        pytest.skip(f"MERT 를 로드할 수 없다: {type(exc).__name__}")


def test_mert_is_deterministic(embedder) -> None:
    a = embedder.embed(_tone(), SAMPLE_RATE)
    b = embedder.embed(_tone(), SAMPLE_RATE)
    assert _cosine(a, b) > 0.9999


def test_mert_separates_pitch_and_timbre(embedder) -> None:
    base = embedder.embed(_tone(440.0), SAMPLE_RATE)
    assert _cosine(base, embedder.embed(_tone(660.0), SAMPLE_RATE)) < 0.85
    assert _cosine(base, embedder.embed(_tone(440.0, brightness=3.0), SAMPLE_RATE)) < 0.97


def test_mert_is_level_invariant_by_design(embedder) -> None:
    """MERT 는 음량을 보지 못한다. 이 사실을 테스트로 못박는다.

    Wav2Vec2FeatureExtractor 가 입력을 정규화하기 때문이다. 이걸 모르고 쓰면
    "MERT 로 봐도 다이내믹 누출이 없다"는 거짓 결론이 나온다 — 볼 수가 없는 것이다.
    그래서 결과 해석 시 loud 군은 MERT 축에서 제외해야 한다.
    """
    loud = embedder.embed(_tone(440.0, amplitude=0.2), SAMPLE_RATE)
    quiet = embedder.embed(_tone(440.0, amplitude=0.05), SAMPLE_RATE)
    assert _cosine(loud, quiet) > 0.999, (
        "MERT 가 음량을 구별하기 시작했다면 이 축의 해석 방침을 다시 세워야 한다"
    )
