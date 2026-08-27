from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    source: Path

    @property
    def sample_rate(self) -> int:
        return int(self.raw["sample_rate"])

    @property
    def clip_seconds(self) -> float:
        return float(self.raw["clip_seconds"])

    @property
    def note_onset_seconds(self) -> float:
        return float(self.raw["note_onset_seconds"])

    @property
    def tempo_bpm(self) -> int:
        return int(self.raw["tempo_bpm"])

    @property
    def techniques(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.raw["techniques"].items()}

    @property
    def dynamics(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.raw["dynamics"].items()}

    @property
    def real_techniques(self) -> dict[str, int]:
        """실악기 반사실 기준선이 존재하는 주법만 반환한다.

        모델 축과 실연주 축을 분리하지 않으면, 주법을 하나 늘릴 때마다
        연주자가 켜야 할 테이크가 216개씩(12프롬프트x3강약x3악기x2회) 늘어난다.
        아마추어가 반복 가능한 주법만 primary audit에 넣는다. 나머지는
        실악기 기준선이 없는 generator-only 탐색 축이며 primary 결론에 쓰지 않는다.
        """
        requested = self.raw.get("real", {}).get("techniques")
        if not requested:
            return self.techniques
        allowed = self.techniques
        missing = [str(name) for name in requested if str(name) not in allowed]
        if missing:
            raise ValueError(f"real.techniques 에 techniques 에 없는 주법: {missing}")
        return {str(name): allowed[str(name)] for name in requested}

    def model_techniques_for_profile(self, profile: str) -> dict[str, int]:
        """Pilot/full은 실악기 대응 주법만, expanded만 탐색 주법까지 사용한다."""
        if profile == "expanded":
            return self.techniques
        if profile in {"pilot", "full"}:
            return self.real_techniques
        raise ValueError("profile must be one of: pilot, full, expanded")

    @property
    def full_violin_prompts(self) -> int:
        """앞에서 몇 개의 프롬프트를 바이올린 3대 전부로 녹음할 것인가.

        나머지 프롬프트는 V1 한 대로만 녹음한다. 반복 2회는 어느 쪽이든 유지한다
        (HCEL 의 기준선이라 절대 줄일 수 없다). 이렇게 하면 프롬프트 수를 늘려
        prompt 단위 부트스트랩 클러스터를 확보하면서도 녹음 시간이 폭발하지 않는다.
        """
        return int(self.raw.get("real", {}).get("full_violin_prompts", 10**9))


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Configuration must be a mapping: {source}")
    required = {
        "sample_rate",
        "clip_seconds",
        "note_onset_seconds",
        "tempo_bpm",
        "techniques",
        "dynamics",
        "model",
        "real",
        "analysis",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Missing configuration keys: {missing}")
    return ExperimentConfig(raw=raw, source=source)


def project_root_from_config(config: ExperimentConfig) -> Path:
    # configs/experiment.yaml -> repository root
    return config.source.parent.parent
