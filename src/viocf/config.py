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
