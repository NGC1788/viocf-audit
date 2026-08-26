from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def collect_preflight(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    disk = shutil.disk_usage(root)
    report: dict[str, Any] = {
        "project_root": str(root),
        "platform": platform.platform(),
        "python": sys.version,
        "disk_total_gib": disk.total / (1024**3),
        "disk_free_gib": disk.free / (1024**3),
        "commands": {
            "nvidia_smi": _run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu",
                    "--format=csv,noheader",
                ]
            ),
            "ffmpeg": _run(["ffmpeg", "-version"]),
            "git": _run(["git", "--version"]),
        },
    }
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gib": (
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if torch.cuda.is_available()
                else None
            ),
        }
    except Exception as exc:  # noqa: BLE001 - report optional torch import/runtime failures
        report["torch"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    gpu_hardware_available = report["commands"]["nvidia_smi"].get("returncode") == 0
    cuda_runtime_available = bool(report.get("torch", {}).get("cuda_available", False))
    report["ready_for_analysis"] = report["disk_free_gib"] >= 20.0
    report["ready_for_violet_hardware"] = bool(
        gpu_hardware_available and report["disk_free_gib"] >= 50.0
    )
    report["ready_for_violet_runtime"] = bool(
        gpu_hardware_available
        and cuda_runtime_available
        and report["disk_free_gib"] >= 50.0
    )
    # Kept for old notebooks: this means the machine is suitable, not that the
    # separate vendor/VIOLET/.venv-violet runtime has already been installed.
    report["ready_for_violet"] = report["ready_for_violet_hardware"]
    return report


def write_preflight(project_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    report = collect_preflight(project_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
