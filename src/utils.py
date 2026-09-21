"""Shared helpers: seeding, config loading and result serialisation."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

RESULTS_DIR = Path("results")


def set_seed(seed: int) -> int:
    """Seed Python, NumPy and (when installed) PyTorch, including CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def load_config(path: Path | str) -> dict[str, Any]:
    """Read a YAML config file."""
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_results(
    payload: dict[str, Any],
    name: str,
    results_dir: Path | str = RESULTS_DIR,
) -> Path:
    """Write `payload` to results/<name>.json with a UTC timestamp attached."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    enriched = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **payload,
    }
    out_path = results_dir / f"{name}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(enriched, handle, ensure_ascii=False, indent=2)
    return out_path


def describe_environment() -> dict[str, Any]:
    """Capture the runtime details a reader needs to judge reproducibility."""
    import platform

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        info["torch"] = None
    return info
