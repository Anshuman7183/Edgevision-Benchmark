"""Shared utilities for reproducible CPU-first experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set common random seeds for reproducible local runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = False) -> torch.device:
    """Select CUDA only when explicitly requested and available."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_size_mb(path: str | Path) -> float:
    """Return file size in megabytes."""
    file_path = Path(path)
    return file_path.stat().st_size / (1024 * 1024)


def save_json(data: dict[str, Any] | list[Any], path: str | Path) -> Path:
    """Save JSON data with stable indentation."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    return output_path
