"""Reproducibility controls and runtime provenance."""

from __future__ import annotations

import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def seed_everything(seed: int, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def runtime_manifest(extra: dict | None = None) -> dict:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "process_id": os.getpid(),
    }
    try:
        import torch

        manifest.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            }
        )
    except ImportError:
        manifest["torch"] = None
    if extra:
        manifest.update(extra)
    return manifest


def write_json(path: str | Path, payload: dict):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
