"""Explicit one-split model evaluation, kept separate from training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .evaluation import SegmentationAccumulator
from .rugd_labels import color_mask_to_binary
from .segformer import SegFormerPredictor
from .training_data import read_manifest


def evaluate_checkpoint(checkpoint: str | Path, manifest: str | Path, split="test") -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test")
    predictor = SegFormerPredictor(checkpoint)
    rows = read_manifest(manifest, split)
    accumulator = SegmentationAccumulator()
    for row in rows:
        image = np.asarray(Image.open(row.image_path).convert("RGB"))
        target = color_mask_to_binary(np.asarray(Image.open(row.mask_path).convert("RGB")))
        prediction, _ = predictor.predict(image)
        accumulator.update(prediction, target)
    result = {
        "checkpoint": str(checkpoint),
        "split": split,
        **accumulator.compute().to_dict(),
    }
    if len(rows) >= 2:
        result["uncertainty"] = accumulator.confidence_intervals()
    return result
