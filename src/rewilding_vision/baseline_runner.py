"""Leakage-safe ExG tuning and held-out evaluation from the frozen manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .evaluation import SegmentationAccumulator
from .exg import segment_exg, select_threshold
from .rugd_labels import color_mask_to_binary
from .training_data import ManifestRow, read_manifest


def load_sample(row: ManifestRow):
    image = np.asarray(Image.open(row.image_path).convert("RGB"))
    mask = color_mask_to_binary(np.asarray(Image.open(row.mask_path).convert("RGB")))
    return image, mask


def run_exg_baseline(manifest: str | Path, candidates=None) -> dict:
    validation_rows = read_manifest(manifest, "validation")
    test_rows = read_manifest(manifest, "test")
    validation_samples = [load_sample(row) for row in validation_rows]
    if candidates is None:
        candidates = np.linspace(-0.5, 1.0, 101)
    threshold, validation_iou = select_threshold(validation_samples, candidates)
    accumulator = SegmentationAccumulator()
    for row in test_rows:
        image, target = load_sample(row)
        accumulator.update(segment_exg(image, threshold), target)
    result = {
        "method": "normalized_excess_green",
        "selected_threshold": threshold,
        "validation_global_iou": validation_iou,
        "test": accumulator.compute().to_dict(),
        "validation_images": len(validation_rows),
        "test_images": len(test_rows),
    }
    if len(test_rows) >= 2:
        result["test_uncertainty"] = accumulator.confidence_intervals()
    return result
