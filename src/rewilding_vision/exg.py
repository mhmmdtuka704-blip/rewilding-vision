"""Deterministic Excess Green baseline and validation-only threshold tuning."""

from __future__ import annotations

import numpy as np

from .evaluation import SegmentationAccumulator


def excess_green(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, received {image_rgb.shape}")
    rgb = image_rgb.astype(np.float32) / 255.0
    channel_sum = rgb.sum(axis=-1, keepdims=True)
    normalized = np.divide(rgb, channel_sum, out=np.zeros_like(rgb), where=channel_sum > 0)
    return 2.0 * normalized[..., 1] - normalized[..., 0] - normalized[..., 2]


def segment_exg(image_rgb: np.ndarray, threshold: float) -> np.ndarray:
    return (excess_green(image_rgb) >= threshold).astype(np.uint8)


def select_threshold(
    validation_samples, candidates: np.ndarray | None = None
) -> tuple[float, float]:
    """Select threshold using validation data only; return threshold and global IoU."""
    samples = list(validation_samples)
    if not samples:
        raise ValueError("At least one validation sample is required")
    if candidates is None:
        candidates = np.linspace(-0.5, 1.0, 101)
    best_threshold, best_score = None, -1.0
    for threshold in candidates:
        accumulator = SegmentationAccumulator()
        for image, target in samples:
            accumulator.update(segment_exg(image, float(threshold)), target)
        score = accumulator.compute().vegetation_iou
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold, best_score
