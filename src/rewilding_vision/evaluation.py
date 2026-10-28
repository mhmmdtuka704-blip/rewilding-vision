"""Dataset-level evaluation shared by classical and deep-learning models."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class EvaluationSummary:
    vegetation_iou: float
    coverage_mae_pp: float
    evaluated_images: int
    valid_pixels: int
    vegetation_intersection: int
    vegetation_union: int

    def to_dict(self) -> dict:
        return asdict(self)


class SegmentationAccumulator:
    """Aggregate global IoU and mean per-image coverage error."""

    def __init__(self, ignore_index: int = 255):
        self.ignore_index = ignore_index
        self.intersection = 0
        self.union = 0
        self.valid_pixels = 0
        self.coverage_errors: list[float] = []
        self.image_intersections: list[int] = []
        self.image_unions: list[int] = []

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch: {prediction.shape} != {target.shape}")
        valid = target != self.ignore_index
        if not np.any(valid):
            raise ValueError("Target has no valid pixels")
        pred = prediction[valid]
        truth = target[valid]
        if not np.isin(pred, [0, 1]).all() or not np.isin(truth, [0, 1]).all():
            raise ValueError("Valid pixels must contain binary labels")
        pred_bool, truth_bool = pred.astype(bool), truth.astype(bool)
        image_intersection = int(np.logical_and(pred_bool, truth_bool).sum())
        image_union = int(np.logical_or(pred_bool, truth_bool).sum())
        self.intersection += image_intersection
        self.union += image_union
        self.image_intersections.append(image_intersection)
        self.image_unions.append(image_union)
        self.valid_pixels += int(valid.sum())
        self.coverage_errors.append(float(abs(100.0 * pred.mean() - 100.0 * truth.mean())))

    def compute(self) -> EvaluationSummary:
        if not self.coverage_errors:
            raise ValueError("No samples were evaluated")
        score = 1.0 if self.union == 0 else self.intersection / self.union
        return EvaluationSummary(
            vegetation_iou=float(score),
            coverage_mae_pp=float(np.mean(self.coverage_errors)),
            evaluated_images=len(self.coverage_errors),
            valid_pixels=self.valid_pixels,
            vegetation_intersection=self.intersection,
            vegetation_union=self.union,
        )

    def confidence_intervals(self, iterations=2000, seed=20260812) -> dict:
        """Bootstrap images while preserving pixel weighting in each replicate."""
        if len(self.coverage_errors) < 2:
            raise ValueError("At least two evaluated images are required")
        intersections = np.asarray(self.image_intersections, dtype=np.float64)
        unions = np.asarray(self.image_unions, dtype=np.float64)
        errors = np.asarray(self.coverage_errors, dtype=np.float64)
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(errors), size=(iterations, len(errors)))
        sampled_intersections = intersections[indices].sum(axis=1)
        sampled_unions = unions[indices].sum(axis=1)
        sampled_iou = np.divide(
            sampled_intersections,
            sampled_unions,
            out=np.ones_like(sampled_intersections),
            where=sampled_unions > 0,
        )
        sampled_mae = errors[indices].mean(axis=1)
        return {
            "confidence_level": 0.95,
            "method": "image-level percentile bootstrap",
            "iterations": iterations,
            "vegetation_iou": [
                float(np.quantile(sampled_iou, 0.025)),
                float(np.quantile(sampled_iou, 0.975)),
            ],
            "coverage_mae_pp": [
                float(np.quantile(sampled_mae, 0.025)),
                float(np.quantile(sampled_mae, 0.975)),
            ],
        }


def bootstrap_confidence_interval(
    values, statistic=np.mean, confidence=0.95, iterations=2000, seed=20260812
):
    """Percentile bootstrap interval for transparent uncertainty reporting."""
    data = np.asarray(list(values), dtype=np.float64)
    if data.size < 2:
        raise ValueError("At least two observations are required for a confidence interval")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(iterations, data.size))
    samples = np.asarray([statistic(data[index]) for index in indices])
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))
