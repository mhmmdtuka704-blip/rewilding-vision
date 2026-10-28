"""Small, independently testable metrics for the frozen Week 1 scope."""

from __future__ import annotations

import numpy as np


def _valid_flat(prediction: np.ndarray, target: np.ndarray, ignore_index: int = 255):
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} != {target.shape}")
    valid = target != ignore_index
    if not np.any(valid):
        raise ValueError("Target has no valid pixels")
    pred = prediction[valid]
    truth = target[valid]
    if not np.isin(pred, [0, 1]).all() or not np.isin(truth, [0, 1]).all():
        raise ValueError("Valid pixels must contain only binary labels 0 and 1")
    return pred.astype(bool), truth.astype(bool)


def vegetation_iou(prediction: np.ndarray, target: np.ndarray, ignore_index: int = 255) -> float:
    pred, truth = _valid_flat(prediction, target, ignore_index)
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, truth).sum() / union)


def vegetation_coverage(mask: np.ndarray, ignore_index: int = 255) -> float:
    valid = mask != ignore_index
    if not np.any(valid):
        raise ValueError("Mask has no valid pixels")
    values = mask[valid]
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Valid pixels must contain only binary labels 0 and 1")
    return float(100.0 * values.mean())


def coverage_absolute_error_pp(
    prediction: np.ndarray, target: np.ndarray, ignore_index: int = 255
) -> float:
    """Absolute visible vegetation coverage error, in percentage points."""
    valid = target != ignore_index
    pred = np.where(valid, prediction, ignore_index)
    return abs(vegetation_coverage(pred, ignore_index) - vegetation_coverage(target, ignore_index))


def change_absolute_error_pp(
    before_prediction: np.ndarray,
    after_prediction: np.ndarray,
    before_target: np.ndarray,
    after_target: np.ndarray,
    ignore_index: int = 255,
) -> float:
    shapes = {
        before_prediction.shape,
        after_prediction.shape,
        before_target.shape,
        after_target.shape,
    }
    if len(shapes) != 1:
        raise ValueError("All paired predictions and targets must have matching shapes")
    common_valid = np.logical_and(before_target != ignore_index, after_target != ignore_index)
    if not np.any(common_valid):
        raise ValueError("Before and after targets have no shared valid pixels")
    before_pred = np.where(common_valid, before_prediction, ignore_index)
    after_pred = np.where(common_valid, after_prediction, ignore_index)
    before_truth = np.where(common_valid, before_target, ignore_index)
    after_truth = np.where(common_valid, after_target, ignore_index)
    predicted_change = vegetation_coverage(after_pred, ignore_index) - vegetation_coverage(
        before_pred, ignore_index
    )
    true_change = vegetation_coverage(after_truth, ignore_index) - vegetation_coverage(
        before_truth, ignore_index
    )
    return abs(predicted_change - true_change)
