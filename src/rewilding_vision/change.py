"""Visible vegetation change computation over a shared valid footprint."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ChangeSummary:
    before_coverage_percent: float
    after_coverage_percent: float
    change_percentage_points: float
    gain_percent: float
    loss_percent: float
    unchanged_vegetation_percent: float
    evaluated_pixels: int
    relative_change_percent: float | None

    def to_dict(self):
        return asdict(self)


def calculate_change(
    before_mask: np.ndarray, after_mask: np.ndarray, valid_mask: np.ndarray | None = None
):
    if before_mask.shape != after_mask.shape:
        raise ValueError("Before and after masks must have the same shape")
    if valid_mask is None:
        valid_mask = np.ones(before_mask.shape, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != before_mask.shape or not np.any(valid_mask):
        raise ValueError("Valid overlap mask must match and contain pixels")
    before = before_mask[valid_mask]
    after = after_mask[valid_mask]
    if not np.isin(before, [0, 1]).all() or not np.isin(after, [0, 1]).all():
        raise ValueError("Change masks must be binary")
    gain = np.logical_and(before == 0, after == 1)
    loss = np.logical_and(before == 1, after == 0)
    stable = np.logical_and(before == 1, after == 1)
    before_coverage = float(100.0 * before.mean())
    after_coverage = float(100.0 * after.mean())
    summary = ChangeSummary(
        before_coverage,
        after_coverage,
        after_coverage - before_coverage,
        float(100.0 * gain.mean()),
        float(100.0 * loss.mean()),
        float(100.0 * stable.mean()),
        int(valid_mask.sum()),
        float(100.0 * (after.sum(dtype=np.float64) - before.sum(dtype=np.float64))
              / before.sum(dtype=np.float64)) if before.sum() else None,
    )
    change_map = np.full(before_mask.shape, 255, dtype=np.uint8)
    change_map[valid_mask] = 0
    change_map[np.logical_and(valid_mask, np.logical_and(before_mask == 0, after_mask == 1))] = 1
    change_map[np.logical_and(valid_mask, np.logical_and(before_mask == 1, after_mask == 0))] = 2
    change_map[np.logical_and(valid_mask, np.logical_and(before_mask == 1, after_mask == 1))] = 3
    return summary, change_map
