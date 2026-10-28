"""Dependency-light visual outputs for vegetation and change."""

from __future__ import annotations

import numpy as np


def vegetation_overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha=0.42) -> np.ndarray:
    if image_rgb.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions differ")
    output = image_rgb.astype(np.float32).copy()
    green = np.zeros_like(output)
    green[..., 1] = 230
    selected = mask == 1
    output[selected] = (1.0 - alpha) * output[selected] + alpha * green[selected]
    return np.clip(output, 0, 255).astype(np.uint8)


def colorize_change(change_map: np.ndarray) -> np.ndarray:
    """White=no vegetation, green=gain, red=loss, teal=stable, grey=invalid."""
    colors = {
        0: (245, 245, 240),
        1: (44, 160, 90),
        2: (210, 65, 65),
        3: (25, 105, 92),
        255: (100, 100, 100),
    }
    output = np.zeros((*change_map.shape, 3), dtype=np.uint8)
    for value, color in colors.items():
        output[change_map == value] = color
    return output


def horizontal_panel(images: list[np.ndarray], gap=12, background=(246, 244, 237)) -> np.ndarray:
    if not images:
        raise ValueError("At least one image is required")
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images) + gap * (len(images) - 1)
    panel = np.full((height, width, 3), background, dtype=np.uint8)
    cursor = 0
    for image in images:
        panel[: image.shape[0], cursor : cursor + image.shape[1]] = image
        cursor += image.shape[1] + gap
    return panel
