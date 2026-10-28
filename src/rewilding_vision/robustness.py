"""Controlled green-object stress fixtures, never a substitute for real evaluation."""

import numpy as np

from .rugd_labels import RUGD_CLASSES

HARD_NEGATIVE_NAMES = frozenset({"person", "container", "vehicle", "bicycle", "sign"})


def hard_negative_masks(mask_rgb):
    return {
        c.name: np.all(mask_rgb == c.rgb, axis=-1)
        for c in RUGD_CLASSES
        if c.name in HARD_NEGATIVE_NAMES
    }


def recolor_objects_green(image_rgb, mask_rgb):
    if image_rgb.shape != mask_rgb.shape or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Image and RGB annotation shapes must match")
    out = image_rgb.copy()
    masks = hard_negative_masks(mask_rgb)
    selected = np.logical_or.reduce(list(masks.values()))
    light = np.maximum(35, image_rgb.astype(np.float32).mean(axis=-1))
    green = np.stack([light * 0.35, np.minimum(255, light * 1.65), light * 0.30], axis=-1)
    out[selected] = green[selected].astype(np.uint8)
    return out
