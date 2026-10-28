"""RUGD ontology and deterministic conversion to binary vegetation masks.

The source ontology follows the official RUGD label map. Only grass, tree, and
bush are vegetation in v1. Void pixels remain ignored; mulch is deliberately
non-vegetation because it is dead ground cover rather than visible live canopy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RugdClass:
    class_id: int
    name: str
    rgb: tuple[int, int, int]


RUGD_CLASSES = (
    RugdClass(0, "void", (0, 0, 0)),
    RugdClass(1, "dirt", (108, 64, 20)),
    RugdClass(2, "sand", (255, 229, 204)),
    RugdClass(3, "grass", (0, 102, 0)),
    RugdClass(4, "tree", (0, 255, 0)),
    RugdClass(5, "pole", (0, 153, 153)),
    RugdClass(6, "water", (0, 128, 255)),
    RugdClass(7, "sky", (0, 0, 255)),
    RugdClass(8, "vehicle", (255, 255, 0)),
    RugdClass(9, "container", (255, 0, 127)),
    RugdClass(10, "asphalt", (64, 64, 64)),
    RugdClass(11, "gravel", (255, 128, 0)),
    RugdClass(12, "building", (255, 0, 0)),
    RugdClass(13, "mulch", (153, 76, 0)),
    RugdClass(14, "rock-bed", (102, 102, 0)),
    RugdClass(15, "log", (102, 0, 0)),
    RugdClass(16, "bicycle", (0, 255, 128)),
    RugdClass(17, "person", (204, 153, 255)),
    RugdClass(18, "fence", (102, 0, 204)),
    RugdClass(19, "bush", (255, 153, 204)),
    RugdClass(20, "sign", (0, 102, 102)),
    RugdClass(21, "rock", (153, 204, 255)),
    RugdClass(22, "bridge", (102, 255, 255)),
    RugdClass(23, "concrete", (101, 101, 11)),
    RugdClass(24, "picnic-table", (114, 85, 47)),
)

VEGETATION_NAMES = frozenset({"grass", "tree", "bush"})
IGNORE_INDEX = np.uint8(255)


def color_mask_to_binary(mask_rgb: np.ndarray, strict: bool = True) -> np.ndarray:
    """Convert an HxWx3 RUGD colour mask to {0, 1, 255}.

    Unknown colours raise in strict mode. This catches corrupt masks and colour
    interpolation, a common silent source of invalid segmentation labels.
    """
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 mask, received shape {mask_rgb.shape}")

    output = np.full(mask_rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    known = np.zeros(mask_rgb.shape[:2], dtype=bool)
    for label in RUGD_CLASSES:
        selected = np.all(mask_rgb == label.rgb, axis=-1)
        known |= selected
        if label.name == "void":
            output[selected] = IGNORE_INDEX
        else:
            output[selected] = 1 if label.name in VEGETATION_NAMES else 0

    if strict and not np.all(known):
        unknown = np.unique(mask_rgb[~known].reshape(-1, 3), axis=0)
        raise ValueError(f"Mask contains unknown RUGD colours: {unknown[:10].tolist()}")
    return output
