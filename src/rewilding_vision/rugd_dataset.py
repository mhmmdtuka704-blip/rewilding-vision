"""Manifest discovery and audit utilities for an extracted RUGD download."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .rugd_labels import color_mask_to_binary
from .splits import assign_group_splits


@dataclass(frozen=True)
class Sample:
    sequence: str
    image_path: Path
    mask_path: Path


def discover_samples(image_root: Path, mask_root: Path) -> list[Sample]:
    """Pair images and masks by sequence-relative stem; fail on ambiguity."""
    extensions = {".png", ".jpg", ".jpeg"}
    images = {
        p.relative_to(image_root).with_suffix(""): p
        for p in image_root.rglob("*")
        if p.suffix.lower() in extensions
    }
    masks = {p.relative_to(mask_root).with_suffix(""): p for p in mask_root.rglob("*.png")}
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            f"Unpaired files: {len(missing_masks)} images, {len(missing_images)} masks"
        )
    samples = []
    for key in sorted(images):
        parts = key.parts
        sequence = parts[0] if len(parts) > 1 else images[key].parent.name
        samples.append(Sample(sequence, images[key], masks[key]))
    if not samples:
        raise ValueError("No paired RUGD samples found")
    return samples


def audit_samples(samples: list[Sample]) -> dict:
    counts = Counter()
    pixels = Counter()
    for sample in samples:
        image = np.asarray(Image.open(sample.image_path).convert("RGB"))
        source_mask = np.asarray(Image.open(sample.mask_path).convert("RGB"))
        if image.shape[:2] != source_mask.shape[:2]:
            raise ValueError(f"Image/mask size mismatch for {sample.image_path}")
        binary = color_mask_to_binary(source_mask)
        counts[sample.sequence] += 1
        pixels["non_vegetation"] += int((binary == 0).sum())
        pixels["vegetation"] += int((binary == 1).sum())
        pixels["ignored"] += int((binary == 255).sum())
    return {
        "samples": len(samples),
        "sequences": len(counts),
        "samples_per_sequence": dict(sorted(counts.items())),
        "pixels": dict(pixels),
    }


def write_split_manifest(samples: list[Sample], output_path: Path, seed: int = 20260812):
    assignments = assign_group_splits([sample.sequence for sample in samples], seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "split", "image_path", "mask_path"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sequence": sample.sequence,
                    "split": assignments[sample.sequence],
                    "image_path": str(sample.image_path),
                    "mask_path": str(sample.mask_path),
                }
            )
    return assignments
