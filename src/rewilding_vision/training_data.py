"""Manifest-backed PyTorch dataset with synchronized image/mask transforms."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .rugd_labels import color_mask_to_binary


@dataclass(frozen=True)
class ManifestRow:
    sequence: str
    split: str
    image_path: Path
    mask_path: Path


def read_manifest(path: str | Path, split: str | None = None) -> list[ManifestRow]:
    manifest_path = Path(path)
    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence", "split", "image_path", "mask_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Manifest is missing fields: {sorted(required - set(reader.fieldnames or []))}"
            )
        for item in reader:
            if split is None or item["split"] == split:
                rows.append(
                    ManifestRow(
                        item["sequence"],
                        item["split"],
                        Path(item["image_path"]),
                        Path(item["mask_path"]),
                    )
                )
    if not rows:
        raise ValueError(f"No rows found for split={split!r}")
    return rows


def paired_transform(
    image: Image.Image, mask: Image.Image, size: int, training: bool, rng: random.Random
):
    """Apply geometry identically and colour augmentation to the RGB image only."""
    if image.size != mask.size:
        raise ValueError(f"Image/mask size mismatch: {image.size} != {mask.size}")
    width, height = image.size
    scale = max(size / width, size / height)
    if training:
        scale *= rng.uniform(1.0, 1.25)
    resized = (max(size, round(width * scale)), max(size, round(height * scale)))
    image = image.resize(resized, Image.Resampling.BILINEAR)
    mask = mask.resize(resized, Image.Resampling.NEAREST)
    max_left, max_top = resized[0] - size, resized[1] - size
    left = rng.randint(0, max_left) if training and max_left else max_left // 2
    top = rng.randint(0, max_top) if training and max_top else max_top // 2
    box = (left, top, left + size, top + size)
    image, mask = image.crop(box), mask.crop(box)
    if training and rng.random() < 0.5:
        image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
    if training:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.88, 1.12))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.90, 1.10))
    return image, mask


class RugdTorchDataset:
    """A lazily imported torch Dataset to keep Week 1 lightweight."""

    def __init__(self, rows: list[ManifestRow], size=512, training=False, seed=20260812):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Install the training dependencies: pip install -e '.[training]'"
            ) from exc
        self.rows, self.size, self.training, self.seed = rows, size, training, seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        import torch

        row = self.rows[index]
        image = Image.open(row.image_path).convert("RGB")
        source_mask = np.asarray(Image.open(row.mask_path).convert("RGB"))
        binary_mask = Image.fromarray(color_mask_to_binary(source_mask))
        rng = random.Random(self.seed + self.epoch * len(self.rows) + index)
        image, binary_mask = paired_transform(image, binary_mask, self.size, self.training, rng)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std
        pixels = torch.from_numpy(image_array.transpose(2, 0, 1).copy()).float()
        labels = torch.from_numpy(np.asarray(binary_mask, dtype=np.uint8).copy()).long()
        return {
            "pixel_values": pixels,
            "labels": labels,
            "sequence": row.sequence,
            "image_path": str(row.image_path),
        }
