"""SegFormer-B0 creation, checkpointing, and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

LABELS = {0: "non_vegetation", 1: "vegetation"}
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "model" / "ade20k"
VEGETATION_CLASSES = frozenset({"tree", "grass", "plant", "flower", "palm"})


def letterbox_rgb(image_rgb: np.ndarray, size: int):
    """Fit an RGB image into a square without changing its aspect ratio."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB image")
    height, width = image_rgb.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    image = Image.fromarray(image_rgb).resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    canvas.paste(image, (left, top))
    return np.asarray(canvas), (left, top, resized_width, resized_height)


def create_segformer(checkpoint="nvidia/mit-b0", local_files_only=False):
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[training]'") from exc
    return SegformerForSemanticSegmentation.from_pretrained(
        checkpoint,
        num_labels=2,
        id2label=LABELS,
        label2id={name: index for index, name in LABELS.items()},
        ignore_mismatched_sizes=True,
        semantic_loss_ignore_index=255,
        local_files_only=local_files_only,
    )


def set_encoder_trainable(model, trainable: bool):
    for parameter in model.segformer.parameters():
        parameter.requires_grad = trainable


class SegFormerPredictor:
    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT,
                 device: str | None = None, image_size=512, enable_adapter=True):
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "Install training dependencies: pip install -e '.[training]'"
            ) from exc
        self.torch = torch
        torch.set_num_threads(4)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(
                str(checkpoint), local_files_only=True
            ).to(self.device).eval()
        )
        labels = {int(k): v.strip().lower() for k, v in self.model.config.id2label.items()}
        if labels == LABELS:
            self.vegetation_ids = [1]
        else:
            self.vegetation_ids = [k for k, v in labels.items() if v in VEGETATION_CLASSES]
            if {labels[k] for k in self.vegetation_ids} != VEGETATION_CLASSES:
                raise ValueError("Checkpoint lacks the required trained vegetation classes")
        self.vegetation_labels = [labels[k] for k in self.vegetation_ids]
        self.adapter_metadata = None
        adapter_dir = Path(__file__).resolve().parents[2] / "model/rugd_adapter"
        if enable_adapter and (adapter_dir / "metadata.json").is_file():
            import json
            from safetensors.torch import load_file
            metadata = json.loads((adapter_dir / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("deployment_approved") and labels != LABELS:
                wrapper = make_residual_head(self.model.decode_head.classifier, self.vegetation_ids)
                wrapper.residual.load_state_dict(load_file(str(adapter_dir / "head.safetensors")))
                self.model.decode_head.classifier = wrapper.to(self.device).eval()
                self.vegetation_ids = [1]
                self.adapter_metadata = metadata
                self.vegetation_labels = ["grass", "tree", "bush"]

    def predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        original_size = image_rgb.shape[:2]
        boxed, content = letterbox_rgb(image_rgb, self.image_size)
        array = np.asarray(boxed, dtype=np.float32) / 255.0
        array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = (
            torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(self.device)
        )
        with torch.inference_mode():
            logits = self.model(pixel_values=tensor).logits
            probabilities = torch.softmax(logits, dim=1)[:, self.vegetation_ids].sum(1, keepdim=True)
            probabilities = torch.nn.functional.interpolate(
                probabilities,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        probability_square = probabilities[0, 0].cpu().numpy().astype(np.float32)
        left, top, resized_width, resized_height = content
        probability_content = probability_square[
            top : top + resized_height, left : left + resized_width
        ]
        probability = np.asarray(
            Image.fromarray(probability_content).resize(
                (original_size[1], original_size[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        return (probability >= 0.5).astype(np.uint8), probability


def make_residual_head(pretrained_classifier, vegetation_ids):
    """Trainable 1x1 binary correction on frozen semantic decoder features."""
    import torch

    class ResidualVegetationHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pretrained = pretrained_classifier
            self.residual = torch.nn.Conv2d(pretrained_classifier.in_channels, 2, 1)
            torch.nn.init.zeros_(self.residual.weight)
            torch.nn.init.zeros_(self.residual.bias)
            self.positive_ids = list(vegetation_ids)
            self.negative_ids = [i for i in range(pretrained_classifier.out_channels)
                                 if i not in vegetation_ids]

        def forward(self, features):
            logits = self.pretrained(features)
            grouped = torch.stack([
                torch.logsumexp(logits[:, self.negative_ids], dim=1),
                torch.logsumexp(logits[:, self.positive_ids], dim=1),
            ], dim=1)
            return grouped + self.residual(features)

    return ResidualVegetationHead()
