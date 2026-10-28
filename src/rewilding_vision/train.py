"""Auditable SegFormer training loop; the held-out test split is never loaded."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .evaluation import SegmentationAccumulator
from .reproducibility import runtime_manifest, seed_everything, write_json
from .segformer import create_segformer, set_encoder_trainable
from .training_data import RugdTorchDataset, read_manifest


def _dependencies():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[training]'") from exc
    return torch, DataLoader


def build_dataloaders(config: dict):
    torch, DataLoader = _dependencies()
    data = config["data"]
    training = config["training"]
    train_rows = read_manifest(data["manifest"], "train")
    validation_rows = read_manifest(data["manifest"], "validation")
    train_dataset = RugdTorchDataset(
        train_rows, data["image_size"], True, config["project"]["seed"]
    )
    validation_dataset = RugdTorchDataset(
        validation_rows, data["image_size"], False, config["project"]["seed"]
    )
    generator = torch.Generator().manual_seed(config["project"]["seed"])
    kwargs = {
        "batch_size": training["batch_size"],
        "num_workers": data["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **kwargs)
    return train_dataset, train_loader, validation_loader


def _predictions_from_logits(logits, labels):
    import torch

    logits = torch.nn.functional.interpolate(
        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
    )
    return logits.argmax(dim=1)


def validate_model(model, loader, device):
    import torch

    model.eval()
    accumulator = SegmentationAccumulator()
    losses = []
    with torch.inference_mode():
        for batch in loader:
            pixels, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
            outputs = model(pixel_values=pixels, labels=labels)
            losses.append(float(outputs.loss.detach().cpu()))
            predictions = _predictions_from_logits(outputs.logits, labels).cpu().numpy()
            targets = labels.cpu().numpy()
            for prediction, target in zip(predictions, targets):
                accumulator.update(prediction, target)
    return float(np.mean(losses)), accumulator.compute()


def train_segformer(config: dict) -> dict:
    torch, _ = _dependencies()
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        SummaryWriter = None
    seed_everything(config["project"]["seed"])
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    if history_path.exists():
        raise FileExistsError(
            f"Refusing to mix experiments in existing run directory: {output_dir}"
        )
    write_json(output_dir / "run_manifest.json", runtime_manifest({"config": config}))
    train_dataset, train_loader, validation_loader = build_dataloaders(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_segformer(config["model"]["checkpoint"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    use_amp = bool(config["training"]["use_amp"] and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(output_dir / "tensorboard") if SummaryWriter else None
    best_iou, stale_epochs, history = -math.inf, 0, []

    for epoch in range(config["training"]["epochs"]):
        train_dataset.set_epoch(epoch)
        encoder_trainable = epoch >= config["training"]["frozen_encoder_epochs"]
        set_encoder_trainable(model, encoder_trainable)
        model.train()
        losses = []
        for batch in train_loader:
            pixels, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                output = model(pixel_values=pixels, labels=labels)
                loss = output.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation_loss, summary = validate_model(model, validation_loader, device)
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            **summary.to_dict(),
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if writer:
            writer.add_scalar("loss/train", record["train_loss"], epoch + 1)
            writer.add_scalar("loss/validation", validation_loss, epoch + 1)
            writer.add_scalar("metrics/vegetation_iou", summary.vegetation_iou, epoch + 1)
            writer.add_scalar("metrics/coverage_mae_pp", summary.coverage_mae_pp, epoch + 1)
        if summary.vegetation_iou > best_iou:
            best_iou, stale_epochs = summary.vegetation_iou, 0
            model.save_pretrained(output_dir / "best_model")
        else:
            stale_epochs += 1
            if stale_epochs >= config["training"]["patience"]:
                break
    if writer:
        writer.close()
    result = {
        "best_validation_iou": best_iou,
        "epochs_completed": len(history),
        "best_model": str(output_dir / "best_model"),
        "test_evaluated": False,
    }
    write_json(output_dir / "training_summary.json", result)
    return result
