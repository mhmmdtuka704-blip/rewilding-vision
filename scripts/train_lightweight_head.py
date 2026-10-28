"""Bounded CPU transfer learning. Test images/labels never enter training/selection."""
import csv
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rewilding_vision.rugd_labels import color_mask_to_binary
from rewilding_vision.segformer import SegFormerPredictor, letterbox_rgb

SEED = 251025


def extract(predictor, row, augmentation=False):
    image = Image.open(ROOT / row["image_path"]).convert("RGB")
    if augmentation:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(.58, 1.25))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(.85, 1.15))
    array = np.asarray(image)
    boxed, content = letterbox_rgb(array, 512)
    truth = color_mask_to_binary(np.asarray(Image.open(ROOT / row["mask_path"]).convert("RGB")))
    left, top, width, height = content
    labels = Image.new("L", (512, 512), 255)
    labels.paste(Image.fromarray(truth).resize((width, height), Image.Resampling.NEAREST), (left, top))
    normalized = (boxed.astype(np.float32) / 255 - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    pixels = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)
    captured = []
    handle = predictor.model.decode_head.classifier.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().cpu().numpy().astype(np.float16)))
    try:
        with torch.inference_mode():
            logits = predictor.model(pixel_values=pixels).logits
            positive = predictor.vegetation_ids
            negative = [i for i in range(logits.shape[1]) if i not in positive]
            grouped = torch.stack([torch.logsumexp(logits[:, negative], 1),
                                   torch.logsumexp(logits[:, positive], 1)], 1)
            base = grouped.cpu().numpy().astype(np.float16)[0]
    finally:
        handle.remove()
    small_labels = np.asarray(labels.resize((base.shape[2], base.shape[1]), Image.Resampling.NEAREST)).copy()
    return {"features": captured[0][0], "base": base, "labels": small_labels,
            "truth": truth if not augmentation else None, "content": content,
            "image": row["image_path"]}


def score(head, cached):
    tp = fp = fn = tn = 0
    errors = []
    with torch.inference_mode():
        for item in cached:
            logits = (torch.from_numpy(item["base"]).float().unsqueeze(0) +
                      head(torch.from_numpy(item["features"]).float().unsqueeze(0)))
            probability = F.interpolate(logits.softmax(1)[:, 1:2], (512, 512), mode="bilinear", align_corners=False)[0, 0].numpy()
            left, top, width, height = item["content"]
            truth = item["truth"]
            p = np.asarray(Image.fromarray(probability[top:top+height, left:left+width]).resize(
                (truth.shape[1], truth.shape[0]), Image.Resampling.BILINEAR)) >= .5
            valid = truth != 255
            p, y = p[valid], truth[valid] == 1
            tp += int((p & y).sum()); fp += int((p & ~y).sum())
            fn += int((~p & y).sum()); tn += int((~p & ~y).sum())
            errors.append(abs(float(p.mean()-y.mean())*100))
    return {"vegetation_iou": tp / (tp+fp+fn), "dice_f1": 2*tp/(2*tp+fp+fn),
            "pixel_accuracy": (tp+tn)/(tp+tn+fp+fn), "coverage_mae_pp": float(np.mean(errors)),
            "images": len(cached)}


def main():
    start = time.perf_counter()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
    with (ROOT / "data/rugd_subset/manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    assert set(r["sequence"] for r in train_rows).isdisjoint(r["sequence"] for r in val_rows)
    predictor = SegFormerPredictor(enable_adapter=False)
    predictor.model.requires_grad_(False).eval()
    validation = [extract(predictor, row) for row in val_rows]
    head = torch.nn.Conv2d(256, 2, 1)
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    baseline = score(head, validation)
    print("Frozen baseline validation:", json.dumps(baseline), flush=True)
    training = []
    for index, row in enumerate(train_rows):
        training.append(extract(predictor, row))
        if index % 2 == 0:
            training.append(extract(predictor, row, augmentation=True))
        if (index+1) % 20 == 0:
            print(f"Cached {index+1}/{len(train_rows)} training photos", flush=True)
    del predictor
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002, weight_decay=.01)
    best = baseline["vegetation_iou"]
    best_state = None
    history = []
    run = ROOT / "training/head_run"
    run.mkdir(parents=True, exist_ok=True)
    for epoch in range(5):
        head.train(); random.shuffle(training); losses = []
        for i in range(0, len(training), 4):
            batch = training[i:i+4]
            features = torch.from_numpy(np.stack([b["features"] for b in batch])).float()
            base = torch.from_numpy(np.stack([b["base"] for b in batch])).float()
            labels = torch.from_numpy(np.stack([b["labels"] for b in batch])).long()
            if random.random() < .5:
                features, base, labels = features.flip(-1), base.flip(-1), labels.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = base + head(features)
            loss = F.cross_entropy(logits, labels, ignore_index=255)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        head.eval()
        validation_score = score(head, validation)
        record = {"epoch": epoch+1, "train_loss": float(np.mean(losses)), "validation": validation_score}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_score["vegetation_iou"] > best + .0001:
            best = validation_score["vegetation_iou"]
            best_state = {k: v.detach().clone().contiguous() for k, v in head.state_dict().items()}
        (run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metadata = {"method": "Frozen pretrained SegFormer-B0 plus learned residual binary segmentation head",
                "seed": SEED, "train_images": len(train_rows), "validation_images": len(val_rows),
                "augmented_training_views": len(training), "trainable_parameters": sum(p.numel() for p in head.parameters()),
                "epochs": len(history), "baseline_validation": baseline,
                "best_validation_iou": best, "deployment_approved": best_state is not None,
                "test_used_for_selection": False, "elapsed_seconds": time.perf_counter()-start,
                "dataset_manifest": "data/rugd_subset/manifest.csv", "input_size": 512,
                "limits": "Small domain adaptation; no guarantee on other cameras or scenes"}
    if best_state is not None:
        destination = ROOT / "model/rugd_adapter"
        destination.mkdir(parents=True, exist_ok=True)
        save_file(best_state, str(destination / "head.safetensors"))
        pending = destination / "metadata.pending"
        pending.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        pending.replace(destination / "metadata.json")
    (run / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
