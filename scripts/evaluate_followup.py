"""Evaluate the validation-selected deployed model on the frozen test sequences."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rewilding_vision.rugd_labels import color_mask_to_binary
from rewilding_vision.segformer import SegFormerPredictor
from rewilding_vision.visualization import vegetation_overlay


def evaluate(predictor, rows, brightness=1.0):
    counts = np.zeros(4, np.int64)
    errors, elapsed, records = [], [], []
    for index, row in enumerate(rows):
        image = np.asarray(Image.open(ROOT / row["image_path"]).convert("RGB"))
        image = np.clip(image.astype(np.float32)*brightness, 0, 255).astype(np.uint8)
        truth = color_mask_to_binary(np.asarray(Image.open(ROOT / row["mask_path"]).convert("RGB")))
        start = time.perf_counter()
        pred, _ = predictor.predict(image)
        elapsed.append(time.perf_counter()-start)
        valid = truth != 255
        p, y = pred[valid] == 1, truth[valid] == 1
        per_counts = np.array([(p & y).sum(), (p & ~y).sum(), (~p & y).sum(), (~p & ~y).sum()], np.int64)
        counts += per_counts
        error = abs(float(p.mean()-y.mean())*100)
        errors.append(error)
        records.append({"image": row["image_path"], "sequence": row["sequence"],
            "counts_tp_fp_fn_tn": per_counts.tolist(), "coverage_error_pp": error,
            "predicted_coverage_percent": float(p.mean()*100), "labeled_coverage_percent": float(y.mean()*100)})
        if brightness == 1 and index == 0:
            out = ROOT / "verification/followup"
            Image.fromarray(image).save(out / "test_image.png")
            Image.fromarray(vegetation_overlay(image, pred)).save(out / "test_overlay.png")
            Image.fromarray(np.where(truth == 255, 128, truth*255).astype(np.uint8)).save(out / "test_ground_truth.png")
        if (index+1) % 12 == 0:
            print(f"Evaluated {index+1}/{len(rows)} test images (brightness {brightness})", flush=True)
    tp, fp, fn, tn = (int(n) for n in counts)
    return {"images": len(rows), "vegetation_iou": tp/(tp+fp+fn), "dice_f1": 2*tp/(2*tp+fp+fn),
            "precision": tp/(tp+fp), "recall": tp/(tp+fn), "pixel_accuracy": (tp+tn)/(tp+fp+fn+tn),
            "coverage_mae_pp": float(np.mean(errors)), "valid_pixels": int(counts.sum()),
            "confusion_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "mean_inference_seconds": float(np.mean(elapsed)), "per_image": records}


def main():
    output = ROOT / "verification/followup"
    output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data/rugd_subset/manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    test = [r for r in rows if r["split"] == "test"]
    train_sequences = {r["sequence"] for r in rows if r["split"] == "train"}
    val_sequences = {r["sequence"] for r in rows if r["split"] == "validation"}
    test_sequences = {r["sequence"] for r in test}
    assert test_sequences.isdisjoint(train_sequences | val_sequences)
    assert len(test) == 48
    predictor = SegFormerPredictor()
    result = evaluate(predictor, test)
    result.update(dataset="RUGD bounded sequence-disjoint subset", split="test", date="2026-10-25",
                  test_sequences=sorted(test_sequences), model_training=predictor.adapter_metadata,
                  model="SegFormer-B0 with residual binary head" if predictor.adapter_metadata else "SegFormer-B0 ADE20K",
                  manifest_sha256=hashlib.sha256((ROOT / "data/rugd_subset/manifest.csv").read_bytes()).hexdigest(),
                  test_used_for_model_selection=False,
                  limitations=["Small outdoor ground-view subset; not TOD field or aerial validation",
                               "Adjacent frames within each test sequence are correlated",
                               "Cannot guarantee a fixed accuracy on arbitrary imagery"])
    (output / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    dark = evaluate(predictor, test[::4], .65)
    (output / "darkening_metrics.json").write_text(json.dumps(dark, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"per_image", "model_training"}}, indent=2))


if __name__ == "__main__":
    main()
