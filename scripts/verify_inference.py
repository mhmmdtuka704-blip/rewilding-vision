"""Milestone 1: real, offline neural inference on an existing labeled photo."""
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rewilding_vision.segformer import SegFormerPredictor
from rewilding_vision.visualization import vegetation_overlay


if __name__ == "__main__":
    start = time.perf_counter()
    predictor = SegFormerPredictor()
    load_seconds = time.perf_counter() - start
    photo = next((ROOT / "data/rugd_subset" / "images").glob("*.png"))
    image = np.asarray(Image.open(photo).convert("RGB"))
    start = time.perf_counter()
    mask, probability = predictor.predict(image)
    inference_seconds = time.perf_counter() - start
    assert mask.shape == image.shape[:2]
    assert np.isfinite(probability).all() and 0 <= probability.min() <= probability.max() <= 1.00001
    assert set(np.unique(mask)) == {0, 1}, "Expected both vegetation and background on sample"
    destination = ROOT / "verification" / "milestone1"
    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask * 255).save(destination / "mask.png")
    Image.fromarray(vegetation_overlay(image, mask)).save(destination / "overlay.png")
    result = {"passed": True, "photo": str(photo.relative_to(ROOT)),
              "model_parameters": sum(p.numel() for p in predictor.model.parameters()),
              "trained_classes": predictor.vegetation_labels,
              "load_seconds": load_seconds, "inference_seconds": inference_seconds,
              "coverage_percent": float(mask.mean() * 100), "device": predictor.device}
    (destination / "inference.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
