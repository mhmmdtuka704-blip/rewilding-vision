"""Milestone 2: run real ML against known camera perturbations, record errors."""
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rewilding_vision.config import load_yaml
from rewilding_vision.pair_analysis import analyze_pair, save_analysis
from rewilding_vision.segformer import SegFormerPredictor


def main():
    image = np.asarray(Image.open(ROOT / "data/rugd_subset/images/creek_00001.png").convert("RGB"))
    h, w = image.shape[:2]
    cfg = load_yaml(ROOT / "configs/week3.yaml")["registration"]
    predictor = SegFormerPredictor()
    destination = ROOT / "verification" / "alignment"
    destination.mkdir(parents=True, exist_ok=True)
    output = {}
    for name, angle, scale, dx, dy in [
        ("identical", 0, 1, 0, 0), ("translation", 0, 1, 25, -18),
        ("rotation_scale_translation", 4, 1.04, 16, -10),
    ]:
        forward = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        forward[:, 2] += (dx, dy)
        moving = cv2.warpAffine(image, forward, (w, h))
        start = time.perf_counter()
        result, artifacts = analyze_pair(image, moving, predictor, cfg)
        assert result["accepted"], (name, result)
        points = np.float32([[[w * .2, h * .2]], [[w * .8, h * .2]],
                             [[w * .8, h * .8]], [[w * .2, h * .8]]])
        restored = cv2.perspectiveTransform(cv2.transform(points, forward),
                                            np.asarray(result["transform_after_to_before"]))
        error = float(np.linalg.norm(restored - points, axis=2).max())
        change = result["change"]
        result["known_transform_max_error_px"] = error
        result["elapsed_seconds"] = time.perf_counter() - start
        result["false_changed_area_percent"] = change["gain_percent"] + change["loss_percent"]
        assert error < 2, (name, error)
        assert abs(change["change_percentage_points"]) < 2, (name, change)
        if name == "identical":
            assert change["gain_percent"] == change["loss_percent"] == 0
        save_analysis(destination / name, result, artifacts)
        output[name] = result
        print(name, "PASS", "error_px", round(error, 3), "net_pp",
              round(change["change_percentage_points"], 3), flush=True)
    (destination / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
