"""End-to-end before/after analysis with quality-gated registration."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from .change import calculate_change
from .config import load_yaml
from .exg import excess_green, segment_exg
from .registration import identity_registration, register_to_reference
from .reproducibility import write_json
from .visualization import colorize_change, horizontal_panel, vegetation_overlay


class ExGPredictor:
    def __init__(self, threshold=0.1):
        self.threshold = float(threshold)

    def predict(self, image_rgb):
        score = excess_green(image_rgb)
        mask = segment_exg(image_rgb, self.threshold)
        confidence = np.clip((score + 0.5) / 1.5, 0.0, 1.0).astype(np.float32)
        return mask, confidence


def analyze_pair(before_rgb, after_rgb, predictor, registration_config, assume_aligned=False,
                 before_valid=None, after_valid=None, geo_aligned=False):
    registration = (
        identity_registration(before_rgb, after_rgb)
        if assume_aligned or geo_aligned
        else register_to_reference(before_rgb, after_rgb, registration_config)
    )
    result = {
        "registration": registration.quality.to_dict(),
        "accepted": registration.quality.accepted,
        "alignment_method": ("CRS_reprojection" if geo_aligned else
                             "explicit_identity" if assume_aligned else "ORB_RANSAC_similarity"),
    }
    if not registration.quality.accepted:
        result["message"] = (
            "Pair rejected: change was not calculated. Retake photographs from a more consistent viewpoint."
        )
        return result, {}
    if before_valid is not None or after_valid is not None:
        import cv2
        valid = registration.overlap_mask.copy()
        if before_valid is not None:
            if before_valid.shape != before_rgb.shape[:2]:
                raise ValueError("Before validity shape mismatch")
            valid &= before_valid.astype(bool)
        if after_valid is not None:
            if after_valid.shape != after_rgb.shape[:2]:
                raise ValueError("After validity shape mismatch")
            warped_valid = cv2.warpPerspective(after_valid.astype(np.uint8), registration.homography,
                                               (before_rgb.shape[1], before_rgb.shape[0]), flags=cv2.INTER_NEAREST)
            valid &= warped_valid.astype(bool)
        registration.overlap_mask = valid
        fraction = float(valid.mean())
        accepted = fraction >= float(registration_config.get("minimum_overlap_ratio", .70))
        reason = ("CRS/transform alignment; source georeferencing accuracy is assumed" if geo_aligned
                  else registration.quality.reason)
        if not accepted:
            reason = f"Valid common footprint {fraction:.1%} is below the required overlap"
        registration.quality = replace(registration.quality, accepted=accepted,
                                        reason=reason, overlap_ratio=fraction)
        result.update(accepted=accepted, registration=registration.quality.to_dict())
        if not accepted:
            result["message"] = "Insufficient valid shared area. No change was calculated."
            return result, {}
    before_mask, before_probability = predictor.predict(before_rgb)
    if np.array_equal(before_rgb, after_rgb):
        after_mask, after_probability = before_mask.copy(), before_probability.copy()
    else:
        # Avoid black warp borders changing the neural model's image context.
        inference_after = registration.aligned_image.copy()
        inference_after[~registration.overlap_mask] = before_rgb[~registration.overlap_mask]
        after_mask, after_probability = predictor.predict(inference_after)
    summary, change_map = calculate_change(before_mask, after_mask, registration.overlap_mask)
    result["change"] = summary.to_dict()
    result["comparison_area"] = "Shared valid overlap in the before-image pixel grid"
    result["relative_change_note"] = (
        "Undefined because before vegetation coverage is zero"
        if summary.relative_change_percent is None
        else "100 * (after vegetation pixels - before vegetation pixels) / before vegetation pixels"
    )
    result["transform_after_to_before"] = registration.homography.tolist()
    result["mean_model_vegetation_score"] = {
        "before": float(before_probability[registration.overlap_mask].mean()),
        "after": float(after_probability[registration.overlap_mask].mean()),
    }
    artifacts = {
        "aligned_after": registration.aligned_image,
        "before_mask": before_mask,
        "after_mask": after_mask,
        "before_overlay": vegetation_overlay(before_rgb, before_mask),
        "after_overlay": vegetation_overlay(registration.aligned_image, after_mask),
        "change_map": colorize_change(change_map),
        "change_classes": change_map,
        "valid_overlap": registration.overlap_mask.astype(np.uint8),
    }
    for name in ("before_overlay", "after_overlay"):
        artifacts[name][~registration.overlap_mask] = (100, 100, 100)
    artifacts["panel"] = horizontal_panel(
        [artifacts["before_overlay"], artifacts["after_overlay"], artifacts["change_map"]]
    )
    return result, artifacts


def save_analysis(output_dir: str | Path, result: dict, artifacts: dict):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "result.json", result)
    for name, array in artifacts.items():
        rendered = array * 255 if name in {"before_mask", "after_mask", "valid_overlap"} else array
        Image.fromarray(rendered.astype(np.uint8)).save(destination / f"{name}.png")


def pair_analysis_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--config", default="configs/week3.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--assume-aligned", action="store_true")
    parser.add_argument("--output", default="artifacts/pair_analysis")
    args = parser.parse_args()
    config = load_yaml(args.config)
    before = np.asarray(Image.open(args.before).convert("RGB"))
    after = np.asarray(Image.open(args.after).convert("RGB"))
    if args.checkpoint:
        from .segformer import SegFormerPredictor

        predictor = SegFormerPredictor(args.checkpoint)
        method = "segformer_b0"
    else:
        from .segformer import SegFormerPredictor

        predictor = SegFormerPredictor()
        method = "segformer_b0_ade20k"
    result, artifacts = analyze_pair(
        before, after, predictor, config["registration"], args.assume_aligned
    )
    result["method"] = method
    save_analysis(args.output, result, artifacts)
    print(json.dumps(result, indent=2))
