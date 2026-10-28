"""Regression tests for shared-area math and rejected/perturbed camera pairs."""
from pathlib import Path
import cv2
import numpy as np

from rewilding_vision.change import calculate_change
from rewilding_vision.config import load_yaml
from rewilding_vision.pair_analysis import analyze_pair
from rewilding_vision.registration import register_to_reference

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_yaml(ROOT / "configs/week3.yaml")["registration"]


def test_relative_change_and_excluded_pixels():
    before = np.array([[1, 0, 0], [0, 1, 1]], np.uint8)
    after = np.array([[1, 1, 1], [0, 0, 0]], np.uint8)
    valid = np.array([[1, 1, 1], [1, 0, 0]], np.uint8)
    summary, classes = calculate_change(before, after, valid)
    assert summary.before_coverage_percent == 25
    assert summary.after_coverage_percent == 75
    assert summary.gain_percent == 50
    assert summary.loss_percent == 0
    assert summary.change_percentage_points == 50
    assert summary.relative_change_percent == 200
    assert summary.evaluated_pixels == 4
    assert np.all(classes[1, 1:] == 255)


def test_zero_baseline_relative_change_is_undefined():
    summary, _ = calculate_change(np.zeros((3, 3)), np.ones((3, 3)))
    assert summary.relative_change_percent is None
    assert summary.change_percentage_points == 100


def test_rejected_alignment_never_invokes_predictor_or_reports_change():
    class MustNotRun:
        def predict(self, image):
            raise AssertionError("No ML or measurements for rejected pairs")
    blank = np.zeros((300, 400, 3), np.uint8)
    result, artifacts = analyze_pair(blank, blank, MustNotRun(), CONFIG)
    assert not result["accepted"]
    assert "change" not in result
    assert artifacts == {}


def test_combined_translation_rotation_scale_recovers_geometry():
    rng = np.random.default_rng(35)
    image = rng.integers(0, 256, (500, 700, 3), dtype=np.uint8)
    forward = cv2.getRotationMatrix2D((350, 250), 4, 1.04)
    forward[:, 2] += (17, -12)
    moving = cv2.warpAffine(image, forward, (700, 500))
    result = register_to_reference(image, moving, CONFIG)
    assert result.quality.accepted, result.quality.reason
    points = np.float32([[[100, 100]], [[550, 100]], [[550, 400]], [[100, 400]]])
    recovered = cv2.perspectiveTransform(
        cv2.transform(points, forward), result.homography
    )
    assert np.linalg.norm(points - recovered, axis=2).max() < 2
    assert 0.7 < result.quality.overlap_ratio < 1


def test_unrelated_photographs_rejected():
    rng = np.random.default_rng(12)
    a = rng.integers(0, 256, (400, 500, 3), dtype=np.uint8)
    b = rng.integers(0, 256, a.shape, dtype=np.uint8)
    result = register_to_reference(a, b, CONFIG)
    assert not result.quality.accepted
