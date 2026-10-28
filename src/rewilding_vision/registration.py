"""Feature-based image registration with explicit quality rejection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RegistrationQuality:
    accepted: bool
    reason: str
    detected_matches: int
    inliers: int
    inlier_ratio: float
    reprojection_rmse_px: float | None
    overlap_ratio: float
    spatial_coverage: float = 0.0
    estimated_scale: float | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class RegistrationResult:
    aligned_image: np.ndarray | None
    overlap_mask: np.ndarray | None
    homography: np.ndarray | None
    quality: RegistrationQuality


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install prototype dependencies: pip install -e '.[prototype]'") from exc
    return cv2


def register_to_reference(
    reference_rgb: np.ndarray, moving_rgb: np.ndarray, config: dict
) -> RegistrationResult:
    """ORB + RANSAC similarity transform; reject weak or implausible geometry."""
    cv2 = _cv2()
    if (reference_rgb.ndim != 3 or moving_rgb.ndim != 3
            or reference_rgb.shape[2] != 3 or moving_rgb.shape[2] != 3):
        raise ValueError("Registration expects two HxWx3 RGB images")
    reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    moving_gray = cv2.cvtColor(moving_rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.ORB_create(nfeatures=int(config["max_features"]))
    reference_points, reference_desc = detector.detectAndCompute(reference_gray, None)
    moving_points, moving_desc = detector.detectAndCompute(moving_gray, None)
    if reference_desc is None or moving_desc is None:
        quality = RegistrationQuality(False, "Insufficient visual features", 0, 0, 0.0, None, 0.0)
        return RegistrationResult(None, None, None, quality)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(moving_desc, reference_desc, k=2)
    good = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < float(config["ratio_test"]) * second.distance
    ]
    # Prevent duplicate reference features from inflating the inlier count.
    unique = {}
    for match in sorted(good, key=lambda m: m.distance):
        unique.setdefault(match.trainIdx, match)
    good = list(unique.values())
    minimum_matches = int(config["minimum_matches"])
    if len(good) < minimum_matches:
        quality = RegistrationQuality(
            False,
            f"Only {len(good)} reliable matches; need {minimum_matches}",
            len(good),
            0,
            0.0,
            None,
            0.0,
        )
        return RegistrationResult(None, None, None, quality)

    source = np.float32([moving_points[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
    destination = np.float32([reference_points[match.trainIdx].pt for match in good]).reshape(
        -1, 1, 2
    )
    affine, status = cv2.estimateAffinePartial2D(
        source, destination, method=cv2.RANSAC,
        ransacReprojThreshold=float(config["ransac_threshold_px"]),
        maxIters=3000, confidence=0.995, refineIters=10,
    )
    if affine is None or status is None or not np.isfinite(affine).all():
        quality = RegistrationQuality(
            False, "Similarity transform estimation failed", len(good), 0, 0.0, None, 0.0
        )
        return RegistrationResult(None, None, None, quality)
    homography = np.vstack([affine, [0.0, 0.0, 1.0]])
    inlier_mask = status.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    inlier_ratio = float(inliers / len(good))
    projected = cv2.perspectiveTransform(source[inlier_mask], homography)
    errors = np.linalg.norm(
        projected.reshape(-1, 2) - destination[inlier_mask].reshape(-1, 2), axis=1
    )
    rmse = float(np.sqrt(np.mean(errors**2))) if errors.size else None
    height, width = reference_rgb.shape[:2]
    aligned = cv2.warpPerspective(moving_rgb, homography, (width, height), flags=cv2.INTER_LINEAR)
    moving_valid = np.ones(moving_rgb.shape[:2], dtype=np.uint8)
    # Exclude partially sampled edge pixels and a two-pixel interpolation rim.
    overlap = cv2.warpPerspective(
        moving_valid.astype(np.float32), homography, (width, height), flags=cv2.INTER_LINEAR
    ) >= 0.999
    overlap = cv2.erode(overlap.astype(np.uint8), np.ones((5, 5), np.uint8),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    overlap_ratio = float(overlap.mean())

    failures = []
    scale = float(np.hypot(affine[0, 0], affine[1, 0]))
    rotation = abs(float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0]))))
    hull = cv2.convexHull(destination[inlier_mask].reshape(-1, 2))
    spatial_coverage = float(cv2.contourArea(hull) / (width * height))
    if not float(config.get("minimum_scale", 0.75)) <= scale <= float(config.get("maximum_scale", 1.33)):
        failures.append(f"scale {scale:.2f} outside supported modest viewpoint change")
    if rotation > float(config.get("maximum_rotation_degrees", 20)):
        failures.append(f"rotation {rotation:.1f} degrees too large")
    if spatial_coverage < float(config.get("minimum_spatial_coverage", 0.04)):
        failures.append("matched features are too concentrated in a small area")
    if inliers < int(config["minimum_inliers"]):
        failures.append(f"{inliers} inliers below minimum")
    if inlier_ratio < float(config["minimum_inlier_ratio"]):
        failures.append(f"inlier ratio {inlier_ratio:.2f} below minimum")
    if rmse is None or rmse > float(config["maximum_reprojection_rmse_px"]):
        failures.append("reprojection error too high")
    if overlap_ratio < float(config["minimum_overlap_ratio"]):
        failures.append(f"overlap {overlap_ratio:.2f} below minimum")
    accepted = not failures
    quality = RegistrationQuality(
        accepted,
        "Accepted" if accepted else "; ".join(failures),
        len(good),
        inliers,
        inlier_ratio,
        rmse,
        overlap_ratio,
        spatial_coverage,
        scale,
    )
    return RegistrationResult(
        aligned if accepted else None, overlap if accepted else None, homography, quality
    )


def identity_registration(reference_rgb: np.ndarray, moving_rgb: np.ndarray) -> RegistrationResult:
    """Explicit opt-in for photographs guaranteed to share exact coordinates."""
    if reference_rgb.shape != moving_rgb.shape:
        raise ValueError("Assume-aligned mode requires identical image dimensions")
    overlap = np.ones(reference_rgb.shape[:2], dtype=bool)
    quality = RegistrationQuality(
        True, "Alignment explicitly supplied by capture protocol", 0, 0, 1.0, 0.0, 1.0
    )
    return RegistrationResult(moving_rgb.copy(), overlap, np.eye(3), quality)
