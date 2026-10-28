import importlib.util
import unittest

import numpy as np

from rewilding_vision.registration import identity_registration, register_to_reference

CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None


class RegistrationTest(unittest.TestCase):
    def test_identity_requires_same_dimensions(self):
        with self.assertRaises(ValueError):
            identity_registration(
                np.zeros((3, 3, 3), dtype=np.uint8), np.zeros((4, 3, 3), dtype=np.uint8)
            )

    @unittest.skipUnless(CV2_AVAILABLE, "OpenCV prototype dependency not installed")
    def test_blank_pair_is_rejected(self):
        config = {
            "max_features": 1000,
            "ratio_test": 0.75,
            "ransac_threshold_px": 4.0,
            "minimum_matches": 10,
            "minimum_inliers": 6,
            "minimum_inlier_ratio": 0.3,
            "maximum_reprojection_rmse_px": 5.0,
            "minimum_overlap_ratio": 0.7,
        }
        blank = np.zeros((128, 128, 3), dtype=np.uint8)
        result = register_to_reference(blank, blank, config)
        self.assertFalse(result.quality.accepted)

    @unittest.skipUnless(CV2_AVAILABLE, "OpenCV prototype dependency not installed")
    def test_known_translation_is_accepted(self):
        import cv2

        rng = np.random.default_rng(8)
        reference = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        matrix = np.float32([[1, 0, 5], [0, 1, 7]])
        moving = cv2.warpAffine(reference, matrix, (256, 256))
        config = {
            "max_features": 5000,
            "ratio_test": 0.8,
            "ransac_threshold_px": 4.0,
            "minimum_matches": 20,
            "minimum_inliers": 12,
            "minimum_inlier_ratio": 0.35,
            "maximum_reprojection_rmse_px": 5.0,
            "minimum_overlap_ratio": 0.7,
        }
        result = register_to_reference(reference, moving, config)
        self.assertTrue(result.quality.accepted, result.quality.reason)
        self.assertGreater(result.quality.overlap_ratio, 0.9)


if __name__ == "__main__":
    unittest.main()
