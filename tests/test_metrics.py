import unittest

import numpy as np

from rewilding_vision.metrics import change_absolute_error_pp, vegetation_coverage, vegetation_iou


class MetricsTest(unittest.TestCase):
    def test_iou_and_ignore(self):
        pred = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        truth = np.array([[1, 0], [255, 1]], dtype=np.uint8)
        self.assertAlmostEqual(vegetation_iou(pred, truth), 2 / 3)

    def test_coverage_is_percentage(self):
        self.assertAlmostEqual(
            vegetation_coverage(np.array([[1, 0, 1, 255]], dtype=np.uint8)), 200 / 3
        )

    def test_change_error(self):
        before_truth = np.array([[1, 0, 0, 0]], dtype=np.uint8)
        after_truth = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        before_pred = np.array([[1, 0, 0, 0]], dtype=np.uint8)
        after_pred = np.array([[1, 1, 1, 0]], dtype=np.uint8)
        self.assertEqual(
            change_absolute_error_pp(before_pred, after_pred, before_truth, after_truth), 25.0
        )

    def test_change_error_excludes_void_from_predictions(self):
        before_truth = np.array([[1, 0, 255, 255]], dtype=np.uint8)
        after_truth = np.array([[1, 1, 255, 255]], dtype=np.uint8)
        before_pred = np.array([[1, 0, 1, 1]], dtype=np.uint8)
        after_pred = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        self.assertEqual(
            change_absolute_error_pp(before_pred, after_pred, before_truth, after_truth), 0.0
        )

    def test_change_error_uses_shared_valid_footprint(self):
        before_truth = np.array([[1, 255, 0]], dtype=np.uint8)
        after_truth = np.array([[1, 1, 255]], dtype=np.uint8)
        before_pred = np.array([[1, 0, 1]], dtype=np.uint8)
        after_pred = np.array([[1, 1, 0]], dtype=np.uint8)
        self.assertEqual(
            change_absolute_error_pp(before_pred, after_pred, before_truth, after_truth), 0.0
        )


if __name__ == "__main__":
    unittest.main()
