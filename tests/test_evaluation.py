import unittest

import numpy as np

from rewilding_vision.evaluation import SegmentationAccumulator, bootstrap_confidence_interval


class EvaluationTest(unittest.TestCase):
    def test_global_iou_and_coverage_mae(self):
        meter = SegmentationAccumulator()
        meter.update(
            np.array([[1, 1], [0, 0]], dtype=np.uint8), np.array([[1, 0], [0, 0]], dtype=np.uint8)
        )
        meter.update(
            np.array([[1, 0], [0, 0]], dtype=np.uint8), np.array([[1, 1], [0, 0]], dtype=np.uint8)
        )
        result = meter.compute()
        self.assertAlmostEqual(result.vegetation_iou, 2 / 4)
        self.assertAlmostEqual(result.coverage_mae_pp, 25.0)
        self.assertEqual(result.evaluated_images, 2)

    def test_void_prediction_is_ignored(self):
        meter = SegmentationAccumulator()
        meter.update(np.array([[1, 1]], dtype=np.uint8), np.array([[1, 255]], dtype=np.uint8))
        self.assertEqual(meter.compute().vegetation_iou, 1.0)

    def test_bootstrap_is_deterministic(self):
        first = bootstrap_confidence_interval([1, 2, 3, 4], iterations=100)
        second = bootstrap_confidence_interval([1, 2, 3, 4], iterations=100)
        self.assertEqual(first, second)

    def test_accumulator_confidence_intervals(self):
        meter = SegmentationAccumulator()
        meter.update(np.array([[1, 0]], dtype=np.uint8), np.array([[1, 0]], dtype=np.uint8))
        meter.update(np.array([[0, 0]], dtype=np.uint8), np.array([[1, 0]], dtype=np.uint8))
        interval = meter.confidence_intervals(iterations=100)
        self.assertEqual(interval["confidence_level"], 0.95)
        self.assertEqual(len(interval["vegetation_iou"]), 2)


if __name__ == "__main__":
    unittest.main()
