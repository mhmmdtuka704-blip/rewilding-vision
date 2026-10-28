import unittest

import numpy as np

from rewilding_vision.change import calculate_change


class ChangeTest(unittest.TestCase):
    def test_change_categories_and_summary(self):
        before = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        after = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        summary, change_map = calculate_change(before, after)
        self.assertEqual(summary.before_coverage_percent, 50.0)
        self.assertEqual(summary.after_coverage_percent, 50.0)
        self.assertEqual(summary.gain_percent, 25.0)
        self.assertEqual(summary.loss_percent, 25.0)
        np.testing.assert_array_equal(change_map, np.array([[1, 2], [3, 0]], dtype=np.uint8))

    def test_overlap_is_the_denominator(self):
        before = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        after = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        overlap = np.array([[True, False], [False, False]])
        summary, change_map = calculate_change(before, after, overlap)
        self.assertEqual(summary.change_percentage_points, 100.0)
        self.assertEqual(summary.evaluated_pixels, 1)
        self.assertEqual(int((change_map == 255).sum()), 3)


if __name__ == "__main__":
    unittest.main()
