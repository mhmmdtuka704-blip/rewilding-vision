import unittest

import numpy as np

from rewilding_vision.pair_analysis import ExGPredictor, analyze_pair


class PairAnalysisTest(unittest.TestCase):
    def test_assumed_aligned_pair_completes(self):
        before = np.zeros((8, 8, 3), dtype=np.uint8)
        after = before.copy()
        before[:, :4] = (120, 60, 30)
        after[:, :4] = (20, 180, 20)
        result, artifacts = analyze_pair(before, after, ExGPredictor(0.1), {}, assume_aligned=True)
        self.assertTrue(result["accepted"])
        self.assertGreater(result["change"]["change_percentage_points"], 0)
        self.assertIn("panel", artifacts)


if __name__ == "__main__":
    unittest.main()
