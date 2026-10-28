import unittest

import numpy as np

from rewilding_vision.segformer import letterbox_rgb


class SegformerHelperTest(unittest.TestCase):
    def test_letterbox_preserves_aspect_ratio(self):
        image = np.zeros((50, 100, 3), dtype=np.uint8)
        boxed, (left, top, width, height) = letterbox_rgb(image, 64)
        self.assertEqual(boxed.shape, (64, 64, 3))
        self.assertEqual((width, height), (64, 32))
        self.assertEqual((left, top), (0, 16))


if __name__ == "__main__":
    unittest.main()
