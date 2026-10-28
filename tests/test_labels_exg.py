import unittest

import numpy as np

from rewilding_vision.exg import segment_exg, select_threshold
from rewilding_vision.rugd_labels import RUGD_CLASSES, color_mask_to_binary


class LabelsAndBaselineTest(unittest.TestCase):
    def test_rugd_mapping(self):
        mask = np.array(
            [[(0, 102, 0), (0, 255, 0), (255, 153, 204), (108, 64, 20), (0, 0, 0)]], dtype=np.uint8
        )
        expected = np.array([[1, 1, 1, 0, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(color_mask_to_binary(mask), expected)

    def test_unknown_color_fails(self):
        with self.assertRaises(ValueError):
            color_mask_to_binary(np.array([[(1, 2, 3)]], dtype=np.uint8))

    def test_official_rock_class_names_and_colors(self):
        labels = {item.class_id: (item.name, item.rgb) for item in RUGD_CLASSES}
        self.assertEqual(labels[14], ("rock-bed", (102, 102, 0)))
        self.assertEqual(labels[21], ("rock", (153, 204, 255)))

    def test_green_pixel_is_detected(self):
        image = np.array([[(10, 200, 10), (200, 10, 10)]], dtype=np.uint8)
        np.testing.assert_array_equal(segment_exg(image, 0.1), np.array([[1, 0]], dtype=np.uint8))

    def test_threshold_selected_from_validation_samples(self):
        image = np.array([[(10, 200, 10), (200, 10, 10)]], dtype=np.uint8)
        target = np.array([[1, 0]], dtype=np.uint8)
        threshold, score = select_threshold([(image, target)], np.array([0.1, 0.9]))
        self.assertEqual(threshold, 0.1)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
