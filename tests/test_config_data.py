import csv
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rewilding_vision.config import deep_merge
from rewilding_vision.training_data import paired_transform, read_manifest


class ConfigAndDataTest(unittest.TestCase):
    def test_deep_merge_does_not_mutate_source(self):
        base = {"model": {"size": 512, "name": "b0"}, "seed": 1}
        merged = deep_merge(base, {"model": {"size": 256}})
        self.assertEqual(merged["model"], {"size": 256, "name": "b0"})
        self.assertEqual(base["model"]["size"], 512)

    def test_manifest_split_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["sequence", "split", "image_path", "mask_path"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sequence": "a",
                        "split": "train",
                        "image_path": "a.png",
                        "mask_path": "a_mask.png",
                    }
                )
                writer.writerow(
                    {
                        "sequence": "b",
                        "split": "test",
                        "image_path": "b.png",
                        "mask_path": "b_mask.png",
                    }
                )
            rows = read_manifest(path, "test")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].sequence, "b")

    def test_mask_resize_keeps_discrete_labels(self):
        image = Image.fromarray(np.zeros((2, 4, 3), dtype=np.uint8))
        mask = Image.fromarray(np.array([[0, 1, 255, 1], [1, 0, 255, 0]], dtype=np.uint8))
        _, transformed = paired_transform(image, mask, 8, False, random.Random(1))
        self.assertEqual(set(np.unique(np.asarray(transformed))), {0, 1, 255})


if __name__ == "__main__":
    unittest.main()
