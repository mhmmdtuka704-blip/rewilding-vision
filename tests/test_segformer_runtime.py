import importlib.util
import tempfile
import unittest
from pathlib import Path

TRANSFORMERS_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Training dependencies not installed")
class SegformerRuntimeTest(unittest.TestCase):
    def test_classification_encoder_initializes_binary_segmentation(self):
        from transformers import SegformerConfig, SegformerForImageClassification

        from rewilding_vision.segformer import create_segformer

        config = SegformerConfig(
            num_labels=3,
            depths=[1, 1, 1, 1],
            hidden_sizes=[8, 16, 32, 64],
            decoder_hidden_size=32,
            num_attention_heads=[1, 1, 2, 4],
            sr_ratios=[8, 4, 2, 1],
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "classifier"
            SegformerForImageClassification(config).save_pretrained(checkpoint)
            model = create_segformer(checkpoint, local_files_only=True)
            self.assertEqual(model.config.num_labels, 2)
            self.assertEqual(model.config.id2label[1], "vegetation")


if __name__ == "__main__":
    unittest.main()
