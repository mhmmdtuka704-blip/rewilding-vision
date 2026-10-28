"""Release regression checks that need neither raw imagery nor downloaded models."""
import hashlib
import importlib.util
import io
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def setup_module_object():
    spec = importlib.util.spec_from_file_location("setup_model", ROOT / "scripts/setup_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verified_download_rejects_corrupt_content_without_overwrite(tmp_path):
    module = setup_module_object()
    destination = tmp_path / "weights.bin"
    destination.write_bytes(b"preserve original")
    with patch.object(module, "urlopen", return_value=io.BytesIO(b"corrupt")):
        with pytest.raises(ValueError, match="Checksum mismatch"):
            module.download_verified("https://example.invalid/file", destination, hashlib.sha256(b"expected").hexdigest())
    assert destination.read_bytes() == b"preserve original"


def test_verified_download_skips_valid_cache_without_network(tmp_path):
    module = setup_module_object()
    destination = tmp_path / "weights.bin"
    destination.write_bytes(b"valid")
    with patch.object(module, "urlopen", side_effect=AssertionError("Network must not be used")):
        assert module.download_verified("https://example.invalid/file", destination, hashlib.sha256(b"valid").hexdigest()) is False


def test_verified_download_writes_verified_content(tmp_path):
    module = setup_module_object()
    destination = tmp_path / "model" / "weights.bin"
    with patch.object(module, "urlopen", return_value=io.BytesIO(b"valid")):
        assert module.download_verified("https://example.invalid/file", destination, hashlib.sha256(b"valid").hexdigest()) is True
    assert destination.read_bytes() == b"valid"


def test_app_starts_without_dataset_logo_or_base_weights():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.button[0].disabled
    assert list(app.radio[0].options) == ["ML segmentation", "Binary ExG baseline"]


def test_dataset_split_and_hashes_match_recorded_test():
    import csv
    import json
    with (ROOT / "data/rugd_subset/manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = {split: {r["sequence"] for r in rows if r["split"] == split}
              for split in ("train", "validation", "test")}
    assert groups["train"].isdisjoint(groups["validation"] | groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert [sum(r["split"] == s for r in rows) for s in groups] == [160, 36, 48]
    metrics = json.loads((ROOT / "benchmarks/test_metrics.json").read_text())
    assert hashlib.sha256((ROOT / "data/rugd_subset/manifest.csv").read_bytes()).hexdigest() == metrics["manifest_sha256"]
