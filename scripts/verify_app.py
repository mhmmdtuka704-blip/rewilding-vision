"""Exercise the app on synthetic software fixtures; --ml runs actual neural inference."""
import argparse
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class Upload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


def synthetic_rgb():
    rng = np.random.default_rng(104)
    image = rng.integers(0, 35, (384, 512, 3), dtype=np.uint8)
    image += np.array([115, 65, 25], dtype=np.uint8)
    image[90:240, 130:340] = [30, 180, 40]
    return image


def png(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, "PNG")
    return buffer.getvalue()


def run_smoke(use_ml=False):
    if use_ml and not (ROOT / "model/ade20k/model.safetensors").is_file():
        raise RuntimeError("Run python scripts/setup_model.py before --ml")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception and app.button[0].disabled
    assert len(app.radio) == 1 and not app.selectbox
    image = synthetic_rgb()
    photo = png(image)
    uploads = {"before": Upload(photo, "synthetic_before.png"),
               "after": Upload(photo, "synthetic_after.png")}
    with patch("streamlit.file_uploader", side_effect=lambda *args, **kw: uploads[kw["key"]]):
        app.run()
        if not use_ml:
            app.radio[0].set_value("Binary ExG baseline").run()
        app.button[0].click().run()
        assert not app.exception and not app.error
        assert len(app.metric) == 6 and len(app.get("download_button")) == 3
        result = app.session_state["analysis"][1]
        assert result["accepted"]
        assert result["change"]["gain_percent"] == result["change"]["loss_percent"] == 0
        assert result["method"] == ("segformer_b0_rugd_adapted" if use_ml else "binary_exg_baseline")
        assert "Binary masks" in [tab.label for tab in app.tabs]
        app.run()
        assert len(app.metric) == 6
        uploads["after"] = Upload(photo + b"new_file_bytes", "changed.png")
        app.run()
        assert not app.metric
        uploads["after"] = Upload(b"invalid", "broken.png")
        app.run()
        assert not app.exception and app.error and app.button[0].disabled
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    transform = from_origin(0, 1000, 2, 2)  # Synthetic test coordinates, not a site location.
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=512, height=384, count=3, dtype="uint8",
                         crs="EPSG:3857", transform=transform) as dataset:
            dataset.write(np.moveaxis(image, -1, 0))
        tiff = memory.read()
    uploads = {"before": Upload(tiff, "synthetic_before.tif"), "after": Upload(tiff, "synthetic_after.tif")}
    with patch("streamlit.file_uploader", side_effect=lambda *args, **kw: uploads[kw["key"]]):
        app.run()
        app.button[0].click().run()
        assert not app.exception and not app.error
        result = app.session_state["analysis"][1]
        assert result["alignment_method"] == "CRS_reprojection" and result["accepted"]
        with zipfile.ZipFile(io.BytesIO(app.session_state["analysis"][3])) as archive:
            assert len([name for name in archive.namelist() if name.endswith(".tif")]) == 4
            with MemoryFile(archive.read("gis/vegetation_change.tif")) as memory, memory.open() as raster:
                assert raster.crs.to_epsg() == 3857 and raster.transform == transform
                assert raster.nodata == 255
    return {"passed": True, "engine": "ML" if use_ml else "ExG", "fixtures": "synthetic software checks",
            "six_metrics": True, "binary_masks": True, "rerun_persistence": True,
            "corrupt_input_handled": True, "geotiff_export_verified": True,
            "accuracy_benchmark": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml", action="store_true", help="Use actual downloaded neural weights")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.ml), indent=2))
