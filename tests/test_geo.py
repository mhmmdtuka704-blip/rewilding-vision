import io
import zipfile

import numpy as np
import pytest
from affine import Affine
from rasterio.io import MemoryFile

from rewilding_vision.exports import export_zip
from rewilding_vision.geo import load_asset, reproject_to_reference
from rewilding_vision.pair_analysis import analyze_pair


def make_tiff(rgb, transform=Affine(2, 0, 500000, 0, -2, 3700000), crs="EPSG:32636", valid=None):
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=rgb.shape[1], height=rgb.shape[0], count=3,
                         dtype=str(rgb.dtype), crs=crs, transform=transform) as dataset:
            dataset.write(np.moveaxis(rgb, -1, 0))
            if valid is not None:
                dataset.write_mask(valid.astype(np.uint8)*255)
        return memory.read()


class BinaryPredictor:
    def predict(self, image):
        mask = (image[..., 1] > image[..., 0]).astype(np.uint8)
        return mask, mask.astype(np.float32)


def test_resizing_preserves_bounds_and_nodata():
    rgb = np.full((200, 300, 3), 100, np.uint8)
    valid = np.ones((200, 300), bool); valid[:20] = False
    asset = load_asset(make_tiff(rgb, valid=valid), name="reference.tif", max_side=150)
    assert asset.rgb.shape == (100, 150, 3)
    transform = Affine(*asset.geo["transform"])
    assert (transform.a*150 + transform.b*100 + transform.c,
            transform.d*150 + transform.e*100 + transform.f) == (500600, 3699600)
    assert not asset.valid[:10].any()
    assert asset.valid[10:].all()


def test_shifted_geotiff_shared_area_and_qgis_export_roundtrip():
    rgb = np.zeros((100, 120, 3), np.uint8)
    rgb[:] = (150, 90, 40); rgb[:, :60] = (30, 160, 40)
    before = load_asset(make_tiff(rgb), name="before.tif")
    # Same landscape, shifted GIS footprint by 20 pixels; use source pixels for that footprint.
    shifted = np.zeros_like(rgb); shifted[:, :100] = rgb[:, 20:]
    after = load_asset(make_tiff(shifted, Affine(2, 0, 500040, 0, -2, 3700000)), name="after.tif")
    moving = reproject_to_reference(before, after)
    result, artifacts = analyze_pair(before.rgb, moving.rgb, BinaryPredictor(),
                                    {"minimum_overlap_ratio": .7}, before_valid=before.valid,
                                    after_valid=moving.valid, geo_aligned=True)
    assert result["accepted"] and result["alignment_method"] == "CRS_reprojection"
    assert result["change"]["evaluated_pixels"] == 100*100
    assert result["change"]["gain_percent"] == result["change"]["loss_percent"] == 0
    data = export_zip(result, artifacts, before.geo)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert len([name for name in archive.namelist() if name.endswith(".tif")]) == 4
        with MemoryFile(archive.read("gis/vegetation_change.tif")) as memory, memory.open() as dataset:
            assert dataset.crs.to_epsg() == 32636
            assert dataset.transform == Affine(*before.geo["transform"])
            assert dataset.nodata == 255
            assert np.all(dataset.read(1)[:, :20] == 255)
            assert dataset.colormap(1)[3][:3] == (25, 105, 92)


def test_different_crs_reprojects_without_fabricating_reference_metadata():
    from rasterio.warp import transform_bounds
    from rasterio.transform import from_bounds
    rgb = np.full((80, 100, 3), [30, 160, 40], np.uint8)
    bounds = (35.49, 33.89, 35.50, 33.90)
    before = load_asset(make_tiff(rgb, from_bounds(*bounds, 100, 80), "EPSG:4326"), name="before.tif")
    projected = transform_bounds("EPSG:4326", "EPSG:3857", *bounds)
    after = load_asset(make_tiff(rgb, from_bounds(*projected, 100, 80), "EPSG:3857"), name="after.tif")
    moved = reproject_to_reference(before, after)
    assert moved.geo == before.geo
    assert moved.valid.mean() > .95
    assert np.all(moved.rgb[moved.valid] == [30, 160, 40])


def test_disjoint_geographic_footprints_reject_change():
    rgb = np.full((50, 60, 3), 100, np.uint8)
    before = load_asset(make_tiff(rgb), name="before.tif")
    after = load_asset(make_tiff(rgb, Affine(2, 0, 600000, 0, -2, 3700000)), name="after.tif")
    moved = reproject_to_reference(before, after)
    result, artifacts = analyze_pair(before.rgb, moved.rgb, BinaryPredictor(), {},
                                    before_valid=before.valid, after_valid=moved.valid, geo_aligned=True)
    assert not result["accepted"] and "change" not in result and not artifacts


def test_high_bit_depth_reflectance_rejected_with_actionable_error():
    with pytest.raises(ValueError, match="8-bit RGB"):
        load_asset(make_tiff(np.full((20, 20, 3), 10000, np.uint16)), name="reflectance.tif")


def test_photos_do_not_invent_georeferencing():
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (120, 80), (30, 160, 40)).save(buffer, "PNG")
    asset = load_asset(buffer.getvalue(), name="photo.png")
    assert asset.geo is None and asset.valid.all()
