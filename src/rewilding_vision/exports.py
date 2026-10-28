"""Portable measured-result and QGIS export bundles."""
import io
import json
import zipfile
import numpy as np
from PIL import Image
from .geo import geotiff_exports


def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def export_zip(result, artifacts, geo=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("result.json", json.dumps(result, indent=2, allow_nan=False))
        archive.writestr("README.txt", "All outputs use the processed BEFORE image grid.\n"
            "valid_overlap.png: 255 valid, 0 invalid. Apply it to all PNG masks.\n"
            "PNG vegetation masks: 255 vegetation, 0 background.\n"
            "Categorical change: 0 stable background, 1 gain, 2 loss, 3 stable vegetation, 255 invalid.\n"
            "GeoTIFF vegetation masks: 0 background, 1 vegetation, 255 nodata.\n"
            "GIS exports preserve BEFORE CRS and its resolution-adjusted affine transform.\n"
            "QGIS: drag gis/*.tif into QGIS; palette and nodata are embedded.\n"
            "Gain/loss/coverage percentages use the SAME valid comparison footprint.\n"
            "Measurements describe visible image coverage, not biomass or biodiversity.\n")
        for name, values in artifacts.items():
            array = values * 255 if name in {"before_mask", "after_mask", "valid_overlap"} else values
            archive.writestr(f"{name}.png", png_bytes(array))
        for filename, data in geotiff_exports(artifacts, geo).items():
            archive.writestr(filename, data)
    return buffer.getvalue()
