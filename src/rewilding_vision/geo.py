"""Bounded RGB image/GeoTIFF loading, common GIS grids, and QGIS rasters."""
from dataclasses import dataclass, field
import io
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, ImageOps


@dataclass
class ImageAsset:
    rgb: np.ndarray
    valid: np.ndarray
    geo: dict | None = None
    notes: list[str] = field(default_factory=list)


def load_asset(source, name=None, max_side=1600):
    if isinstance(source, (str, Path)):
        path = Path(source)
        data, name = path.read_bytes(), name or path.name
    elif isinstance(source, bytes):
        data = source
    else:
        data, name = source.getvalue(), name or getattr(source, "name", "image.png")
    is_tiff = str(name or "").lower().endswith((".tif", ".tiff")) or data[:4] in (b"II*\x00", b"MM\x00*")
    if not is_tiff:
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            rgb = np.asarray(image).copy()
        return ImageAsset(rgb, np.ones(rgb.shape[:2], bool))
    try:
        from rasterio.io import MemoryFile
        from rasterio.enums import ColorInterp, Resampling
        from rasterio.errors import NotGeoreferencedWarning
        from affine import Affine
    except ImportError as exc:
        raise RuntimeError("GeoTIFF support requires rasterio. Run SETUP_AND_RUN.bat.") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile(data) as memory, memory.open() as dataset:
            if dataset.count < 3:
                raise ValueError("GeoTIFF must contain RGB imagery (at least three bands), not a DEM or class mask.")
            colors = list(dataset.colorinterp)
            indices = ([colors.index(c)+1 for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)]
                       if all(c in colors for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)) else [1, 2, 3])
            scale = min(1.0, max_side / max(dataset.width, dataset.height))
            w, h = max(1, round(dataset.width*scale)), max(1, round(dataset.height*scale))
            values = dataset.read(indices, out_shape=(3, h, w), resampling=Resampling.bilinear)
            masks = dataset.read_masks(indices, out_shape=(3, h, w), resampling=Resampling.nearest)
            valid = (masks > 0).all(0) & np.isfinite(values).all(0)
            if not valid.any():
                raise ValueError("GeoTIFF contains no valid RGB pixels.")
            selected = values[:, valid]
            if values.dtype != np.uint8:
                if np.issubdtype(values.dtype, np.floating) and selected.min() >= 0 and selected.max() <= 1:
                    values = values * 255
                elif selected.min() < 0 or selected.max() > 255:
                    raise ValueError("Export this raster as 8-bit RGB in QGIS first; raw high-bit-depth reflectance is not display RGB.")
            rgb = np.moveaxis(np.rint(np.clip(np.nan_to_num(values), 0, 255)).astype(np.uint8), 0, -1)
            rgb[~valid] = 0
            geo = None
            notes = []
            if dataset.crs is not None:
                t = dataset.transform
                sx, sy = dataset.width/w, dataset.height/h
                transform = Affine(t.a*sx, t.b*sy, t.c, t.d*sx, t.e*sy, t.f)
                if not np.isfinite(tuple(transform)).all() or abs(transform.determinant) < 1e-15:
                    raise ValueError("GeoTIFF has an invalid geographic transform.")
                geo = {"crs": dataset.crs.to_string(), "transform": list(transform)[:6],
                       "width": w, "height": h}
            else:
                notes.append("This TIFF has no CRS; it will be handled as an ordinary image.")
            if indices == [1, 2, 3] and not all(c in colors for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
                notes.append("Using bands 1/2/3 as RGB; confirm that this is true-color imagery.")
            return ImageAsset(rgb, valid, geo, notes)


def reproject_to_reference(reference, moving):
    """Resample moving RGB + validity onto the processed reference GeoTIFF grid."""
    if reference.geo is None or moving.geo is None:
        raise ValueError("Both rasters require a CRS to use geographic alignment.")
    from affine import Affine
    from rasterio.warp import reproject, Resampling
    destination = np.full(reference.rgb.shape, np.nan, np.float32)
    kwargs = dict(src_transform=Affine(*moving.geo["transform"]), src_crs=moving.geo["crs"],
                  dst_transform=Affine(*reference.geo["transform"]), dst_crs=reference.geo["crs"])
    for band in range(3):
        source = moving.rgb[..., band].astype(np.float32)
        source[~moving.valid] = np.nan
        reproject(source, destination[..., band], src_nodata=np.nan, dst_nodata=np.nan,
                  resampling=Resampling.bilinear, **kwargs)
    validity = np.zeros(reference.valid.shape, np.uint8)
    reproject(moving.valid.astype(np.uint8), validity, src_nodata=0, dst_nodata=0,
              resampling=Resampling.nearest, **kwargs)
    valid = (validity > 0) & np.isfinite(destination).all(-1)
    rgb = np.rint(np.clip(np.nan_to_num(destination), 0, 255)).astype(np.uint8)
    return ImageAsset(rgb, valid, dict(reference.geo),
                      ["Geographic alignment uses source CRS and transforms; accuracy depends on source georeferencing."])


def geotiff_exports(artifacts, geo):
    if geo is None:
        return {}
    from affine import Affine
    from rasterio.io import MemoryFile
    valid = artifacts["valid_overlap"].astype(bool)
    palette = {0: (245, 245, 240, 255), 1: (44, 160, 90, 255),
               2: (210, 65, 65, 255), 3: (25, 105, 92, 255), 255: (0, 0, 0, 0)}
    outputs = {}
    for name, values in [("before_vegetation", artifacts["before_mask"]),
                         ("after_vegetation", artifacts["after_mask"]),
                         ("vegetation_change", artifacts["change_classes"]),
                         ("valid_overlap", artifacts["valid_overlap"])]:
        array = values.astype(np.uint8).copy()
        if name != "valid_overlap":
            array[~valid] = 255
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                             dtype="uint8", crs=geo["crs"], transform=Affine(*geo["transform"]),
                             nodata=255 if name != "valid_overlap" else None,
                             compress="deflate") as dataset:
                dataset.write(array, 1)
                dataset.set_band_description(1, name)
                dataset.update_tags(software="Rewilding Vision", comparison_grid="processed BEFORE raster")
                if name != "valid_overlap":
                    dataset.write_colormap(1, palette if name == "vegetation_change" else
                                           {0: (245, 245, 240, 255), 1: (25, 105, 92, 255), 255: (0, 0, 0, 0)})
            outputs[f"gis/{name}.tif"] = memory.read()
    return outputs
