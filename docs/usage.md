# Inputs, measurements and exports

## Choose a method

- **ML segmentation:** pretrained SegFormer + the deployed supervised adapter. It uses learned image features and needs local base weights from `scripts/setup_model.py`.
- **Binary ExG baseline:** `2g-r-b >= 0.10`, where each RGB channel is divided by their sum with guarded zero division. It measures green dominance, can miss dry/dark vegetation, and can include artificial green objects. It is not ML and does not inherit the neural benchmark.

Both output binary masks. White/black describes an output encoding, not whether the algorithm is learned.

## Image preparation and alignment

Use repeated views of the same scene. JPG/PNG orientation follows EXIF. Longest-side processing is capped at 1600 pixels. Neural inference uses a 512-pixel letterbox.

Photographs undergo ORB/RANSAC registration and quality gates. Two georeferenced rasters use CRS/affine reprojection onto the BEFORE grid. A GeoTIFF without a CRS is treated as ordinary imagery; no coordinates are invented. Mixed image/raster pairs use visual alignment, with geographic output only when the BEFORE reference has georeferencing.

The shared valid footprint must cover at least 70% of the processed reference under the current configuration. Low-texture, unrelated, severely changed-viewpoint or disjoint-footprint pairs can be rejected. A rejection yields no change metrics.

Supported geographic inputs are true-color RGB: 8-bit or display floating RGB in [0,1]. Raw high-bit-depth reflectance, DEMs, single-band masks and arbitrary NIR composites need conversion to display RGB first. CRS correctness is a source-data assumption, not proven by the application.

## Formulas

Let N be valid overlapping pixels; B/A vegetation before/after; G pixels changing background→vegetation; L vegetation→background.

| Measurement | Formula |
|---|---|
| Before / after coverage | 100B/N; 100A/N |
| Gain / loss | 100G/N; 100L/N |
| Net, percentage points | 100(A−B)/N = gain − loss |
| Relative, percent | 100(A−B)/B; undefined when B=0 |

Gain and loss can both be positive even when net change is zero. The relative metric intentionally uses a different denominator. A zero starting vegetation area returns JSON `null` and UI `N/A`.

## Export contract

The ZIP includes JSON, both binary masks and overlays, aligned AFTER, valid footprint, categorical/colored change maps, and the three-panel visual report. It embeds a README for the encodings.

| Raster | Encoding |
|---|---|
| PNG binary masks | 255 vegetation, 0 background; apply valid footprint |
| PNG valid footprint | 255 valid, 0 invalid |
| Change classes | 0 stable background, 1 gain, 2 loss, 3 stable vegetation, 255 invalid |
| GeoTIFF vegetation masks | 0 background, 1 vegetation, 255 nodata |
| GeoTIFF valid footprint | 0 invalid, 1 valid |

When the reference is georeferenced, four TIFFs appear under `gis/`. They preserve its CRS and resolution-adjusted affine transform, with nodata and palettes. Drag them into QGIS. The project verifies TIFF reopening via Rasterio/GDAL; it does not claim a separately certified external QGIS desktop integration.

Green/red/teal/white/gray in the rendered change map mean gain/loss/stable vegetation/stable background/excluded. Values concern visible pixel coverage, not biomass, species, biodiversity or ecological causation.
