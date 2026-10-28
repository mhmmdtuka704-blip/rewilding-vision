"""Rewilding Vision — local vegetation change and georeferenced raster analysis."""
from pathlib import Path
import hashlib
import base64
import json
import sys

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from rewilding_vision.config import load_yaml
from rewilding_vision.exports import export_zip, png_bytes
from rewilding_vision.geo import load_asset, reproject_to_reference
from rewilding_vision.pair_analysis import ExGPredictor, analyze_pair
from rewilding_vision.segformer import SegFormerPredictor

st.set_page_config(page_title="Rewilding Vision | Vegetation analysis",
                   page_icon=":material/eco:", layout="wide", initial_sidebar_state="collapsed")
st.html("""<style>
.stApp {background:radial-gradient(ellipse at 10% 0%,#17372b 0%,transparent 48%),#071b15;}
.block-container {max-width:1320px;padding-top:1.7rem;padding-bottom:3rem;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stAppDeployButton"] {display:none;}
h1 {letter-spacing:-.055em;line-height:1.02!important;font-size:clamp(3rem,5vw,4.7rem)!important;color:#f0f5e8;}
h2,h3 {letter-spacing:-.025em;color:#ecf3e5;}
[data-testid="stMetric"] {background:linear-gradient(140deg,#18392b,#10291f);border:1px solid #34533e;border-radius:16px;padding:1rem 1.2rem;}
[data-testid="stMetricValue"] {font-weight:600;letter-spacing:-.035em;}
[data-testid="stFileUploader"] {border-radius:14px;background:#10291f;border:1px dashed #4b7056;padding:.45rem;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#31543f!important;border-radius:18px!important;}
[data-testid="stBaseButton-primary"] {background:#c4df91;color:#12281b;border:0;border-radius:12px;min-height:3rem;font-weight:700;}
[data-testid="stBaseButton-primary"]:hover {background:#d8ecae;color:#12281b;}
[data-testid="stBaseButton-primary"]:disabled {background:#314b37;color:#9aaf9d;}
[data-testid="stTabs"] button[aria-selected="true"] {color:#d2e5a1;}
.rv-brand {display:flex;align-items:center;gap:22px;margin-bottom:1.3rem;}
.rv-logo {display:flex;align-items:center;background:#e8ede4;padding:9px 16px;border-radius:12px;}
.rv-logo img {width:185px;height:auto;display:block;}
.rv-brand-name {font-size:.76rem;letter-spacing:.15em;color:#b7cba8;font-weight:650;line-height:1.9;}
.rv-hero {border:1px solid #3b5b41;border-radius:24px;padding:30px 36px;margin-bottom:1.3rem;background-size:cover;background-position:center 42%;box-shadow:0 20px 60px #0003;}
.rv-hero h1 {margin:.6rem 0 1rem!important;max-width:700px;}
.rv-hero em {font-family:Georgia,serif;font-weight:400;color:#d4e7a6;}
.rv-hero p {max-width:510px;color:#c9d9c8;line-height:1.65;margin:0;}
.rv-eyebrow {font-size:.7rem;letter-spacing:.18em;font-weight:650;color:#d1e2ae;}
.rv-pill {display:inline-block;border:1px solid #8bac7666;background:#071b1599;border-radius:30px;padding:5px 11px;font-size:.7rem;color:#d6e7bf;margin:17px 8px 0 0;}
.rv-rule {height:1px;background:#31503b;margin:1.4rem 0;}
.rv-footer {font-size:.78rem;color:#9ab29c;margin-top:2rem;}
@media(max-width:700px){h1{font-size:2.2rem!important;}.block-container{padding-top:1rem;}}
</style>""")


def model_signature():
    path = ROOT / "model/rugd_adapter/metadata.json"
    return path.read_text(encoding="utf-8") if path.exists() else "pretrained"


@st.cache_resource(show_spinner=False, max_entries=1)
def load_model(signature):
    if not (ROOT / "model/ade20k/model.safetensors").is_file():
        raise RuntimeError("ML weights are missing. Run python scripts/setup_model.py, or use Binary ExG baseline.")
    return SegFormerPredictor()


@st.cache_data(show_spinner=False, max_entries=4)
def read_upload(data, name):
    return load_asset(data, name=name)


logo_path = ROOT / "assets/tod-logo.png"
logo = base64.b64encode(logo_path.read_bytes()).decode() if logo_path.exists() else None
forest = base64.b64encode((ROOT / "assets/canopy.svg").read_bytes()).decode()
logo_markup = f'<img src="data:image/png;base64,{logo}" alt="theOtherDada logo">' if logo else '<strong style="font-size:1.8rem;color:#163526">tOD</strong>'
st.html(f'<div class="rv-brand"><div class="rv-logo">{logo_markup}</div>'
        '<div class="rv-brand-name">tOD · theOtherDada<br>REGENERATIVE DESIGN / VEGETATION INTELLIGENCE</div></div>')
st.html(f'<div class="rv-hero" style="background-image:linear-gradient(90deg,rgba(7,27,21,.98) 8%,rgba(7,27,21,.87) 42%,rgba(7,27,21,.38)),url(data:image/svg+xml;base64,{forest})">'
        '<div class="rv-eyebrow">SEE THE CHANGE. SUPPORT THE RECOVERY.</div>'
        '<h1>Rewilding <em>Vision.</em></h1>'
        '<p>A clearer view of what grows. Compare vegetation across time with machine learning, binary analysis and geographic context.</p>'
        '<span class="rv-pill">LOCAL & PRIVATE</span><span class="rv-pill">ML + BINARY MASKS</span><span class="rv-pill">QGIS READY</span></div>')

method_choice = st.radio("Vegetation detection", ["ML segmentation", "Binary ExG baseline"],
                         horizontal=True, key="detection_method")
use_ml = method_choice == "ML segmentation"
st.caption("ML recognizes vegetation from trained image features. The binary ExG baseline measures green-dominant pixels; "
           "both produce black-and-white masks and use the same alignment and GIS pipeline.")

st.subheader("01  Upload your imagery")
st.caption("Choose two photographs or RGB GeoTIFFs of the same location. "
           "Use consistent viewpoints for photographs; geographic rasters align through their CRS.")
before = after = None
files = []
first, second = st.columns(2, gap="large")
for column, label, key in [(first, "Before", "before"), (second, "After", "after")]:
    with column:
        with st.container(border=True):
            st.markdown(f"**{label} imagery**")
            uploaded = st.file_uploader(f"{label} image", type=["png", "jpg", "jpeg", "tif", "tiff"],
                                       key=key, label_visibility="collapsed")
            files.append(uploaded)
            if uploaded:
                try:
                    asset = read_upload(uploaded.getvalue(), uploaded.name)
                    if key == "before":
                        before = asset
                    else:
                        after = asset
                    st.image(asset.rgb, width="stretch")
                    h, w = asset.rgb.shape[:2]
                    st.caption(f"{uploaded.name} · {w:,} × {h:,} pixels" +
                               (f" · {asset.geo['crs']}" if asset.geo else " · Photograph / no CRS"))
                    for note in asset.notes:
                        st.warning(note)
                except Exception as exc:
                    st.error(f"Could not read {label.lower()} image: {exc}")
            else:
                st.caption("JPG, PNG, or true-color GeoTIFF · Up to 100 MB per file")

ready = before is not None and after is not None
signature = "ml:" + model_signature() if use_ml else "exg:0.10"
fingerprint = hashlib.sha256(files[0].getvalue() + files[1].getvalue() + signature.encode()).hexdigest() if ready else None
action, explanation = st.columns([1, 2])
with action:
    analyze = st.button("Analyze vegetation change", type="primary", width="stretch",
                        icon=":material/analytics:", disabled=not ready)
with explanation:
    st.caption("Both images use one shared comparison area. Unsupported alignment is rejected. "
               "Large images are processed at up to 1600 pixels per side.")

if analyze:
    st.session_state.pop("analysis", None)
    try:
        with st.spinner("Aligning imagery and identifying vegetation..."):
            cfg = load_yaml(ROOT / "configs/week3.yaml")
            predictor = load_model(signature) if use_ml else ExGPredictor(cfg["analysis"]["exg_threshold"])
            geo_aligned = before.geo is not None and after.geo is not None
            moving = reproject_to_reference(before, after) if geo_aligned else after
            result, artifacts = analyze_pair(before.rgb, moving.rgb, predictor, cfg["registration"],
                before_valid=before.valid, after_valid=moving.valid, geo_aligned=geo_aligned)
            result.update(method=("segformer_b0_rugd_adapted" if getattr(predictor, "adapter_metadata", None)
                                  else "segformer_b0_ade20k") if use_ml else "binary_exg_baseline",
                          detection_label=method_choice,
                          exg_threshold=None if use_ml else predictor.threshold,
                          vegetation_classes=getattr(predictor, "vegetation_labels", ["visible_green_pixels"]),
                          model_training=getattr(predictor, "adapter_metadata", None),
                          georeferencing=before.geo, input_names=[file.name for file in files],
                          input_shapes={"before": list(before.rgb.shape), "after": list(after.rgb.shape)},
                          model_input_size=getattr(predictor, "image_size", None),
                          limitations=["Visible image coverage, not biomass or biodiversity",
                                       "Source georeferencing may contain positional error" if geo_aligned else
                                       "Camera parallax or occlusion can defeat alignment",
                                       "Segmentation errors can produce residual change"])
            if result["accepted"]:
                bundle = export_zip(result, artifacts, before.geo)
            else:
                bundle = None
            st.session_state["analysis"] = (fingerprint, result, artifacts, bundle)
    except Exception as exc:
        st.error(f"Analysis could not be completed: {exc}")
        st.caption("Check that both files contain readable RGB imagery of the same scene.")

stored = st.session_state.get("analysis")
if stored and stored[0] == fingerprint:
    _, result, artifacts, bundle = stored
    st.html('<div class="rv-rule"></div>')
    st.subheader("02  Your vegetation comparison")
    if not result["accepted"]:
        st.warning("The images could not be compared reliably. No vegetation-change measurements were calculated.")
        st.caption(result["registration"]["reason"])
        with st.expander("Alignment diagnostics"):
            st.json(result)
    else:
        change = result["change"]
        st.badge(result["detection_label"], icon=":material/eco:", color="green")
        a, b, c = st.columns(3)
        a.metric("Before coverage", f"{change['before_coverage_percent']:.2f}%")
        b.metric("After coverage", f"{change['after_coverage_percent']:.2f}%")
        c.metric("Net vegetation change", f"{change['change_percentage_points']:+.2f} pp")
        a, b, c = st.columns(3)
        a.metric("Vegetation gained", f"{change['gain_percent']:.2f}%")
        b.metric("Vegetation lost", f"{change['loss_percent']:.2f}%")
        relative = change["relative_change_percent"]
        c.metric("Relative change", "N/A" if relative is None else f"{relative:+.2f}%")
        st.caption(f"Shared valid area: {change['evaluated_pixels']:,} pixels · "
                   f"{result['registration']['overlap_ratio']:.1%} of the reference image. "
                   "Gain and loss use that same area; relative change uses before vegetation coverage.")
        if relative is None:
            st.caption("Relative change is undefined because the before image has zero vegetation coverage.")
        if result["alignment_method"] == "CRS_reprojection":
            st.info("Aligned to the reference GeoTIFF grid. Positional accuracy depends on the source georeferencing.")
        else:
            st.badge("Visual alignment accepted", icon=":material/check_circle:", color="green")
        if result["georeferencing"]:
            st.badge(f"QGIS export ready · {result['georeferencing']['crs']}", icon=":material/public:", color="blue")

        maps, binary, vegetation, alignment = st.tabs(["Change map", "Binary masks", "Vegetation overlays", "Alignment & footprint"])
        with maps:
            st.image(artifacts["change_map"], width="stretch")
            st.caption("GREEN  Gain   ·   RED  Loss   ·   TEAL  Stable vegetation   ·   WHITE  Stable background   ·   GRAY  Excluded")
        with vegetation:
            a, b = st.columns(2)
            a.image(artifacts["before_overlay"], caption="Before · vegetation highlighted", width="stretch")
            b.image(artifacts["after_overlay"], caption="Aligned after · vegetation highlighted", width="stretch")
        with binary:
            a, b = st.columns(2)
            a.image(artifacts["before_mask"]*255, caption="Before · binary vegetation mask", width="stretch")
            b.image(artifacts["after_mask"]*255, caption="After · binary vegetation mask", width="stretch")
            st.caption("White = vegetation; black = background. Apply the valid-footprint mask to exclude unobserved areas.")
        with alignment:
            a, b = st.columns(2)
            a.image(artifacts["aligned_after"], caption="After, on the before image grid", width="stretch")
            b.image(artifacts["valid_overlap"]*255, caption="White = valid shared comparison area", width="stretch")
            st.caption("Inspect alignment before interpreting small changes. Movement, shadows, occlusions "
                       "and segmentation errors can affect the result.")
        st.subheader("03  Export your results")
        a, b, c = st.columns(3)
        a.download_button("Measurements · JSON", json.dumps(result, indent=2, allow_nan=False),
                          "rewilding_vision_result.json", "application/json", on_click="ignore", width="stretch")
        b.download_button("Visual report · PNG", png_bytes(artifacts["panel"]),
                          "rewilding_vision_report.png", "image/png", on_click="ignore", width="stretch")
        c.download_button("GIS + results · ZIP" if result["georeferencing"] else "Masks + results · ZIP", bundle,
                          "rewilding_vision_results.zip", "application/zip", on_click="ignore", width="stretch")
        if result["georeferencing"]:
            st.caption("Open the ZIP and drag gis/*.tif into QGIS. CRS, reference-grid transform, class colors "
                       "and nodata are embedded. Geographic metadata is adjusted when the reference is resized.")
        with st.expander("Method, formulas & diagnostics"):
            st.write("Coverage = vegetation / valid shared pixels × 100. Gain/loss use the same footprint. "
                     "Net = after − before = gain − loss. Relative = (after − before) / before × 100.")
            st.json(result)
else:
    st.caption("Your comparison, change map, and downloads will appear after analysis.")

with st.expander("Model verification & supported imagery"):
    metrics_path = ROOT / "benchmarks/test_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.write(f"ML only — held-out RUGD subset: {metrics['images']} images. "
                 f"Vegetation IoU {metrics['vegetation_iou']:.2%}; Dice/F1 {metrics['dice_f1']:.2%}; "
                 f"coverage MAE {metrics['coverage_mae_pp']:.2f} percentage points.")
    st.caption("The model is intended for visible RGB vegetation. Small benchmark results do not guarantee accuracy "
               "on your imagery. Ground photos use ORB/RANSAC; paired GeoTIFFs use CRS reprojection. "
               "Raw multispectral reflectance, DEMs and single-band masks are not RGB inputs.")
st.html('<div class="rv-footer">tOD / theOtherDada · REWILDING VISION<br>'
        'Visible vegetation coverage from repeat imagery. Ecological interpretation requires field context.</div>')
