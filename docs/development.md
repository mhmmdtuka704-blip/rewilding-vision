# Development journey and engineering decisions

This project began as an ExG prototype during an internship at theOtherDada. The upgrade preserved the functional package/UI rather than replacing everything. Codex assisted implementation and verification; measured results were obtained from the retained runs, not generated as illustrative accuracy numbers.

## Milestones

1. **Inspect and preserve.** Find the functional project among existing copies; create a separate workspace; check CPU/RAM/Python; reuse working image handling and metrics.
2. **Make real ML work first.** Verify SegFormer-B0's trained vegetation-related labels, bundle a local checkpoint, and run an actual image through it.
3. **Make comparison spatially meaningful.** Strengthen ORB/RANSAC rejection, constrain the transform to modest similarity motion, exclude invalid boundaries, and use one denominator for change.
4. **Integrate and verify.** Cache the model, preserve session outputs, add relative change, overlays/maps/exports and repeat-image controls.
5. **Bounded domain adaptation.** Fetch a 244-pair sequence-disjoint subset; train only a residual head; use validation for deployment; evaluate 48 held-out images once.
6. **Add GIS and improve usability.** Reproject RGB GeoTIFFs/validity, preserve transforms through resizing, export standard GeoTIFFs, remove presentation controls, and expose binary masks plus the original ExG baseline.
7. **Prepare a reviewable public release.** Add portable setup, pinned/hash-checked acquisition, automated checks, diagrams, model/data cards, evidence links and clear rights boundaries. Keep the original local release intact.

## Why these choices?

| Decision | Reason / trade-off |
|---|---|
| SegFormer-B0, one backbone | A small pretrained semantic model fit the CPU budget; no competing architecture sweep was needed. |
| Residual head, frozen features | Genuine supervised adaptation without repeated full-network training. It preserves useful original evidence but has limited adaptation capacity. |
| Sequence-disjoint data | Avoid near-neighbor frame leakage across splits; still acknowledge within-sequence correlation. |
| Similarity rather than free homography | Enough for modest translation/rotation/scale while limiting implausible image distortion. It does not solve parallax. |
| Reject bad pairs | No measurement is better than presenting a precise-looking false change. |
| Shared valid footprint | Prevent denominator differences or nodata from masquerading as coverage change. |
| Raster reprojection for GIS | Use established CRS/grid behavior; do not build a map service or invent georeferencing. |
| Keep ExG explicit | A simple, inspectable baseline is useful; the UI must not call it ML or assign it neural benchmark scores. |
| Saved provenance and raw counts | Reviewers can trace data/model identity and reconstruct reported metrics. |

## What would add the most value next?

A labeled TOD repeat-view pilot with independent sites held out for evaluation. That would test the actual deployment domain and temporal error. More badges, a larger backbone or another synthetic demo would not replace that evidence.
