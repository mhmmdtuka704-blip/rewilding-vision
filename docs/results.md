# Recorded results and what they mean

These results were measured on **25 October 2026**, before this repository was prepared. Publishing/documentation changes do not create a new training result.

## Held-out segmentation test

Source: [`benchmarks/test_metrics.json`](../benchmarks/test_metrics.json).

| Quantity | Value |
|---|---:|
| Images / held-out sequences | 48 / 4 |
| Evaluated valid pixels | 18,162,203 |
| True positive / false positive | 12,796,415 / 1,155,579 |
| False negative / true negative | 483,959 / 3,726,250 |
| Vegetation IoU | **88.6427%** |
| Vegetation Dice / F1 | **93.9795%** |
| Precision / recall | **91.7175% / 96.3558%** |
| Pixel accuracy | **90.9728%** |
| Coverage MAE | **5.7314 percentage points** |
| Mean measured inference | **0.4708 seconds/image**, CPU |

IoU = TP/(TP+FP+FN); F1 = 2TP/(2TP+FP+FN); accuracy = (TP+TN)/all valid pixels. Coverage MAE is an image-level mean absolute error in percentage points, not an accuracy percentage.

**Do not replace 88.64% test IoU with 93.20% validation IoU, call 93.98% F1 “94% IoU,” or claim universal 90%+ accuracy.** A pretrained-versus-adapter comparison on these same 48 test images was not recorded. The substantiated improvement is the before/after comparison on the same validation set.

## Training / validation

| Checkpoint | Validation IoU | Coverage MAE |
|---|---:|---:|
| Frozen pretrained grouping | 92.5153% | 3.5424 pp |
| Selected residual head | 93.2049% | 2.9152 pp |

Five epochs were completed; [all epoch records](../benchmarks/training_history.json) are available. Checkpoint selection did not load test labels. This is a modest improvement, about **0.69 percentage points of validation IoU**, rather than a claimed large breakthrough.

## Alignment controls after training

Source: [`benchmarks/alignment_controls.json`](../benchmarks/alignment_controls.json).

| Same-image perturbation | Maximum geometric recovery error | Apparent net change |
|---|---:|---:|
| Identical image | 0.000 px | 0.000 pp |
| +25/−18 px translation | 0.036 px | −0.021 pp |
| 4° rotation, scale 1.04, +16/−10 px | 0.093 px | −0.145 pp |

Identical-image gain/loss were exactly zero. Spatial false gain/loss counts for perturbed images are retained in the raw controls rather than hidden. These tests measure known-motion software behavior, not ecological accuracy.

## Synthetic lighting check

Twelve fixed test images (`test[::4]`) were darkened by multiplying RGB values by 0.65. Results were IoU **91.0630%**, F1 **95.3225%**, accuracy **92.4206%**, coverage MAE **4.3993 pp**. [Raw records](../benchmarks/darkening_metrics.json).

This smaller subset differs from the full 48-image set. Its higher score does not show that darkening improves the model, establish shadow invariance, or quantify superiority over ExG. No real shadow-removal model was built.

## Testing versus scientific evaluation

The original release passed 40 unit/regression tests, plus real-byte Streamlit integration and actual browser upload/export checks. The public repository adds portable setup and publication checks. CI runs software fixtures without distributing the real dataset or automatically training a model.

Passing software tests establishes tested behavior: correct denominators, rejection paths, transform metadata and exports. It is not an additional machine-learning accuracy measurement. Public-release test results are recorded in [release verification](release-verification.md).

## What remains unmeasured

There is no labeled TOD repeat-view pilot, external aerial benchmark, calibrated temporal error interval or independent deployment assessment. Real temporal change accuracy, target-site generalization and ecological significance remain open. The most useful next step is a fresh, labeled deployment-domain dataset with independent sites reserved for testing.
