# Model card

## Identity and intended use

**SegFormer-B0 + residual RUGD vegetation head**, measured release `0.2.0`. Intended for research/evaluation of visible vegetation in RGB repeat imagery. It is not a species, biodiversity, plant-health, biomass or ecological-causation model.

The base checkpoint is [`nvidia/segformer-b0-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512), resolved at revision `489d5cd81a0b59fab9b7ea758d3548ebe99677da`. Local [provenance](../model/ade20k/provenance.json) includes file hashes. The original base has 3,752,694 parameters; this project did not perform its ImageNet/ADE20K pretraining.

## Formulation

Pretrained vegetation evidence groups ADE20K tree (4), grass (9), plant (17), flower (66) and palm (72). Log-sum-exp yields vegetation/background logits. A zero-initialized `Conv2d(256, 2, kernel_size=1)` adds a learned binary residual: 256×2 weights + 2 biases = **514 trainable parameters**.

The entire pretrained model remains frozen during the adapter run. The learned head is [2,192 bytes](../model/rugd_adapter/head.safetensors); it is not a standalone neural model. The final supervised target maps RUGD grass/tree/bush to vegetation and ignores void pixels.

## Preprocessing and inference

- RGB, rescaled to [0,1].
- Aspect-preserving 512×512 letterbox.
- ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
- Evaluation/inference mode; four CPU threads in the recorded run.
- Remove letterbox padding, resize the probability map to the processed image grid, threshold vegetation at 0.5.
- The GUI caps source processing at 1600 pixels per side; it preserves GeoTIFF bounds when resizing.

## Training

| Item | Recorded setting |
|---|---|
| Data | 160 RUGD training images; 36 validation images; whole sequences disjoint |
| Views | 160 original + 80 brightness/contrast augmented = 240 cached views |
| Seed | 251025 |
| Epochs / batch size | 5 / 4 |
| Optimizer | AdamW, learning rate 0.002, weight decay 0.01 |
| Loss | Pixelwise cross-entropy, ignore index 255 |
| Augmentation | Brightness 0.58–1.25, contrast 0.85–1.15, minibatch horizontal flips |
| Cache | Frozen features/grouped logits in float16; optimize the head in float32 |
| Selection | Best validation vegetation IoU; deploy only if better than frozen baseline |
| Runtime | 212.78 seconds on the recorded CPU laptop |

Validation IoU improved **92.52% → 93.20%**, and validation coverage MAE **3.54 → 2.92 pp**. Test labels were not used for training or model selection. The [complete curve](../benchmarks/training_history.json) includes the initial epoch that slightly reduced validation IoU.

## Evaluation

On **48 held-out images from four other sequences**, vegetation IoU was **88.64%**, Dice/F1 **93.98%**, pixel accuracy **90.97%**, and coverage MAE **5.73 pp**. [Results](results.md) provide exact counts and interpretation. These are ground-view subset results, not full-RUGD, aerial or TOD temporal field accuracy.

The separate **Binary ExG baseline** is a deterministic color index. It does not inherit these neural-model scores, and the GUI does not silently fall back to it when ML fails.

## Known limits

Small/occluded vegetation, dry or unfamiliar vegetation, camera changes, shadows, moving objects and domain shift can cause errors. The subset is modest and frames within each sequence remain correlated. There is no calibrated per-pair confidence interval or labeled TOD temporal validation. Inspect overlays and alignment before interpreting small changes.

## Terms and attribution

The upstream model card links the [NVIDIA SegFormer source/model terms](../licenses/NVIDIA-SegFormer.txt), which limit intended use to research/evaluation. This repository does not grant commercial model rights. Dataset and corporate names/logos have separate rights. See [third-party notices](../THIRD_PARTY_NOTICES.md).
