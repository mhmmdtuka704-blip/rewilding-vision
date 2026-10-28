# Reproduce the recorded experiment

## Artifacts and identity

- Base checkpoint identifier, pinned revision and hashes: [model provenance](../model/ade20k/provenance.json).
- Actual trained residual weights and gate: [adapter metadata](../model/rugd_adapter/metadata.json).
- Exact 160/36/48 split: [manifest](../data/rugd_subset/manifest.csv).
- Source revision and every image/annotation hash: [dataset provenance](../data/rugd_subset/provenance.json).
- Five epochs and measured test results: [benchmarks](../benchmarks/README.md).
- Original installed versions: [requirements-validated.txt](../requirements-validated.txt). This is a record, not a complete transitive lockfile.

## Steps

Create the environment using the root README, then:

```bash
python scripts/setup_model.py
python scripts/prepare_training_subset.py
python scripts/evaluate_followup.py
```

The dataset is optional for ordinary app use. Acquisition scripts verify the recorded hashes and do not silently follow newer model/data revisions. Download availability and upstream terms still apply.

## Retraining

```bash
python scripts/train_lightweight_head.py
```

This explicit operation can write a newly selected adapter. Preserve the shipped checkpoint and raw benchmark records before experimenting; do not treat a rerun as the historical run.

The recipe freezes pretrained SegFormer, caches 240 original/photometric feature views, and optimizes a 514-parameter 1×1 head with AdamW (0.002 learning rate, 0.01 decay), batch size 4, cross-entropy ignoring 255, five epochs, seed 251025. Horizontal flips are applied to features, logits and labels together. Validation alone chooses the checkpoint.

The original run used an i7-13620H CPU, about 16.86 GB RAM, Intel integrated graphics and no CUDA. Training took 212.78 seconds; inference timing varied with concurrent laptop activity. Bitwise identity across PyTorch/platform versions is not guaranteed by setting a seed alone.

## Keep the test set honest

The published test split has now been observed. New model tuning should reserve new independent sequences/sites rather than iterating until this test score looks better. Synthetic geometry controls and app fixtures establish software correctness, not accuracy on real temporal field data.

## Public-versus-local packaging

The original working folder remains separate. This public copy omits raw data, base weights, logs, upload caches, company logo files and personal machine paths. It retains the trained adapter under upstream research/evaluation terms, the exact manifest/hashes and recorded evidence. A decorative SVG replaces the RUGD photograph used in the local UI header.
