"""Console entry points with imports delayed until the requested capability."""

from __future__ import annotations

import argparse
import json

from .config import load_yaml


def prepare_rugd_main():
    from pathlib import Path

    from .reproducibility import write_json
    from .rugd_dataset import audit_samples, discover_samples, write_split_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/rugd_splits.csv"))
    parser.add_argument("--audit-output", type=Path, default=Path("artifacts/rugd_audit.json"))
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    samples = discover_samples(args.images, args.masks)
    audit = audit_samples(samples)
    assignments = write_split_manifest(samples, args.output, args.seed)
    write_json(args.audit_output, {"audit": audit, "group_assignments": assignments})
    print(json.dumps(audit, indent=2))


def train_main():
    from .train import train_segformer

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/week2.yaml")
    args = parser.parse_args()
    print(json.dumps(train_segformer(load_yaml(args.config)), indent=2))


def evaluate_main():
    from .model_evaluation import evaluate_checkpoint
    from .reproducibility import write_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/processed/rugd_splits.csv")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output", default="artifacts/test_metrics.json")
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.manifest, args.split)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


def analyze_pair_main():
    from .pair_analysis import pair_analysis_cli

    pair_analysis_cli()
