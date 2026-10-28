"""Deterministic group splitting that prevents sequence leakage."""

from __future__ import annotations

import hashlib
from collections import defaultdict


def assign_group_splits(groups, train_fraction=0.67, validation_fraction=0.17, seed=20260812):
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must leave non-zero train, validation, and test sets")
    unique = sorted(set(groups))
    if len(unique) < 3:
        raise ValueError("At least three sequences are required")

    def stable_score(group):
        digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    ordered = sorted(unique, key=stable_score)
    # Reserve one group for validation and one for test before allocating train.
    train_count = min(max(1, round(len(ordered) * train_fraction)), len(ordered) - 2)
    remaining = len(ordered) - train_count
    validation_count = min(max(1, round(len(ordered) * validation_fraction)), remaining - 1)
    train_end = train_count
    validation_end = train_count + validation_count
    assignments = {}
    for i, group in enumerate(ordered):
        assignments[group] = (
            "train" if i < train_end else "validation" if i < validation_end else "test"
        )
    validate_no_group_leakage(assignments)
    return assignments


def validate_no_group_leakage(group_to_split):
    reverse = defaultdict(set)
    for group, split in group_to_split.items():
        reverse[group].add(split)
    leaked = {group: splits for group, splits in reverse.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Sequence leakage detected: {leaked}")
    present = set(group_to_split.values())
    if present != {"train", "validation", "test"}:
        raise ValueError(f"Expected all three splits, found {sorted(present)}")
