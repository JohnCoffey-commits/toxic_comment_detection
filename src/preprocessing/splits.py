"""Deterministic stratified split construction."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def assign_jigsaw_splits(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create train/val/test split assignments using two-stage StratifiedShuffleSplit."""

    out = df.copy()
    labels = out["binary_label"].astype(int)
    full_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_temp_idx, test_idx = next(full_split.split(out, labels))

    train_temp = out.iloc[train_temp_idx]
    train_temp_labels = train_temp["binary_label"].astype(int)
    relative_val_size = val_size / (1.0 - test_size)
    val_split = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_seed + 1)
    train_rel_idx, val_rel_idx = next(val_split.split(train_temp, train_temp_labels))

    out["split"] = ""
    out.iloc[train_temp_idx[train_rel_idx], out.columns.get_loc("split")] = "train"
    out.iloc[train_temp_idx[val_rel_idx], out.columns.get_loc("split")] = "val"
    out.iloc[test_idx, out.columns.get_loc("split")] = "test"
    summary = {
        "method": "two-stage StratifiedShuffleSplit",
        "stratify_column": "binary_label",
        "random_seed_stage_1": random_seed,
        "random_seed_stage_2": random_seed + 1,
        "target_ratio": {"train": 0.70, "val": val_size, "test": test_size},
        "counts": out["split"].value_counts().sort_index().to_dict(),
    }
    return out, summary


def assign_civil_splits(
    df: pd.DataFrame,
    external_test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create Civil augmentation pool and external test assignments with stratification."""

    out = df.copy()
    labels = out["binary_label"].astype(int)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=external_test_size, random_state=random_seed)
    aug_idx, external_idx = next(splitter.split(out, labels))
    out["split"] = ""
    out.iloc[aug_idx, out.columns.get_loc("split")] = "aug_pool"
    out.iloc[external_idx, out.columns.get_loc("split")] = "external_test"
    summary = {
        "method": "StratifiedShuffleSplit",
        "stratify_column": "binary_label",
        "random_seed": random_seed,
        "target_ratio": {"aug_pool": 1.0 - external_test_size, "external_test": external_test_size},
        "counts": out["split"].value_counts().sort_index().to_dict(),
    }
    return out, summary
