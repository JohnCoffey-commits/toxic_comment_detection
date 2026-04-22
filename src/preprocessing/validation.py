"""Quality checks and leakage validation."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from .tokenization import token_length_diagnostics


class ValidationError(RuntimeError):
    """Raised when strict preprocessing validation fails."""


def _class_dist(df: pd.DataFrame) -> dict[str, Any]:
    labels = df["binary_label"].astype(int)
    return {
        "row_count": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "positive_ratio": float(labels.mean()) if len(df) else None,
    }


def _split_class_dist(df: pd.DataFrame) -> dict[str, Any]:
    return {split: _class_dist(group) for split, group in df.groupby("split")}


def _hash_overlap(df: pd.DataFrame, hash_column: str) -> dict[str, Any]:
    overlaps: dict[str, Any] = {}
    split_sets = {
        split: set(group[hash_column].dropna().astype(str))
        for split, group in df.groupby("split")
    }
    for left, right in combinations(sorted(split_sets), 2):
        overlap = split_sets[left].intersection(split_sets[right])
        overlaps[f"{left}__{right}"] = {
            "overlap_count": len(overlap),
            "example_hashes": sorted(overlap)[:10],
        }
    return overlaps


def validate_dataset(
    df: pd.DataFrame,
    source: str,
    label_drift_threshold: float,
    suspicious_split_min_rows: int,
    fail_on_leakage: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Run strict validation checks for one split-assigned dataset."""

    warnings: list[str] = []
    overall = _class_dist(df)
    by_split = _split_class_dist(df)
    overall_ratio = overall["positive_ratio"] or 0.0
    drift = {
        split: abs((stats["positive_ratio"] or 0.0) - overall_ratio)
        for split, stats in by_split.items()
    }
    for split, stats in by_split.items():
        if stats["row_count"] < suspicious_split_min_rows:
            warnings.append(f"{source}:{split} is suspiciously small with {stats['row_count']} rows.")
    if drift and max(drift.values()) > label_drift_threshold:
        warnings.append(
            f"{source} label-ratio drift across splits exceeds {label_drift_threshold}: {drift}."
        )

    raw_overlap = _hash_overlap(df, "text_hash_raw")
    normalized_overlap = _hash_overlap(df, "text_hash_normalized")
    leakage_pairs = [
        ("raw", pair, value["overlap_count"])
        for pair, value in raw_overlap.items()
        if value["overlap_count"] > 0
    ] + [
        ("normalized", pair, value["overlap_count"])
        for pair, value in normalized_overlap.items()
        if value["overlap_count"] > 0
    ]
    if leakage_pairs:
        message = f"{source} leakage detected by hash overlap: {leakage_pairs}."
        warnings.append(message)
        if fail_on_leakage:
            raise ValidationError(message)

    results = {
        "source_dataset": source,
        "missing_text_after_cleaning": int(df["text_clean"].fillna("").astype(str).str.strip().eq("").sum()),
        "missing_labels_after_harmonization": int(df["binary_label"].isna().sum()),
        "duplicate_counts_after_removal": {
            "raw_hash_duplicates": int(df["text_hash_raw"].duplicated(keep="first").sum()),
            "normalized_hash_duplicates": int(df["text_hash_normalized"].duplicated(keep="first").sum()),
        },
        "class_distribution_overall": overall,
        "class_distribution_by_split": by_split,
        "label_ratio_drift_abs": drift,
        "raw_hash_overlap_across_splits": raw_overlap,
        "normalized_hash_overlap_across_splits": normalized_overlap,
        "distilbert_token_length_diagnostics": token_length_diagnostics(
            df["bert_token_len"], "distilbert-base-uncased"
        )
        if "bert_token_len" in df.columns
        else {"available": False},
    }
    return results, warnings


def validate_civil_aug_external_overlap(df: pd.DataFrame, fail_on_leakage: bool) -> tuple[dict[str, Any], list[str]]:
    """Validate Civil augmentation pool/external test overlap."""

    warnings: list[str] = []
    aug = df.loc[df["split"].eq("aug_pool"), "text_hash_normalized"]
    external = df.loc[df["split"].eq("external_test"), "text_hash_normalized"]
    overlap = set(aug).intersection(set(external))
    result = {
        "normalized_hash_overlap_count": len(overlap),
        "example_hashes": sorted(overlap)[:10],
    }
    if overlap:
        message = f"Civil augmentation pool overlaps Civil external test: {len(overlap)} normalized hashes."
        warnings.append(message)
        if fail_on_leakage:
            raise ValidationError(message)
    return result, warnings

