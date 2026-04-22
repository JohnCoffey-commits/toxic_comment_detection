"""Initial raw data audit computations."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from .text import (
    contains_all_caps_token,
    contains_digits,
    contains_email,
    contains_repeated_punctuation,
    contains_url,
    normalize_for_duplicate,
    word_count,
)


def _series_stats(series: pd.Series) -> dict[str, Any]:
    desc = series.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
    keys = ("count", "mean", "std", "min", "25%", "50%", "75%", "90%", "95%", "max")
    return {key: float(desc.get(key, 0.0)) for key in keys}


def _class_distribution(labels: pd.Series) -> dict[str, Any]:
    clean = labels.dropna().astype(int)
    counts = clean.value_counts().sort_index()
    total = int(counts.sum())
    positive = int(counts.get(1, 0))
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "positive_ratio": positive / total if total else None,
    }


def audit_dataset(
    df: pd.DataFrame,
    source: str,
    text_column: str,
    label_columns: Sequence[str],
    binary_labels: pd.Series | None = None,
    duplicate_lowercase: bool = True,
) -> dict[str, Any]:
    """Compute machine-readable raw audit metrics for a dataset."""

    text = df[text_column]
    text_as_str = text.fillna("").astype(str)
    normalized = text_as_str.map(lambda value: normalize_for_duplicate(value, duplicate_lowercase))
    char_lengths = text_as_str.map(len)
    word_lengths = text_as_str.map(word_count)
    total = len(df)
    missing_label_count = int(df[list(label_columns)].isna().any(axis=1).sum()) if label_columns else 0

    audit: dict[str, Any] = {
        "source_dataset": source,
        "total_row_count": int(total),
        "missing_null_text_count": int(text.isna().sum()),
        "missing_label_count": missing_label_count,
        "empty_string_text_count": int(text_as_str.str.strip().eq("").sum()),
        "exact_duplicate_count": int(text.duplicated(keep="first").sum()),
        "normalized_duplicate_count": int(normalized.duplicated(keep="first").sum()),
        "character_length_statistics": _series_stats(char_lengths),
        "word_length_statistics": _series_stats(word_lengths),
        "feature_percentages": {
            "urls": float(text_as_str.map(contains_url).mean()) if total else 0.0,
            "emails": float(text_as_str.map(contains_email).mean()) if total else 0.0,
            "digits": float(text_as_str.map(contains_digits).mean()) if total else 0.0,
            "repeated_punctuation": float(text_as_str.map(contains_repeated_punctuation).mean()) if total else 0.0,
            "all_caps_tokens": float(text_as_str.map(contains_all_caps_token).mean()) if total else 0.0,
        },
    }
    if binary_labels is not None:
        audit["class_distribution"] = _class_distribution(binary_labels)
    return audit

