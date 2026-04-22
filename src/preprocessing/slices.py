"""Challenge slice construction."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .text import has_explicit_toxic_term, has_identity_term, has_obfuscation


def add_identity_column(
    df: pd.DataFrame,
    identity_terms: Sequence[str],
    identity_columns: Sequence[str] | None = None,
    identity_threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add has_identity_term using provided identity score columns when available."""

    out = df.copy()
    available = [col for col in identity_columns or [] if col in out.columns]
    if available:
        identity_scores = out[available].apply(pd.to_numeric, errors="coerce")
        out["has_identity_term"] = identity_scores.ge(identity_threshold).any(axis=1).astype(int)
        method = "dataset_identity_columns"
    else:
        out["has_identity_term"] = out["text_clean"].map(lambda value: has_identity_term(value, identity_terms)).astype(int)
        method = "rule_based_identity_lexicon"
    return out, {
        "column": "has_identity_term",
        "method": method,
        "identity_columns_used": available,
        "identity_threshold": identity_threshold if available else None,
        "lexicon_terms": list(identity_terms) if not available else [],
    }


def add_obfuscation_column(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add has_obfuscation with documented character-level heuristics."""

    out = df.copy()
    out["has_obfuscation"] = out["text_clean"].map(has_obfuscation).astype(int)
    return out, {
        "column": "has_obfuscation",
        "method": "rule_based_obfuscation_heuristics",
        "heuristics": [
            "symbols inserted inside words",
            "leetspeak-like substitutions",
            "three or more repeated alphabetic characters",
            "split abusive words with punctuation or symbols between characters",
        ],
    }


def add_implicit_proxy_column(df: pd.DataFrame, explicit_terms: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add proxy for toxic comments without explicit terms."""

    out = df.copy()
    has_explicit = out["text_clean"].map(lambda value: has_explicit_toxic_term(value, explicit_terms))
    out["implicit_proxy"] = ((out["binary_label"].astype(int) == 1) & (~has_explicit)).astype(int)
    return out, {
        "column": "implicit_proxy",
        "method": "binary_label == 1 and no explicit term from a small transparent profanity lexicon",
        "is_gold_label": False,
        "explicit_lexicon_terms": list(explicit_terms),
    }


def compute_length_thresholds(df: pd.DataFrame) -> dict[str, int]:
    """Choose length buckets from empirical Jigsaw word-length quantiles."""

    q33, q66 = df["word_len"].quantile([0.33, 0.66]).tolist()
    short_max = max(1, int(np.ceil(q33)))
    medium_max = max(short_max + 1, int(np.ceil(q66)))
    return {"short_max_word_len": short_max, "medium_max_word_len": medium_max}


def add_length_bucket(df: pd.DataFrame, thresholds: dict[str, int]) -> pd.DataFrame:
    """Add short/medium/long buckets from saved thresholds."""

    out = df.copy()
    short_max = thresholds["short_max_word_len"]
    medium_max = thresholds["medium_max_word_len"]
    out["length_bucket"] = pd.cut(
        out["word_len"],
        bins=[-1, short_max, medium_max, np.inf],
        labels=["short", "medium", "long"],
    ).astype(str)
    return out


def slice_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize slice sizes and label distributions."""

    summary: dict[str, Any] = {}
    for column in ("has_identity_term", "has_obfuscation", "implicit_proxy", "length_bucket"):
        if column not in df.columns:
            continue
        groups: dict[str, Any] = {}
        for value, group in df.groupby(column, dropna=False):
            labels = group["binary_label"].astype(int)
            groups[str(value)] = {
                "row_count": int(len(group)),
                "positive_ratio": float(labels.mean()) if len(group) else None,
                "label_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
            }
        summary[column] = groups
    return summary

