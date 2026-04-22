"""Label harmonization and standardized row construction."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .config import JIGSAW_LABEL_COLUMNS


def _compact_json(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def jigsaw_binary_labels(df: pd.DataFrame) -> pd.Series:
    """Map Jigsaw multi-label toxicity columns to binary toxicity."""

    labels = df[list(JIGSAW_LABEL_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return labels.gt(0).any(axis=1).astype("Int64")


def civil_binary_labels(df: pd.DataFrame, label_column: str, threshold: float) -> pd.Series:
    """Map Civil Comments toxicity score to binary toxicity."""

    scores = pd.to_numeric(df[label_column], errors="coerce")
    return scores.ge(threshold).astype("Int64")


def jigsaw_orig_label_info(row: pd.Series) -> str:
    """Serialize original Jigsaw labels for auditability."""

    return _compact_json({col: int(row[col]) if pd.notna(row[col]) else None for col in JIGSAW_LABEL_COLUMNS})


def civil_score_label_info(score: object, label_column: str) -> str:
    """Serialize the Civil toxicity score without duplicating identity columns."""

    return _compact_json({label_column: float(score) if pd.notna(score) else None})
