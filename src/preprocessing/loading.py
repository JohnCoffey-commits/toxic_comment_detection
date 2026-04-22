"""Raw CSV loading and schema validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    CIVIL_IDENTITY_COLUMNS,
    CIVIL_LABEL_CANDIDATES,
    CIVIL_TEXT_CANDIDATES,
    JIGSAW_LABEL_COLUMNS,
    JIGSAW_TEXT_COLUMN,
)

LOGGER = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Raised when an input file does not match the expected minimum schema."""


def _read_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _missing_error(source: str, path: Path, found: list[str], expected: list[str]) -> SchemaError:
    return SchemaError(
        f"{source} schema validation failed for {path}. "
        f"Expected columns: {expected}. Found columns: {found}."
    )


def load_jigsaw(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load mandatory Jigsaw data with required labels."""

    if not path.exists():
        raise FileNotFoundError(
            f"Mandatory Jigsaw raw file is missing: {path}. "
            "Expected dataset/jigsaw/raw/train.csv."
        )
    found = _read_columns(path)
    expected = [JIGSAW_TEXT_COLUMN, *JIGSAW_LABEL_COLUMNS]
    missing = [col for col in expected if col not in found]
    if missing:
        raise _missing_error("Jigsaw", path, found, expected)

    usecols = [col for col in ("id", JIGSAW_TEXT_COLUMN, *JIGSAW_LABEL_COLUMNS) if col in found]
    df = pd.read_csv(path, usecols=usecols)
    schema = {
        "source_dataset": "jigsaw",
        "path": str(path),
        "found_columns": found,
        "text_column": JIGSAW_TEXT_COLUMN,
        "label_columns": list(JIGSAW_LABEL_COLUMNS),
        "ignored_files_note": "Only dataset/jigsaw/raw/train.csv is used; Kaggle test/submission files are ignored.",
    }
    LOGGER.info("Loaded Jigsaw rows=%s columns=%s", len(df), list(df.columns))
    return df, schema


def load_civil_optional(path: Path) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Load optional Civil Comments data if available."""

    if not path.exists():
        LOGGER.warning(
            "Optional Civil Comments raw file is missing at %s; continuing in Jigsaw-only mode.",
            path,
        )
        return None, {
            "source_dataset": "civil_comments",
            "path": str(path),
            "present": False,
            "warning": "Civil Comments file missing; optional extension artifacts skipped.",
        }

    found = _read_columns(path)
    text_col = next((col for col in CIVIL_TEXT_CANDIDATES if col in found), None)
    label_col = next((col for col in CIVIL_LABEL_CANDIDATES if col in found), None)
    expected = [f"one of {list(CIVIL_TEXT_CANDIDATES)}", f"one of {list(CIVIL_LABEL_CANDIDATES)}"]
    if text_col is None or label_col is None:
        raise _missing_error("Civil Comments", path, found, expected)

    identity_cols = [col for col in CIVIL_IDENTITY_COLUMNS if col in found]
    usecols = [col for col in ("id", text_col, label_col, *identity_cols) if col in found]
    df = pd.read_csv(path, usecols=usecols)
    schema = {
        "source_dataset": "civil_comments",
        "path": str(path),
        "present": True,
        "found_columns": found,
        "text_column": text_col,
        "label_column": label_col,
        "identity_columns": identity_cols,
        "parent_text_used": False,
        "ignored_files_note": "Only dataset/civil_comments/raw/train.csv is used; parent_text and unused files are ignored.",
    }
    LOGGER.info("Loaded Civil Comments rows=%s columns=%s", len(df), list(df.columns))
    return df, schema

