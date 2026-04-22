"""DistilBERT token length diagnostics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def add_bert_token_lengths(
    df: pd.DataFrame,
    tokenizer_name: str,
    batch_size: int,
    require_tokenizer: bool,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Add DistilBERT token lengths and return diagnostics."""

    warnings: list[str] = []
    out = df.copy()
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except Exception as exc:
        message = (
            f"Could not load DistilBERT tokenizer '{tokenizer_name}': {exc}. "
            "Install transformers and ensure the tokenizer is available."
        )
        if require_tokenizer:
            raise RuntimeError(message) from exc
        LOGGER.warning(message)
        warnings.append(message)
        out["bert_token_len"] = np.nan
        return out, {"available": False, "tokenizer_name": tokenizer_name}, warnings

    lengths: list[int] = []
    texts = out["text_clean"].fillna("").astype(str).tolist()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    out["bert_token_len"] = lengths
    diagnostics = token_length_diagnostics(out["bert_token_len"], tokenizer_name)
    return out, diagnostics, warnings


def token_length_diagnostics(lengths: pd.Series, tokenizer_name: str) -> dict[str, Any]:
    """Compute requested token-length metrics."""

    clean = pd.to_numeric(lengths, errors="coerce").dropna()
    if clean.empty:
        return {"available": False, "tokenizer_name": tokenizer_name}
    return {
        "available": True,
        "tokenizer_name": tokenizer_name,
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "fraction_gt_128": float((clean > 128).mean()),
        "fraction_gt_256": float((clean > 256).mean()),
    }

