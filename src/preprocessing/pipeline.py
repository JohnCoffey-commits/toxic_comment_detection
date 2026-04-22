"""End-to-end preprocessing entry point.

This module implements Phase A through Phase I for the toxic comment
classification project. It deliberately performs no model training.
"""

from __future__ import annotations

import argparse
import html
import logging
import re
import unicodedata
from typing import Any

import pandas as pd

from .audit import audit_dataset
from .config import (
    JIGSAW_LABEL_COLUMNS,
    PipelineConfig,
)
from .harmonize import (
    civil_binary_labels,
    civil_score_label_info,
    jigsaw_binary_labels,
    jigsaw_orig_label_info,
)
from .io_utils import ensure_output_dirs, save_frame, write_json, write_yaml
from .loading import load_civil_optional, load_jigsaw
from .reporting import (
    build_data_dictionary,
    build_preprocessing_report,
    build_readme_preprocessing,
    write_markdown,
)
from .slices import (
    add_identity_column,
    add_implicit_proxy_column,
    add_length_bucket,
    add_obfuscation_column,
    compute_length_thresholds,
    slice_summary,
)
from .splits import assign_civil_splits, assign_jigsaw_splits
from .text import clean_text, normalize_for_duplicate, sha256_text, word_count
from .tokenization import add_bert_token_lengths
from .validation import validate_civil_aug_external_overlap, validate_dataset

LOGGER = logging.getLogger(__name__)
EXPORT_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
EXPORT_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
EXPORT_SPECIAL_RE = re.compile(r"[^a-z0-9\s]")
EXPORT_WHITESPACE_RE = re.compile(r"\s+")


def setup_logging() -> None:
    """Configure simple stderr logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _standardize_common(
    df: pd.DataFrame,
    source: str,
    text_column: str,
    binary_label: pd.Series,
    orig_label_info: pd.Series,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, int], pd.Index]:
    """Create standardized columns and remove invalid/duplicate rows."""

    working = pd.DataFrame(
        {
            "__source_row_index": df.index,
            "source_dataset": source,
            "raw_text": df[text_column],
            "binary_label": binary_label,
            "orig_label_info": orig_label_info,
        }
    )
    input_rows = len(working)
    missing_text = working["raw_text"].isna() | working["raw_text"].astype(str).str.strip().eq("")
    removed_missing_text = int(missing_text.sum())
    if config.remove_missing_text:
        working = working.loc[~missing_text].copy()

    missing_labels = working["binary_label"].isna()
    removed_missing_labels = int(missing_labels.sum())
    if config.remove_missing_labels:
        working = working.loc[~missing_labels].copy()

    working["text_clean"] = working["raw_text"].map(
        lambda value: clean_text(
            value,
            replace_urls=config.replace_urls,
            replace_emails=config.replace_emails,
            url_replacement=config.url_replacement,
            email_replacement=config.email_replacement,
        )
    )
    empty_after_clean = working["text_clean"].astype(str).str.strip().eq("")
    if config.remove_missing_text:
        working = working.loc[~empty_after_clean].copy()

    working["text_tfidf"] = working["text_clean"].str.casefold()
    working["char_len"] = working["text_clean"].map(len).astype(int)
    working["word_len"] = working["text_clean"].map(word_count).astype(int)
    working["text_hash_raw"] = working["raw_text"].map(sha256_text)
    working["text_hash_normalized"] = working["raw_text"].map(
        lambda value: sha256_text(normalize_for_duplicate(value, config.duplicate_lowercase))
    )

    duplicate_mask = working["text_hash_normalized"].duplicated(keep="first")
    removed_duplicates = int(duplicate_mask.sum())
    if config.remove_normalized_duplicates:
        working = working.loc[~duplicate_mask].copy()

    working["binary_label"] = working["binary_label"].astype(int)
    retained_source_index = pd.Index(working["__source_row_index"].tolist())
    working = working.reset_index(drop=True)
    working["sample_id"] = [f"{source}_{idx:08d}" for idx in range(len(working))]
    ordered_columns = [
        "sample_id",
        "source_dataset",
        "raw_text",
        "text_clean",
        "text_tfidf",
        "binary_label",
        "orig_label_info",
        "char_len",
        "word_len",
        "text_hash_raw",
        "text_hash_normalized",
    ]
    removal_summary = {
        "input_rows": int(input_rows),
        "removed_missing_text": int(removed_missing_text + empty_after_clean.sum()),
        "removed_missing_labels": removed_missing_labels,
        "removed_normalized_duplicates": removed_duplicates,
        "output_rows": int(len(working)),
    }
    return working[ordered_columns], removal_summary, retained_source_index


def _prepare_jigsaw(
    raw: pd.DataFrame,
    schema: dict[str, Any],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, int]]:
    labels = jigsaw_binary_labels(raw)
    audit = audit_dataset(
        raw,
        "jigsaw",
        schema["text_column"],
        JIGSAW_LABEL_COLUMNS,
        labels,
        config.duplicate_lowercase,
    )
    standardized, removal, _ = _standardize_common(
        raw,
        "jigsaw",
        schema["text_column"],
        labels,
        raw.apply(jigsaw_orig_label_info, axis=1),
        config,
    )
    return standardized, audit, removal


def _prepare_civil(
    raw: pd.DataFrame,
    schema: dict[str, Any],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, int]]:
    label_column = schema["label_column"]
    labels = civil_binary_labels(raw, label_column, config.civil_threshold)
    audit = audit_dataset(
        raw,
        "civil_comments",
        schema["text_column"],
        [label_column],
        labels,
        config.duplicate_lowercase,
    )
    standardized, removal, retained_source_index = _standardize_common(
        raw,
        "civil_comments",
        schema["text_column"],
        labels,
        raw[label_column].map(lambda score: civil_score_label_info(score, label_column)),
        config,
    )
    for col in schema.get("identity_columns", []):
        standardized[col] = pd.to_numeric(raw.loc[retained_source_index, col], errors="coerce").reset_index(drop=True)
    return standardized, audit, removal


def _save_jigsaw_outputs(df: pd.DataFrame, config: PipelineConfig) -> None:
    for split in ("train", "val", "test"):
        split_df = df.loc[df["split"].eq(split)].copy()
        save_frame(
            split_df,
            config.processed_abs / f"jigsaw_{split}.csv",
            config.processed_abs / f"jigsaw_{split}.parquet",
        )
    save_frame(
        df.loc[df["has_identity_term"].eq(1)].copy(),
        config.processed_abs / "jigsaw_identity_slice.csv",
        config.processed_abs / "jigsaw_identity_slice.parquet",
    )
    save_frame(
        df.loc[df["has_obfuscation"].eq(1)].copy(),
        config.processed_abs / "jigsaw_obfuscation_slice.csv",
        config.processed_abs / "jigsaw_obfuscation_slice.parquet",
    )
    save_frame(
        df.loc[df["implicit_proxy"].eq(1)].copy(),
        config.processed_abs / "jigsaw_implicit_proxy_slice.csv",
        config.processed_abs / "jigsaw_implicit_proxy_slice.parquet",
    )


def _teammate_clean_text(value: object) -> str:
    """Create the simplified teammate-facing clean_text export field."""

    raw = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", html.unescape(raw)).lower()
    text = EXPORT_URL_RE.sub(" ", text)
    text = EXPORT_EMAIL_RE.sub(" ", text)
    text = text.replace("<url>", " ").replace("<email>", " ")
    text = EXPORT_SPECIAL_RE.sub(" ", text)
    return EXPORT_WHITESPACE_RE.sub(" ", text).strip()


def _save_teammate_exports(df: pd.DataFrame, config: PipelineConfig) -> None:
    """Save simplified Jigsaw-only CSVs for teammate modeling handoff."""

    for split in ("train", "val", "test"):
        split_df = df.loc[df["split"].eq(split), ["raw_text", "text_clean", "binary_label"]].copy()
        export = pd.DataFrame(
            {
                "raw_text": split_df["raw_text"],
                "clean_text": split_df["text_clean"].map(_teammate_clean_text),
                "label": split_df["binary_label"].astype(int),
            }
        )
        export.to_csv(config.processed_abs / f"{split}.csv", index=False)


def _save_civil_outputs(df: pd.DataFrame, jigsaw_train_rows: int, config: PipelineConfig) -> dict[str, Any]:
    aug_pool = df.loc[df["split"].eq("aug_pool")].copy()
    external = df.loc[df["split"].eq("external_test")].copy()
    capped_n = min(jigsaw_train_rows, len(aug_pool))
    capped = aug_pool.sample(n=capped_n, random_state=config.random_seed + 4) if capped_n else aug_pool.head(0)
    identity = aug_pool.loc[aug_pool["has_identity_term"].eq(1)].copy()

    save_frame(aug_pool, config.processed_abs / "civil_aug_pool_full.csv", config.processed_abs / "civil_aug_pool_full.parquet")
    save_frame(external, config.processed_abs / "civil_external_test.csv", config.processed_abs / "civil_external_test.parquet")
    save_frame(
        capped,
        config.processed_abs / "civil_aug_pool_capped_match_jigsaw_train.csv",
        config.processed_abs / "civil_aug_pool_capped_match_jigsaw_train.parquet",
    )
    save_frame(
        identity,
        config.processed_abs / "civil_aug_pool_identity.csv",
        config.processed_abs / "civil_aug_pool_identity.parquet",
    )
    save_frame(
        df.loc[df["has_obfuscation"].eq(1)].copy(),
        config.processed_abs / "civil_obfuscation_slice.csv",
        config.processed_abs / "civil_obfuscation_slice.parquet",
    )
    save_frame(
        df.loc[df["implicit_proxy"].eq(1)].copy(),
        config.processed_abs / "civil_implicit_proxy_slice.csv",
        config.processed_abs / "civil_implicit_proxy_slice.parquet",
    )
    return {
        "civil_aug_pool_full": int(len(aug_pool)),
        "civil_external_test": int(len(external)),
        "civil_aug_pool_capped_match_jigsaw_train": int(len(capped)),
        "civil_aug_pool_identity": int(len(identity)),
    }


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Run Phase A through Phase I and return the report payload."""

    ensure_output_dirs(config.processed_abs, config.metadata_abs, config.report_abs)
    warnings: list[str] = []

    jigsaw_raw, jigsaw_schema = load_jigsaw(config.jigsaw_raw_abs)
    civil_raw, civil_schema = load_civil_optional(config.civil_raw_abs)
    if not civil_schema.get("present", True):
        warnings.append(civil_schema["warning"])

    jigsaw, jigsaw_audit, jigsaw_removal = _prepare_jigsaw(jigsaw_raw, jigsaw_schema, config)
    civil = None
    civil_audit = None
    civil_removal = None
    if civil_raw is not None:
        civil, civil_audit, civil_removal = _prepare_civil(civil_raw, civil_schema, config)

    jigsaw, token_stats_jigsaw, token_warnings = add_bert_token_lengths(
        jigsaw,
        config.tokenizer_name,
        config.tokenizer_batch_size,
        config.require_transformers_tokenizer,
    )
    warnings.extend(token_warnings)
    if civil is not None:
        civil, _, token_warnings = add_bert_token_lengths(
            civil,
            config.tokenizer_name,
            config.tokenizer_batch_size,
            config.require_transformers_tokenizer,
        )
        warnings.extend(token_warnings)

    jigsaw, identity_def = add_identity_column(jigsaw, config.identity_terms)
    jigsaw, obfuscation_def = add_obfuscation_column(jigsaw)
    jigsaw, implicit_def = add_implicit_proxy_column(jigsaw, config.explicit_toxic_terms)
    length_thresholds = compute_length_thresholds(jigsaw)
    jigsaw = add_length_bucket(jigsaw, length_thresholds)

    civil_defs: dict[str, Any] = {}
    if civil is not None:
        civil, civil_identity_def = add_identity_column(
            civil,
            config.identity_terms,
            civil_schema.get("identity_columns", []),
            config.civil_identity_threshold,
        )
        civil, civil_obfuscation_def = add_obfuscation_column(civil)
        civil, civil_implicit_def = add_implicit_proxy_column(civil, config.explicit_toxic_terms)
        civil = add_length_bucket(civil, length_thresholds)
        civil_defs = {
            "identity": civil_identity_def,
            "obfuscation": civil_obfuscation_def,
            "implicit_proxy": civil_implicit_def,
        }

    jigsaw, jigsaw_split_summary = assign_jigsaw_splits(
        jigsaw,
        config.test_size,
        config.val_size,
        config.random_seed,
    )
    civil_split_summary = None
    if civil is not None:
        civil, civil_split_summary = assign_civil_splits(
            civil,
            config.civil_external_test_size,
            config.random_seed + 2,
        )

    jigsaw_validation, validation_warnings = validate_dataset(
        jigsaw,
        "jigsaw",
        config.label_drift_warning_threshold,
        config.suspicious_split_min_rows,
        config.fail_on_leakage,
    )
    jigsaw_validation["distilbert_token_length_diagnostics"] = token_stats_jigsaw
    warnings.extend(validation_warnings)

    validation = {"jigsaw": jigsaw_validation}
    civil_overlap = None
    if civil is not None:
        civil_validation, validation_warnings = validate_dataset(
            civil,
            "civil_comments",
            config.label_drift_warning_threshold,
            config.suspicious_split_min_rows,
            config.fail_on_leakage,
        )
        validation["civil_comments"] = civil_validation
        warnings.extend(validation_warnings)
        civil_overlap, overlap_warnings = validate_civil_aug_external_overlap(
            civil,
            config.fail_on_leakage,
        )
        warnings.extend(overlap_warnings)

    split_summary = {"jigsaw": jigsaw_split_summary}
    if civil_split_summary is not None:
        split_summary["civil_comments"] = civil_split_summary

    audits = {"jigsaw": jigsaw_audit}
    duplicate_removal = {"jigsaw": jigsaw_removal}
    if civil_audit is not None:
        audits["civil_comments"] = civil_audit
    if civil_removal is not None:
        duplicate_removal["civil_comments"] = civil_removal

    slice_definitions = {
        "identity": identity_def,
        "obfuscation": obfuscation_def,
        "implicit_proxy": implicit_def,
        "length_bucket": {
            "method": "empirical Jigsaw word-length quantiles",
            "thresholds": length_thresholds,
        },
        "civil_comments": civil_defs,
    }
    slice_summaries = {"jigsaw": slice_summary(jigsaw)}
    if civil is not None:
        slice_summaries["civil_comments"] = slice_summary(civil)

    label_mapping = {
        "jigsaw": {
            "source_columns_used": list(JIGSAW_LABEL_COLUMNS),
            "threshold_used": None,
            "mapping_logic": "binary_label = 1 if any source toxicity column is > 0 else 0",
        },
        "civil_comments": {
            "present": civil is not None,
            "source_column_used": civil_schema.get("label_column"),
            "threshold_used": config.civil_threshold,
            "mapping_logic": "binary_label = 1 if toxicity score >= threshold else 0",
        },
    }

    _save_jigsaw_outputs(jigsaw, config)
    _save_teammate_exports(jigsaw, config)
    civil_output_counts = None
    if civil is not None:
        civil_output_counts = _save_civil_outputs(
            civil,
            int((jigsaw["split"] == "train").sum()),
            config,
        )

    write_json(config.metadata_abs / "label_mapping.json", label_mapping)
    write_json(config.metadata_abs / "split_summary.json", split_summary)
    write_json(config.metadata_abs / "data_audit_summary.json", audits)
    write_json(config.metadata_abs / "slice_definitions.json", slice_definitions)
    write_yaml(config.metadata_abs / "preprocessing_config.yaml", config.to_serializable_dict())
    write_markdown(config.metadata_abs / "data_dictionary.md", build_data_dictionary())

    report_payload = {
        "raw_files": {
            "jigsaw": str(config.jigsaw_raw_abs),
            "civil_comments": str(config.civil_raw_abs) if civil is not None else "missing_optional",
        },
        "schema": {"jigsaw": jigsaw_schema, "civil_comments": civil_schema},
        "data_audit_summary": audits,
        "duplicate_removal": duplicate_removal,
        "split_summary": split_summary,
        "label_mapping": label_mapping,
        "slice_definitions": slice_definitions,
        "slice_summary": slice_summaries,
        "validation": validation,
        "civil_aug_external_overlap": civil_overlap,
        "civil_output_counts": civil_output_counts,
        "teammate_exports": {
            split: {
                "path": str(config.processed_abs / f"{split}.csv"),
                "columns": ["raw_text", "clean_text", "label"],
                "rows": int((jigsaw["split"] == split).sum()),
            }
            for split in ("train", "val", "test")
        },
        "warnings": warnings,
    }
    write_markdown(config.report_abs / "preprocessing_report.md", build_preprocessing_report(report_payload))
    write_json(config.report_abs / "preprocessing_report.json", report_payload)
    write_markdown(config.report_abs / "README_preprocessing.md", build_readme_preprocessing())
    LOGGER.info("Preprocessing complete. Outputs written under %s", config.project_root)
    return report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run toxic-comment preprocessing.")
    parser.add_argument(
        "--allow-missing-tokenizer",
        action="store_true",
        help="Continue without DistilBERT token lengths if transformers/model loading fails.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = PipelineConfig(require_transformers_tokenizer=not args.allow_missing_tokenizer)
    run_pipeline(config)


if __name__ == "__main__":
    main()
