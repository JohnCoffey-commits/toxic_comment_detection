# Data Analysis Report

## 1. Report Scope

This report summarizes the current preprocessing and data-preparation state of the `toxic_comment_detection` repository after the stratified-split update. It is based on the current generated artifacts in:

- `reports/preprocessing/preprocessing_report.md`
- `reports/preprocessing/preprocessing_report.json`
- `metadata/*.json`
- `metadata/preprocessing_config.yaml`
- `metadata/data_dictionary.md`
- `data/processed/`
- `src/preprocessing/`

This report does not summarize model performance. Model training outputs and evaluation results are not available in the current repository artifacts.

## 2. Data Sources Processed

The current generated preprocessing artifacts are from a **Jigsaw-only run**.

| Dataset | Raw/input status in current run | Current role |
| --- | --- | --- |
| Jigsaw Toxic Comment Classification Challenge | `dataset/jigsaw/raw/train.csv` | Mandatory main-experiment dataset |
| Civil Comments | `missing_optional` in the current generated preprocessing report | Optional extension dataset, not processed in the current artifact set |

Civil Comments remains supported by the Python pipeline when enabled and present, but it is optional and disabled by default in the Colab notebook. No current Civil processed outputs are present in `data/processed/`.

## 3. What Preprocessing Work Has Been Completed

The current repository implements and has generated artifacts for the following preprocessing stages:

- Raw CSV loading with pandas.
- Schema validation for required text and label columns.
- Raw data audit summaries.
- Binary label harmonization.
- Light text normalization.
- Model-specific text views:
  - `raw_text`
  - `text_clean`
  - `text_tfidf`
- Deterministic stratified split generation based on `binary_label`.
- Challenge slice construction:
  - identity slice
  - obfuscation slice
  - implicit toxicity proxy slice
  - length buckets
- DistilBERT token-length diagnostics using `distilbert-base-uncased`.
- Validation and quality checks:
  - missing cleaned text
  - missing labels
  - duplicate counts after removal
  - class distribution overall and by split
  - label ratio drift
  - raw-hash and normalized-hash split overlap
  - slice summaries
- Metadata generation.
- Markdown and JSON preprocessing report generation.
- Simplified teammate-facing Jigsaw CSV exports.

No model training was implemented or run as part of these preprocessing artifacts.

## 4. Actual Data Processing Outcomes

### Raw Audit Summary

Raw Jigsaw audit values from `metadata/data_audit_summary.json`:

| Dataset | Raw rows | Missing/null text | Missing labels | Empty-string text | Exact duplicates | Normalized duplicates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jigsaw | 159,571 | 0 | 0 | 0 | 0 | 258 |

Raw Jigsaw class distribution from the audit:

| Dataset | Non-toxic count | Toxic count | Positive ratio |
| --- | ---: | ---: | ---: |
| Jigsaw | 143,346 | 16,225 | 0.10167887648758234 |

Selected raw Jigsaw text-pattern audit percentages:

| Dataset | URLs | Emails | Digits | Repeated punctuation | All-caps tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jigsaw | 0.03191056018950812 | 0.0024315195116907207 | 0.3209355083317144 | 0.1018982145878637 | 0.34863477699582 |

### Filtering and Duplicate Removal

Values from `reports/preprocessing/preprocessing_report.json`:

| Dataset | Input rows | Removed missing text | Removed missing labels | Removed normalized duplicates | Retained rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jigsaw | 159,571 | 0 | 0 | 258 | 159,313 |

### Label Harmonization

Values from `metadata/label_mapping.json`:

| Dataset | Source label fields | Threshold | Mapping |
| --- | --- | ---: | --- |
| Jigsaw | `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` | Not applicable | `binary_label = 1` if any source toxicity column is greater than 0 |
| Civil Comments | Not processed in current run | 0.5 | Civil mapping remains documented for optional extension mode |

### Stratified Split Outcomes

Values from `metadata/split_summary.json` and validation metadata:

| Dataset | Split | Rows | Non-toxic | Toxic | Positive ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Jigsaw | train | 111,519 | 100,196 | 11,323 | 0.10153426770326132 |
| Jigsaw | val | 23,897 | 21,471 | 2,426 | 0.10151901912373938 |
| Jigsaw | test | 23,897 | 21,471 | 2,426 | 0.10151901912373938 |

Split method:

- Jigsaw: two-stage `StratifiedShuffleSplit`.
  - Stratification column: `binary_label`.
  - Stage 1 seed: 42.
  - Stage 2 seed: 43.
  - Target ratio: train 0.70, validation 0.15, test 0.15.

### Teammate-Facing CSV Exports

The pipeline now exports simplified Jigsaw-only CSV files:

| File | Rows | Columns |
| --- | ---: | --- |
| `data/processed/train.csv` | 111,519 | `raw_text`, `clean_text`, `label` |
| `data/processed/val.csv` | 23,897 | `raw_text`, `clean_text`, `label` |
| `data/processed/test.csv` | 23,897 | `raw_text`, `clean_text`, `label` |

The `clean_text` export is lowercase, removes URL/email placeholders and URL/email patterns, removes non-alphanumeric special characters, and collapses extra spaces. It does not apply stemming or lemmatization.

## 5. Validation Outcomes

### Missing Text and Labels After Preprocessing

Values from validation metadata:

| Dataset | Missing cleaned text | Missing labels after harmonization |
| --- | ---: | ---: |
| Jigsaw | 0 | 0 |

### Duplicate Counts After Removal

| Dataset | Raw-hash duplicates after removal | Normalized-hash duplicates after removal |
| --- | ---: | ---: |
| Jigsaw | 0 | 0 |

### Leakage Checks

All current Jigsaw split-overlap checks report zero overlap.

| Hash type | Split pair | Overlap count |
| --- | --- | ---: |
| Raw | test/train | 0 |
| Raw | test/val | 0 |
| Raw | train/val | 0 |
| Normalized | test/train | 0 |
| Normalized | test/val | 0 |
| Normalized | train/val | 0 |

### Label Ratio Drift

Absolute drift from each split's positive ratio to the overall Jigsaw positive ratio:

| Dataset | Split | Absolute drift |
| --- | --- | ---: |
| Jigsaw | train | 0.000004574583428024252 |
| Jigsaw | val | 0.000010673996093909608 |
| Jigsaw | test | 0.000010673996093909608 |

The current generated preprocessing report records one warning: Civil Comments was missing for the current run and optional extension artifacts were skipped.

### DistilBERT Token-Length Diagnostics

Tokenizer: `distilbert-base-uncased`.

| Dataset | Mean | Median | p90 | p95 | Fraction > 128 | Fraction > 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jigsaw | 93.64426004155342 | 51.0 | 204.0 | 307.0 | 0.18985895689617294 | 0.0678977861191491 |

### Slice Sizes and Label Distributions

Values from `reports/preprocessing/preprocessing_report.json`.

| Slice / bucket | Value | Rows | Non-toxic | Toxic | Positive ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `has_identity_term` | 0 | 149,967 | 135,600 | 14,367 | 0.09580107623677209 |
| `has_identity_term` | 1 | 9,346 | 7,538 | 1,808 | 0.19345174406163065 |
| `has_obfuscation` | 0 | 92,860 | 85,921 | 6,939 | 0.07472539306482877 |
| `has_obfuscation` | 1 | 66,453 | 57,217 | 9,236 | 0.13898544836200022 |
| `implicit_proxy` | 0 | 149,102 | 143,138 | 5,964 | 0.03999946345454789 |
| `implicit_proxy` | 1 | 10,211 | 0 | 10,211 | 1.0 |
| `length_bucket` | short | 53,794 | 45,844 | 7,950 | 0.14778599843848755 |
| `length_bucket` | medium | 51,590 | 46,915 | 4,675 | 0.0906183368869936 |
| `length_bucket` | long | 53,929 | 50,379 | 3,550 | 0.06582729143874354 |

Length-bucket thresholds from `metadata/slice_definitions.json`:

- `short`: `word_len <= 22`
- `medium`: `23 <= word_len <= 56`
- `long`: `word_len > 56`

## 6. Output Artifacts Generated

### Processed Dataset Files

Teammate-facing Jigsaw files:

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

Richer Jigsaw files:

- `data/processed/jigsaw_train.csv`
- `data/processed/jigsaw_train.parquet`
- `data/processed/jigsaw_val.csv`
- `data/processed/jigsaw_val.parquet`
- `data/processed/jigsaw_test.csv`
- `data/processed/jigsaw_test.parquet`

Jigsaw challenge slice files:

- `data/processed/jigsaw_identity_slice.csv`
- `data/processed/jigsaw_identity_slice.parquet`
- `data/processed/jigsaw_obfuscation_slice.csv`
- `data/processed/jigsaw_obfuscation_slice.parquet`
- `data/processed/jigsaw_implicit_proxy_slice.csv`
- `data/processed/jigsaw_implicit_proxy_slice.parquet`

Civil processed outputs are not present in the current regenerated artifact set because Civil was disabled/missing for this run.

### Metadata and Documentation Files

Metadata:

- `metadata/label_mapping.json`
- `metadata/split_summary.json`
- `metadata/data_audit_summary.json`
- `metadata/slice_definitions.json`
- `metadata/preprocessing_config.yaml`
- `metadata/data_dictionary.md`

Reports and handoff:

- `reports/preprocessing/preprocessing_report.md`
- `reports/preprocessing/preprocessing_report.json`
- `reports/preprocessing/README_preprocessing.md`

Team documentation:

- `docs/PROJECT_GUIDE.md`
- `docs/DATA_ANALYSIS_REPORT.md`

## 7. Interpretation of Current Preprocessing Status

The current preprocessing state appears ready for the main Jigsaw-only modeling stage:

- Jigsaw train/validation/test files exist in both CSV and Parquet formats.
- Simplified teammate-facing Jigsaw CSVs now exist with exactly `raw_text`, `clean_text`, and `label`.
- Jigsaw splits are stratified by `binary_label`.
- Missing cleaned text and missing labels are both zero in the retained processed dataset.
- Duplicate hash counts after removal are zero.
- Raw and normalized hash leakage checks across saved Jigsaw splits report zero overlap.
- Split label-ratio drift values are very small after stratification.

Civil Comments artifacts are optional extension outputs. They were removed from the current processed artifact set because the current regenerated run is Jigsaw-only.

No model accuracy, training-time, calibration, fairness, or robustness results are available in the current repository artifacts.

## 8. Assumptions and Caveats

Assumptions visible in current code and reports:

- Jigsaw is the only mandatory dataset.
- Civil Comments is optional and extension-oriented.
- The main non-extension experiment uses only Jigsaw processed outputs.
- Jigsaw `identity_hate` is treated as an original toxicity label, not as a dataset-provided identity mention column.
- Jigsaw identity slicing uses a transparent rule-based lexicon.
- Obfuscation slicing uses rule-based heuristics.
- `implicit_proxy` is a proxy, not a gold implicit-toxicity label.
- Length buckets use empirical Jigsaw word-length quantiles.
- URLs and emails are replaced with placeholders in the richer internal `text_clean` field.
- The simplified teammate-facing `clean_text` field applies additional lowercase and special-character cleanup for CSV handoff.
- The preprocessing configuration is deterministic with fixed random seeds.

Caveats:

- The current generated metadata contains `civil_comments` label-mapping information because Civil support remains implemented, but Civil was not processed in this regenerated artifact set.
- The current `preprocessing_config.yaml` records the disabled Civil path used for the Jigsaw-only run.
- Unit tests for the preprocessing code are not available in the inspected repository file tree.
- Challenge slices should be reviewed before being treated as definitive analytical categories.

## 9. Recommended Next Steps for the Team

1. Use `data/processed/train.csv`, `data/processed/val.csv`, and `data/processed/test.csv` for the simplest teammate-facing modeling handoff.
2. Use the richer `jigsaw_*` files when model code needs hashes, slice flags, token lengths, or other metadata.
3. Keep Civil Comments outputs out of the non-extension experiment unless explicitly running augmentation, external evaluation, or robustness work.
4. Use challenge slice files after predictions are available to support error analysis.
5. Add focused tests for schema validation, stratified split construction, duplicate removal, leakage checks, teammate export schema, and slice construction.
6. Review identity, obfuscation, and implicit-proxy heuristics before using slice results for high-stakes conclusions.

