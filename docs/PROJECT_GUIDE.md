# Project Guide

## 1. Project Overview

This repository currently contains a preprocessing-focused toxic comment detection project for binary moderation triage: toxic vs non-toxic.

The current project state is organized around two datasets:

- **Jigsaw Toxic Comment Classification Challenge** is the mandatory main dataset.
- **Civil Comments** is optional and is reserved for extension work such as controlled augmentation, robustness analysis, or external evaluation.

The current documentation and generated artifacts cover the preprocessing/data-preparation stage. Model training code, model evaluation outputs, and comparative model results are not available in the current repository artifacts inspected for this guide.

## 2. Current Repository Structure

Clean tree of the relevant current repository contents:

```text
toxic_comment_detection/
├── data/
│   └── processed/
│       ├── jigsaw_identity_slice.csv
│       ├── jigsaw_identity_slice.parquet
│       ├── jigsaw_implicit_proxy_slice.csv
│       ├── jigsaw_implicit_proxy_slice.parquet
│       ├── jigsaw_obfuscation_slice.csv
│       ├── jigsaw_obfuscation_slice.parquet
│       ├── jigsaw_test.csv
│       ├── jigsaw_test.parquet
│       ├── jigsaw_train.csv
│       ├── jigsaw_train.parquet
│       ├── jigsaw_val.csv
│       ├── jigsaw_val.parquet
│       ├── test.csv
│       ├── train.csv
│       └── val.csv
├── dataset/
│   ├── civil_comments/
│   │   ├── raw/
│   │   │   └── train.csv
│   │   └── unused/
│   │       ├── all_data.csv
│   │       ├── identity_individual_annotations.csv
│   │       ├── sample_submission.csv
│   │       ├── test.csv
│   │       ├── test_private_expanded.csv
│   │       ├── test_public_expanded.csv
│   │       └── toxicity_individual_annotations.csv
│   └── jigsaw/
│       ├── raw/
│       │   └── train.csv
│       └── unused/
│           ├── sample_submission.csv.zip
│           ├── test.csv.zip
│           └── test_labels.csv.zip
├── docs/
│   ├── DATA_ANALYSIS_REPORT.md
│   └── PROJECT_GUIDE.md
├── metadata/
│   ├── data_audit_summary.json
│   ├── data_dictionary.md
│   ├── label_mapping.json
│   ├── preprocessing_config.yaml
│   ├── slice_definitions.json
│   └── split_summary.json
├── reports/
│   └── preprocessing/
│       ├── README_preprocessing.md
│       ├── preprocessing_report.json
│       └── preprocessing_report.md
├── requirements.txt
└── src/
    └── preprocessing/
        ├── __init__.py
        ├── audit.py
        ├── config.py
        ├── harmonize.py
        ├── io_utils.py
        ├── loading.py
        ├── pipeline.py
        ├── reporting.py
        ├── slices.py
        ├── splits.py
        ├── text.py
        ├── tokenization.py
        └── validation.py
```

Role of each major directory:

- `src/preprocessing/`: Python preprocessing package. This is the implementation used to load raw data, standardize schema, clean text, create splits, build challenge slices, validate outputs, and write reports.
- `dataset/`: Raw and unused source data files. The current pipeline uses only `dataset/jigsaw/raw/train.csv` and optionally `dataset/civil_comments/raw/train.csv`.
- `data/processed/`: Generated model-ready and analysis-ready datasets in CSV and Parquet formats.
- `metadata/`: Machine-readable preprocessing configuration, label mapping, split summary, audit summary, slice definitions, and data dictionary.
- `reports/preprocessing/`: Human-readable preprocessing handoff/report files and a JSON report.
- `docs/`: Team-facing project and data documentation.
- `requirements.txt`: Python runtime dependencies for the preprocessing pipeline.

The repository also contains `.DS_Store` files. They are filesystem metadata and are not project artifacts.

## 3. Main Preprocessing Entry Point

The preprocessing orchestrator is:

```text
src/preprocessing/pipeline.py
```

It runs the end-to-end preprocessing workflow and deliberately does not train models.

Current support modules:

- `config.py`: fixed paths, seeds, thresholds, tokenizer name, and heuristic lexicons.
- `loading.py`: raw CSV loading and schema validation for Jigsaw and optional Civil Comments.
- `audit.py`: raw data audit summaries.
- `harmonize.py`: binary label mapping and original-label metadata serialization.
- `text.py`: light text normalization, hashing, and text-pattern helper functions.
- `splits.py`: deterministic stratified split construction.
- `slices.py`: identity, obfuscation, implicit-toxicity proxy, and length-bucket slice construction.
- `tokenization.py`: DistilBERT token-length diagnostics.
- `validation.py`: missing-value, duplicate, class-distribution, label-drift, leakage, and split validation.
- `io_utils.py`: JSON/YAML/table output helpers.
- `reporting.py`: generated Markdown report and handoff document content.

## 4. Environment and Setup

Dependencies are documented in `requirements.txt`:

```text
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
pyarrow>=14.0
PyYAML>=6.0
transformers>=4.36
```

The current preprocessing handoff document says to install dependencies with:

```bash
pip install -r requirements.txt
```

The same handoff document notes that the verified interpreter on this machine is:

```bash
/Users/zhengpeixian/miniforge3/bin/python -m src.preprocessing.pipeline
```

Exact virtual environment creation steps are not available in the current repository artifacts. Use a Python environment that has the packages from `requirements.txt` installed.

Prerequisites before running:

- `dataset/jigsaw/raw/train.csv` must exist.
- `dataset/civil_comments/raw/train.csv` may exist, but Civil Comments is optional.
- The DistilBERT tokenizer configured as `distilbert-base-uncased` must be available locally or downloadable through `transformers` for the default strict token-length diagnostics.
- `pyarrow` is required for Parquet outputs.

## 5. How to Run the Preprocessing Pipeline

The rerun command already documented in `reports/preprocessing/README_preprocessing.md` is:

```bash
python -m src.preprocessing.pipeline
```

On this machine, the generated handoff document records this verified interpreter-specific command:

```bash
/Users/zhengpeixian/miniforge3/bin/python -m src.preprocessing.pipeline
```

Expected raw inputs:

```text
dataset/jigsaw/raw/train.csv
dataset/civil_comments/raw/train.csv
```

Jigsaw-only mode:

- `dataset/jigsaw/raw/train.csv` is mandatory.
- If the Civil file is missing, the code is designed to log a warning and continue with Jigsaw-only preprocessing.

Civil mode:

- If `dataset/civil_comments/raw/train.csv` exists and has a supported schema, the pipeline also creates optional Civil extension artifacts.
- Civil output files are not required for the main non-extension experiment.

## 6. Input Data Expectations

Jigsaw expected schema:

- Text column: `comment_text`
- Label columns: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Binary mapping: `binary_label = 1` if any source toxicity column is greater than 0, otherwise `0`.

Civil Comments expected schema:

- Text column candidates: `comment_text`, `text`
- Label column candidates: `target`, `toxicity`
- Current detected label column: `target`
- Binary mapping: `binary_label = 1` if toxicity score is at least `0.5`, otherwise `0`.
- `parent_text` is not used by the current preprocessing code.

Only these raw training CSV files are used by the current pipeline:

- `dataset/jigsaw/raw/train.csv`
- `dataset/civil_comments/raw/train.csv`

Files under `dataset/*/unused/` are present but are not used by the current preprocessing report or pipeline configuration.

## 7. Processed Outputs: What Each Output Means

All processed outputs are in `data/processed/`. CSV and Parquet versions are saved for the main split files and slice files.

### Jigsaw Main Experiment Files

Use these files for the main non-extension experiment:

- `jigsaw_train.csv` / `jigsaw_train.parquet`
  - Contains the Jigsaw training split.
  - Current row count from validation metadata: 111,519.
  - Used by teammates training TF-IDF + Logistic Regression and DistilBERT models.
- `jigsaw_val.csv` / `jigsaw_val.parquet`
  - Contains the Jigsaw validation split.
  - Current row count: 23,897.
  - Used for development-time validation/model selection.
- `jigsaw_test.csv` / `jigsaw_test.parquet`
  - Contains the Jigsaw test split.
  - Current row count: 23,897.
  - Used for final main-experiment evaluation.
- `train.csv`
  - Simplified teammate-facing Jigsaw training export.
  - Contains exactly `raw_text`, `clean_text`, and `label`.
- `val.csv`
  - Simplified teammate-facing Jigsaw validation export.
  - Contains exactly `raw_text`, `clean_text`, and `label`.
- `test.csv`
  - Simplified teammate-facing Jigsaw test export.
  - Contains exactly `raw_text`, `clean_text`, and `label`.

### Optional Civil Extension Files

The current regenerated artifact set is Jigsaw-only, so Civil processed files are not currently present in `data/processed/`.

When Civil Comments is enabled and present, the pipeline code can generate optional Civil extension files. They should not be used in the main Jigsaw-only comparison unless the team is explicitly running extension experiments:

- `civil_aug_pool_full.csv` / `civil_aug_pool_full.parquet`
  - Civil Comments augmentation pool.
  - Current row count is not available in the regenerated Jigsaw-only artifacts.
  - Intended for controlled augmentation experiments.
- `civil_external_test.csv` / `civil_external_test.parquet`
  - Civil Comments external test split.
  - Current row count is not available in the regenerated Jigsaw-only artifacts.
  - Intended for optional external evaluation or robustness checks.
- `civil_aug_pool_capped_match_jigsaw_train.csv` / `civil_aug_pool_capped_match_jigsaw_train.parquet`
  - Random sample from the Civil augmentation pool capped to match the Jigsaw training row count.
  - Current row count is not available in the regenerated Jigsaw-only artifacts.
  - Intended for size-controlled augmentation experiments.
- `civil_aug_pool_identity.csv` / `civil_aug_pool_identity.parquet`
  - Civil augmentation-pool rows with detected identity content.
  - Current row count is not available in the regenerated Jigsaw-only artifacts.
  - Intended for identity-focused extension analysis or augmentation.

### Challenge Slice Files

Jigsaw slice files:

- `jigsaw_identity_slice.csv` / `jigsaw_identity_slice.parquet`
  - Rows where `has_identity_term == 1`.
  - Useful for identity-related error analysis.
- `jigsaw_obfuscation_slice.csv` / `jigsaw_obfuscation_slice.parquet`
  - Rows where `has_obfuscation == 1`.
  - Useful for robustness checks on obfuscated or distorted text.
- `jigsaw_implicit_proxy_slice.csv` / `jigsaw_implicit_proxy_slice.parquet`
  - Rows where `implicit_proxy == 1`.
  - This is a heuristic proxy slice, not a gold implicit-toxicity label.

Civil optional slice files:

- `civil_obfuscation_slice.csv` / `civil_obfuscation_slice.parquet`
- `civil_implicit_proxy_slice.csv` / `civil_implicit_proxy_slice.parquet`
- `civil_aug_pool_identity.csv` / `civil_aug_pool_identity.parquet`

These Civil slices are extension-oriented and should not be part of the main non-extension comparison.

### Standard Processed Columns

The generated `metadata/data_dictionary.md` documents the common processed schema:

- `sample_id`
- `source_dataset`
- `raw_text`
- `text_clean`
- `text_tfidf`
- `binary_label`
- `orig_label_info`
- `char_len`
- `word_len`
- `text_hash_raw`
- `text_hash_normalized`
- `bert_token_len`
- `split`
- `has_identity_term`
- `has_obfuscation`
- `implicit_proxy`
- `length_bucket`

Civil outputs also include available Civil identity-score columns used for optional identity slicing.

## 8. Which Files Are Used in the Main Experiment

The non-extension phase uses only Jigsaw processed outputs:

```text
data/processed/jigsaw_train.parquet
data/processed/jigsaw_val.parquet
data/processed/jigsaw_test.parquet
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

CSV equivalents are also available:

```text
data/processed/jigsaw_train.csv
data/processed/jigsaw_val.csv
data/processed/jigsaw_test.csv
```

Modeling columns:

- TF-IDF + Logistic Regression input text: `text_tfidf`
- DistilBERT input text: `text_clean`
- Target label: `binary_label`

Civil outputs are optional extension artifacts. Do not use Civil outputs in the main TF-IDF + Logistic Regression vs DistilBERT comparison unless the team explicitly switches to an extension or augmentation experiment.

## 9. Metadata and Reports

Metadata files:

- `metadata/label_mapping.json`
  - Records Jigsaw source label columns and Civil threshold mapping.
  - Current Civil threshold is `0.5`.
- `metadata/split_summary.json`
  - Records split methods, seeds, target ratios, and split counts.
  - Current Jigsaw outputs use two-stage `StratifiedShuffleSplit` with seeds 42 and 43, stratified on `binary_label`.
  - Civil uses `StratifiedShuffleSplit` when Civil processing is enabled and labels are available.
- `metadata/data_audit_summary.json`
  - Raw audit metrics: row counts, missing text/label counts, duplicate counts, length statistics, feature percentages, and class distributions.
- `metadata/slice_definitions.json`
  - Documents identity, obfuscation, implicit proxy, and length-bucket definitions.
  - Length thresholds are `short_max_word_len = 22` and `medium_max_word_len = 56`.
- `metadata/preprocessing_config.yaml`
  - Serializable preprocessing configuration: paths, thresholds, seed values, cleaning flags, tokenizer name, and heuristic terms.
- `metadata/data_dictionary.md`
  - Human-readable definitions of standardized processed columns.

Report files:

- `reports/preprocessing/preprocessing_report.md`
  - Main human-readable preprocessing report with raw files, schema mapping, duplicate removal, split sizes, label distributions, token statistics, leakage checks, slice summaries, assumptions, and warnings.
- `reports/preprocessing/preprocessing_report.json`
  - Machine-readable version of the preprocessing report.
- `reports/preprocessing/README_preprocessing.md`
  - Team handoff document explaining main experiment files, modeling columns, optional Civil artifacts, and rerun command.

## 10. Practical Teammate Workflow

Suggested workflow for a teammate:

1. Start with `reports/preprocessing/README_preprocessing.md` for the shortest handoff.
2. Read this guide for repository structure and artifact purposes.
3. For modeling in the main experiment, load only:
   - `data/processed/train.csv`, `data/processed/val.csv`, and `data/processed/test.csv` for simplified teammate-facing files.
   - `data/processed/jigsaw_train.parquet`
   - `data/processed/jigsaw_val.parquet`
   - `data/processed/jigsaw_test.parquet`
4. Use `text_tfidf` for TF-IDF + Logistic Regression.
5. Use `text_clean` for DistilBERT.
6. Use `binary_label` as the target.
7. Use `reports/preprocessing/preprocessing_report.md` and `metadata/*.json` to verify split sizes, label ratios, and validation outcomes.
8. Ignore Civil files unless working on an extension, augmentation, robustness, or external evaluation task.
9. Use challenge slice files for targeted error analysis after model predictions exist.

## 11. Known Limitations / Notes

- The current repository state covers preprocessing and data preparation. Model training and model evaluation results are not available in the inspected repository artifacts.
- Jigsaw is mandatory for the pipeline; Civil Comments is optional.
- Civil artifacts are extension-oriented and are not part of the main non-extension experiment.
- Identity and obfuscation slices are heuristic unless dataset-provided identity columns are available.
- `implicit_proxy` is explicitly not a gold label.
- Length buckets are based on empirical Jigsaw word-length quantiles.
- The current default configuration requires the DistilBERT tokenizer for token-length diagnostics.
- Exact virtual environment setup beyond `requirements.txt` and the verified Miniforge interpreter note is not available in the current repository artifacts.
