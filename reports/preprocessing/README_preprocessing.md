# Preprocessing Handoff

## Main Experiment Files
Use only the Jigsaw processed files for the non-extension phase:

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/jigsaw_train.parquet` or `.csv`
- `data/processed/jigsaw_val.parquet` or `.csv`
- `data/processed/jigsaw_test.parquet` or `.csv`

The Jigsaw train/validation/test splits are generated with stratification on `binary_label`.

Civil Comments is optional and is not used in the main experiment.

## Modeling Columns
- Teammate-facing CSVs contain exactly `raw_text`, `clean_text`, and `label`.
- TF-IDF + Logistic Regression should use `text_tfidf`.
- DistilBERT should use `text_clean`.
- The binary target is `binary_label`.

## Optional Civil Extension Artifacts
When Civil Comments is enabled and present, the following files are reserved for controlled augmentation, robustness analysis, or optional external evaluation:

- `data/processed/civil_aug_pool_full.parquet`
- `data/processed/civil_external_test.parquet`
- `data/processed/civil_aug_pool_capped_match_jigsaw_train.parquet`
- `data/processed/civil_aug_pool_identity.parquet` when identity rows are available

These files may be dropped without affecting the Jigsaw-only pipeline.

## Rerun Command
From the repository root, run:

```bash
python -m src.preprocessing.pipeline
```

Use the Python environment where `requirements.txt` was installed. On this machine, `/Users/zhengpeixian/miniforge3/bin/python -m src.preprocessing.pipeline` is the verified interpreter.

Install runtime dependencies first if needed:

```bash
pip install -r requirements.txt
```
