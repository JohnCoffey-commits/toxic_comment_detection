# DistilBERT Inference Layer

This folder contains the inference-only layer for the exported DistilBERT toxic comment classifier.

It does not retrain the model, modify preprocessing outputs, or build a frontend.

## Model Paths

The inference layer now uses local-first loading.

Default local model directory:

```text
comment_safety_monitor_app/model/distilbert_model
```

In this local project, that is:

```text
/Users/zhengpeixian/ZPX/UTS/NLP/Assignment3/toxic_comment_detection/comment_safety_monitor_app/model/distilbert_model
```

Google Drive fallback directory:

```text
/content/drive/MyDrive/Colab Notebooks/toxic_comment_detection/comment_safety_monitor_app/model/distilbert_model
```

Expected files in that directory:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

`tokenizer.json` is sufficient for inference even if `vocab.txt` is not present.

## Dependencies

Install the same core runtime libraries used by the training notebook:

```bash
pip install torch transformers pandas
```

## Colab Usage

Mount Google Drive first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then run a single prediction:

```bash
python inference.py --text "You are wonderful."
```

If `comment_safety_monitor_app/model/distilbert_model` is not present in Colab, the loader falls back to the Drive model path after Drive is mounted. If both paths are missing, the script fails with a clear message asking you to mount Google Drive or pass `--model-dir` manually.

## Local Usage

The exported model directory has been downloaded locally under:

```text
/Users/zhengpeixian/ZPX/UTS/NLP/Assignment3/toxic_comment_detection/comment_safety_monitor_app/model/distilbert_model
```

From the `comment_safety_monitor_app/` folder, no `--model-dir` is needed:

```bash
python inference.py --text "You are wonderful."
```

From the project root, use:

```bash
python comment_safety_monitor_app/inference.py --text "You are wonderful."
```

To load a different local copy, pass it explicitly:

```bash
python inference.py --model-dir /path/to/distilbert_model --text "You are wonderful."
```

## CSV Batch Inference

Default CSV text column is `text`:

```bash
python inference.py \
  --csv input_comments.csv \
  --text-column text \
  --output predictions.csv
```

The existing processed project files usually use:

```text
raw_text
clean_text
```

So for project processed CSVs, use one of:

```bash
python inference.py --csv test.csv --text-column raw_text --output predictions.csv
python inference.py --csv test.csv --text-column clean_text --output predictions.csv
```

For DistilBERT consistency, the training source used the internal `text_clean` column. The simplified teammate-facing CSV column `clean_text` has extra cleanup, so `raw_text` is usually the safer CSV input if you want the inference layer to apply the same light DistilBERT cleaning itself.

## Python API

```python
from inference import ToxicCommentPredictor

predictor = ToxicCommentPredictor()
result = predictor.predict("You are wonderful.")
print(result)
```

For batch prediction:

```python
texts = ["You are wonderful.", "I hate you."]
results = predictor.predict_many(texts)
```

For CSV prediction:

```python
df = predictor.predict_csv(
    "input_comments.csv",
    text_column="raw_text",
    output_csv="predictions.csv",
)
```

## Returned Fields

Each prediction returns:

```text
original_text
cleaned_text
predicted_label
numeric_label
confidence
toxic_probability
non_toxic_probability
threshold
```

Prediction rule:

```text
numeric_label = 1 if toxic_probability >= threshold
numeric_label = 0 otherwise
```

Label mapping:

```text
1 = toxic
0 = non-toxic
```

Confidence rule:

```text
confidence = max(toxic_probability, non_toxic_probability)
```

## Preprocessing Used at Inference

The inference layer matches the training notebook's DistilBERT input assumptions as closely as possible:

- Unicode NFKC normalization
- HTML entity unescape
- replace carriage returns and newlines with spaces
- replace email patterns with `<EMAIL>`
- replace URL patterns with `<URL>`
- collapse whitespace and strip

It does not lowercase text, remove punctuation, stem, lemmatize, or apply the TF-IDF-specific `text_tfidf` transformation.

## Caveats Before Building the Web Frontend

- This is an inference layer only, not a frontend or API server.
- The loader checks the local model directory first, then the Google Drive Colab path.
- For local development, the checked-in code expects the downloaded model files under `comment_safety_monitor_app/model/distilbert_model`.
- If you move the model directory, pass `--model-dir` to the new location.
- The model was trained on a 20,000-row training subset in the training notebook, so frontend messaging should avoid overstating production readiness.
