# Comment Safety Monitor

Comment Safety Monitor is a lightweight web MVP for AI-assisted comment review
and risk prioritisation. It is designed as a content safety triage console for
human reviewers, not as an automatic enforcement system.

## What The App Does

- Checks a single pasted comment and returns a product-friendly safety risk
  assessment.
- Uploads CSV or TXT files and creates a prioritised human review queue.
- Highlights detected language cues using rule-based review hints.
- Keeps technical prediction fields hidden under advanced sections by default.

## Pages

### Text Check

Paste one comment, run a risk check, and review:

- Risk score
- Risk level
- Review recommendation
- Detected language cues
- Collapsed advanced details

Empty and whitespace-only comments are blocked before inference.

### File Review

Upload a CSV or TXT file to create a prioritised review queue.

- CSV files can use a single automatic text column or a selected text column.
- TXT files treat each non-empty line as one comment.
- Empty and whitespace-only rows are ignored.
- Results are sorted by highest risk score.
- Filters support All, High, Medium, Borderline, and Low.
- Display limits support Top 10, Top 25, and All.
- The review queue can be downloaded as CSV.

## Existing Inference Layer

The backend reuses the existing inference-only layer in:

```text
comment_safety_monitor_app/inference.py
```

The preferred local model directory is:

```text
comment_safety_monitor_app/model/distilbert_model
```

No model retraining, preprocessing rebuild, or model export is required for this
web MVP.

## Backend Setup

Install the web API dependencies from the `comment_safety_monitor_app/` directory:

```bash
cd comment_safety_monitor_app
pip install -r requirements-web.txt
```

The inference runtime still requires the existing inference dependencies. If
they are not already installed in your environment, install them as well:

```bash
pip install -r requirements-inference.txt
```

Run the backend:

```bash
cd comment_safety_monitor_app
uvicorn backend.app:app --reload --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

## Frontend Setup

Install and run the React frontend:

```bash
cd comment_safety_monitor_app/frontend
npm install
npm run dev
```

The Vite dev server runs on:

```text
http://localhost:5173
```

During local development, `/api` requests are proxied to the FastAPI backend on
port `8000`.

## Supported File Formats

- CSV
- TXT

For CSV uploads, files with multiple columns require selecting the comment text
column before analysis. For TXT uploads, each non-empty line is treated as one
comment.

## Output Interpretation

- Risk score: the product-facing score used to prioritise review.
- Risk level: High, Medium, Borderline, or Low.
- Review recommendation: the suggested human review priority.
- Detected language cues: predefined rule-based hints that help reviewers locate
  potentially problematic language in the text.

Advanced sections expose technical fields such as `toxic_probability`,
`non_toxic_probability`, `confidence`, `threshold`, `numeric_label`, and
`cleaned_text`.

## Disclaimer

This app is decision-support for human review only. It does not make automatic
enforcement decisions.

Keyword cues are rule-based review hints, not causal model explanations. They do
not indicate why the classifier produced a result and do not change the model
prediction.
