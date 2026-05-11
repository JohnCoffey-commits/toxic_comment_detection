"""Inference utilities for the DistilBERT toxic comment detector.

This module intentionally contains no training logic. It loads a saved
Hugging Face model directory and exposes single-text, list, and CSV inference
helpers that can be reused by a future Streamlit app, Gradio app, or API.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "model" / "distilbert_model"
DRIVE_MODEL_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/toxic_comment_detection/comment_safety_monitor_app/model/distilbert_model"
)
DEFAULT_MODEL_DIR = LOCAL_MODEL_DIR

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text_for_distilbert(text: object) -> str:
    """Match the light `text_clean` preprocessing used for DistilBERT training."""

    if text is None or pd.isna(text):
        return ""

    cleaned = unicodedata.normalize("NFKC", html.unescape(str(text)))
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = EMAIL_RE.sub("<EMAIL>", cleaned)
    cleaned = URL_RE.sub("<URL>", cleaned)
    return WHITESPACE_RE.sub(" ", cleaned).strip()


class ToxicCommentPredictor:
    """Reusable DistilBERT inference wrapper for toxic comment detection."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        threshold: float = 0.5,
        max_length: int = 256,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_dir = self._resolve_model_dir(model_dir)
        self.threshold = float(threshold)
        self.max_length = int(max_length)
        self.device = self._resolve_device(device)

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {self.threshold}")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None or str(device).lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return resolved

    @staticmethod
    def _resolve_model_dir(model_dir: str | Path | None) -> Path:
        if model_dir is not None:
            path = Path(model_dir).expanduser()
            if path.exists():
                return path
            raise FileNotFoundError(
                "Model directory not found: "
                f"{path}\n"
                "Pass --model-dir pointing to a local exported model directory, "
                "or download the model files under:\n"
                f"    {LOCAL_MODEL_DIR}\n"
                "If you are running in Google Colab, mount Google Drive first:\n"
                "    from google.colab import drive\n"
                "    drive.mount('/content/drive')"
            )

        for candidate in (LOCAL_MODEL_DIR, DRIVE_MODEL_DIR):
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Model directory not found. Tried local-first path and Drive fallback path:\n"
            f"    local: {LOCAL_MODEL_DIR}\n"
            f"    drive: {DRIVE_MODEL_DIR}\n"
            "For local development, download the exported model files under the local path above. "
            "For Google Colab, mount Google Drive first:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
            "Alternatively, pass --model-dir manually."
        )

    def predict(self, text: object) -> dict[str, object]:
        """Predict toxicity for a single input comment."""

        return self.predict_many([text])[0]

    def predict_many(self, texts: Iterable[object]) -> list[dict[str, object]]:
        """Predict toxicity for a list or other iterable of input comments."""

        original_texts = list(texts)
        if not original_texts:
            return []

        cleaned_texts = [clean_text_for_distilbert(text) for text in original_texts]

        encoded = self.tokenizer(
            cleaned_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()

        results: list[dict[str, object]] = []
        for original, cleaned, probs in zip(original_texts, cleaned_texts, probabilities):
            non_toxic_probability = float(probs[0])
            toxic_probability = float(probs[1])
            numeric_label = 1 if toxic_probability >= self.threshold else 0
            predicted_label = "toxic" if numeric_label == 1 else "non-toxic"
            confidence = max(toxic_probability, non_toxic_probability)

            results.append(
                {
                    "original_text": "" if original is None or pd.isna(original) else str(original),
                    "cleaned_text": cleaned,
                    "predicted_label": predicted_label,
                    "numeric_label": numeric_label,
                    "confidence": confidence,
                    "toxic_probability": toxic_probability,
                    "non_toxic_probability": non_toxic_probability,
                    "threshold": self.threshold,
                }
            )

        return results

    def predict_csv(
        self,
        input_csv: str | Path,
        text_column: str = "text",
        output_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        """Run inference on a CSV and optionally write predictions to another CSV."""

        input_csv = Path(input_csv).expanduser()
        frame = pd.read_csv(input_csv)
        if text_column not in frame.columns:
            raise KeyError(
                f"Column '{text_column}' not found in {input_csv}. "
                f"Available columns: {list(frame.columns)}"
            )

        predictions = pd.DataFrame(self.predict_many(frame[text_column].tolist()))
        output = pd.concat([frame.reset_index(drop=True), predictions], axis=1)

        if output_csv is not None:
            output_path = Path(output_csv).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(output_path, index=False)

        return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DistilBERT toxic comment inference for one text or a CSV file."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "Optional saved model directory override. If omitted, the loader uses "
            "comment_safety_monitor_app/model/distilbert_model first, then the Google Drive Colab path."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Toxic probability threshold.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max sequence length.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or another torch device string.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Single comment string to classify.")
    input_group.add_argument("--csv", help="Input CSV path for batch inference.")

    parser.add_argument("--text-column", default="text", help="CSV column containing comment text.")
    parser.add_argument("--output", help="Optional output CSV path for batch predictions.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    predictor = ToxicCommentPredictor(
        model_dir=args.model_dir,
        threshold=args.threshold,
        max_length=args.max_length,
        device=args.device,
    )

    if args.text is not None:
        result = predictor.predict(args.text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    output = predictor.predict_csv(
        input_csv=args.csv,
        text_column=args.text_column,
        output_csv=args.output,
    )
    if args.output:
        print(f"Wrote predictions to: {args.output}")
    else:
        print(output.to_json(orient="records", indent=2, force_ascii=False))


if __name__ == "__main__":
    main()
