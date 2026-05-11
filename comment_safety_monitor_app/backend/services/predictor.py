"""Cached adapter around the existing inference-only predictor."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable


NEW_ROOT = Path(__file__).resolve().parents[2]
if str(NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(NEW_ROOT))

from inference import LOCAL_MODEL_DIR, ToxicCommentPredictor  # noqa: E402


@lru_cache(maxsize=1)
def get_predictor() -> ToxicCommentPredictor:
    """Load the model once per backend process."""

    return ToxicCommentPredictor(model_dir=LOCAL_MODEL_DIR)


def predict_text(text: str) -> dict[str, object]:
    return get_predictor().predict(text)


def predict_many(texts: Iterable[str]) -> list[dict[str, object]]:
    predictor = get_predictor()
    batch_predict = getattr(predictor, "predict_many", None)
    if callable(batch_predict):
        return list(batch_predict(texts))
    return [predictor.predict(text) for text in texts]
