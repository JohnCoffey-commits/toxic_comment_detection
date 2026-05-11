"""Product-facing risk and review-priority mapping."""

from __future__ import annotations

from statistics import mean

from backend.services.keyword_cues import detect_language_cues


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def classify_risk(toxic_probability: float) -> dict[str, object]:
    if toxic_probability >= 0.85:
        return {
            "risk_level": "High",
            "review_recommendation": "High priority for human review",
            "review_priority": 1,
        }
    if toxic_probability >= 0.65:
        return {
            "risk_level": "Medium",
            "review_recommendation": "Review recommended",
            "review_priority": 2,
        }
    if toxic_probability >= 0.50:
        return {
            "risk_level": "Borderline",
            "review_recommendation": "Review if moderation capacity allows",
            "review_priority": 3,
        }
    return {
        "risk_level": "Low",
        "review_recommendation": "No immediate review priority",
        "review_priority": 4,
    }


def _as_float(value: object, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_int(value: object, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def build_review_item(
    prediction: dict[str, object],
    *,
    row_id: int | None = None,
    original_text: str | None = None,
) -> dict[str, object]:
    source_text = original_text
    if source_text is None:
        source_text = str(prediction.get("original_text", ""))

    threshold = _as_float(prediction.get("threshold"), 0.5)
    toxic_probability = _as_float(prediction.get("toxic_probability"), 0.0)
    non_toxic_probability = _as_float(
        prediction.get("non_toxic_probability"),
        max(0.0, 1.0 - toxic_probability),
    )
    numeric_label = _as_int(
        prediction.get("numeric_label"),
        1 if toxic_probability >= threshold else 0,
    )
    predicted_label = str(
        prediction.get("predicted_label")
        or ("toxic" if numeric_label == 1 else "non-toxic")
    )
    confidence = _as_float(
        prediction.get("confidence"),
        max(toxic_probability, non_toxic_probability),
    )
    cleaned_text = str(prediction.get("cleaned_text", ""))

    risk_score = toxic_probability
    risk = classify_risk(risk_score)
    item: dict[str, object] = {
        "original_text": source_text,
        "prediction": predicted_label,
        "risk_level": risk["risk_level"],
        "risk_score": round(risk_score, 4),
        "risk_score_percent": format_percent(risk_score),
        "review_recommendation": risk["review_recommendation"],
        "review_priority": risk["review_priority"],
        "detected_cues": detect_language_cues(source_text),
        "advanced": {
            "predicted_label": predicted_label,
            "numeric_label": numeric_label,
            "toxic_probability": toxic_probability,
            "non_toxic_probability": non_toxic_probability,
            "confidence": confidence,
            "threshold": threshold,
            "cleaned_text": cleaned_text,
        },
    }
    if row_id is not None:
        item["row_id"] = row_id
    return item


def build_summary(items: list[dict[str, object]]) -> dict[str, object]:
    total = len(items)
    risk_counts = {"High": 0, "Medium": 0, "Borderline": 0, "Low": 0}
    predicted_toxic = 0

    for item in items:
        risk_level = str(item.get("risk_level", "Low"))
        if risk_level in risk_counts:
            risk_counts[risk_level] += 1
        if item.get("prediction") == "toxic":
            predicted_toxic += 1

    average_risk = mean(float(item.get("risk_score", 0.0)) for item in items) if items else 0.0
    needs_review = risk_counts["High"] + risk_counts["Medium"] + risk_counts["Borderline"]

    return {
        "total_comments": total,
        "needs_review": needs_review,
        "high_risk": risk_counts["High"],
        "medium_risk": risk_counts["Medium"],
        "borderline": risk_counts["Borderline"],
        "low_risk": risk_counts["Low"],
        "predicted_toxic": predicted_toxic,
        "predicted_non_toxic": total - predicted_toxic,
        "average_risk_score": round(average_risk, 4),
        "average_risk_score_percent": format_percent(average_risk),
    }
