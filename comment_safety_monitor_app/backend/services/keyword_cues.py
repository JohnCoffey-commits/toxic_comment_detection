"""Rule-based language cue detection for reviewer highlighting.

These cues are review hints only. They are not model explanations and do not
change the classifier output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CueDefinition:
    keyword: str
    category: str
    severity: str


CUE_DEFINITIONS: tuple[CueDefinition, ...] = (
    CueDefinition("stupid", "Direct insult", "High"),
    CueDefinition("idiot", "Direct insult", "High"),
    CueDefinition("moron", "Direct insult", "High"),
    CueDefinition("dumb", "Direct insult", "Medium"),
    CueDefinition("garbage", "Dismissive language", "Medium"),
    CueDefinition("trash", "Dismissive language", "Medium"),
    CueDefinition("useless", "Dismissive language", "High"),
    CueDefinition("nonsense", "Dismissive language", "Low"),
    CueDefinition("pathetic", "Direct insult", "High"),
    CueDefinition("shut up", "Hostile language", "High"),
    CueDefinition("loser", "Direct insult", "High"),
    CueDefinition("worthless", "Direct insult", "High"),
    CueDefinition("hate you", "Hostile language", "High"),
    CueDefinition("go die", "Threat-like language", "High"),
    CueDefinition("kill you", "Threat-like language", "High"),
    CueDefinition("hurt you", "Threat-like language", "High"),
)


def _compile_keyword_pattern(keyword: str) -> re.Pattern[str]:
    escaped = re.escape(keyword).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![\w]){escaped}(?![\w])", re.IGNORECASE | re.UNICODE)


def _positions_for_keyword(text: str, keyword: str) -> list[dict[str, int]]:
    pattern = _compile_keyword_pattern(keyword)
    return [{"start": match.start(), "end": match.end()} for match in pattern.finditer(text)]


def detect_language_cues(text: object) -> list[dict[str, object]]:
    """Return unique cue chips with first span and all highlight positions."""

    if text is None:
        return []

    source = str(text)
    if not source:
        return []

    cues: list[dict[str, object]] = []
    for definition in CUE_DEFINITIONS:
        positions = _positions_for_keyword(source, definition.keyword)
        if not positions:
            continue

        first = positions[0]
        cues.append(
            {
                "keyword": definition.keyword,
                "category": definition.category,
                "severity": definition.severity,
                "start": first["start"],
                "end": first["end"],
                "count": len(positions),
                "positions": positions,
            }
        )

    return sorted(cues, key=lambda cue: (int(cue["start"]), int(cue["end"])))
