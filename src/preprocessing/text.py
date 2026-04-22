"""Text normalization, hashing, and transparent heuristic detectors."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections.abc import Iterable

import pandas as pd

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
REPEATED_PUNCT_RE = re.compile(r"([!?.,;:])\1{2,}")
ALL_CAPS_TOKEN_RE = re.compile(r"\b[A-Z]{2,}\b")
DIGIT_RE = re.compile(r"\d")
WHITESPACE_RE = re.compile(r"\s+")
INSERTED_SYMBOL_RE = re.compile(r"(?i)\b[a-z]+(?:[^a-z0-9\s]+[a-z]+){2,}\b")
LEETSPEAK_RE = re.compile(r"(?i)\b[a-z]*[4@][a-z]*|[a-z]*[3][a-z]*|[a-z]*[1!][a-z]*|[a-z]*[0][a-z]*\b")
REPEATED_CHAR_RE = re.compile(r"(?i)([a-z])\1{2,}")
SPLIT_ABUSIVE_RE = re.compile(
    r"(?ix)\b("
    r"f[\W_]*u[\W_]*c[\W_]*k|"
    r"s[\W_]*h[\W_]*i[\W_]*t|"
    r"b[\W_]*i[\W_]*t[\W_]*c[\W_]*h|"
    r"c[\W_]*u[\W_]*n[\W_]*t"
    r")\b"
)


def coerce_text(value: object) -> str | None:
    """Return None for missing values and a string otherwise."""

    if pd.isna(value):
        return None
    return str(value)


def normalize_for_duplicate(text: object, lowercase: bool = True) -> str:
    """Light normalization for duplicate detection only."""

    raw = coerce_text(text)
    if raw is None:
        return ""
    normalized = unicodedata.normalize("NFKC", html.unescape(raw))
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized.casefold() if lowercase else normalized


def clean_text(
    text: object,
    replace_urls: bool = True,
    replace_emails: bool = True,
    url_replacement: str = "<URL>",
    email_replacement: str = "<EMAIL>",
) -> str:
    """Apply light, non-destructive text normalization."""

    raw = coerce_text(text)
    if raw is None:
        return ""
    cleaned = unicodedata.normalize("NFKC", html.unescape(raw))
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    if replace_emails:
        cleaned = EMAIL_RE.sub(email_replacement, cleaned)
    if replace_urls:
        cleaned = URL_RE.sub(url_replacement, cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def sha256_text(text: object) -> str:
    """Hash text deterministically with SHA-256."""

    raw = "" if text is None or pd.isna(text) else str(text)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def word_count(text: object) -> int:
    """Count non-empty whitespace-delimited tokens."""

    raw = coerce_text(text)
    if raw is None:
        return 0
    return len([part for part in WHITESPACE_RE.split(raw.strip()) if part])


def contains_url(text: object) -> bool:
    raw = coerce_text(text) or ""
    return bool(URL_RE.search(raw))


def contains_email(text: object) -> bool:
    raw = coerce_text(text) or ""
    return bool(EMAIL_RE.search(raw))


def contains_digits(text: object) -> bool:
    raw = coerce_text(text) or ""
    return bool(DIGIT_RE.search(raw))


def contains_repeated_punctuation(text: object) -> bool:
    raw = coerce_text(text) or ""
    return bool(REPEATED_PUNCT_RE.search(raw))


def contains_all_caps_token(text: object) -> bool:
    raw = coerce_text(text) or ""
    return bool(ALL_CAPS_TOKEN_RE.search(raw))


def build_terms_regex(terms: Iterable[str]) -> re.Pattern[str]:
    """Build a case-insensitive whole-term regex."""

    escaped = [re.escape(term).replace(r"\ ", r"\s+") for term in terms]
    return re.compile(r"(?i)\b(?:" + "|".join(escaped) + r")\b")


def has_identity_term(text: object, terms: Iterable[str]) -> bool:
    raw = coerce_text(text) or ""
    return bool(build_terms_regex(terms).search(raw))


def has_explicit_toxic_term(text: object, terms: Iterable[str]) -> bool:
    raw = coerce_text(text) or ""
    return bool(build_terms_regex(terms).search(raw))


def has_obfuscation(text: object) -> bool:
    """Detect transparent character-level obfuscation heuristics."""

    raw = coerce_text(text) or ""
    return any(
        pattern.search(raw)
        for pattern in (
            INSERTED_SYMBOL_RE,
            LEETSPEAK_RE,
            REPEATED_CHAR_RE,
            SPLIT_ABUSIVE_RE,
        )
    )

