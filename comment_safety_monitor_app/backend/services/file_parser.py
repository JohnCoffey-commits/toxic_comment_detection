"""Upload inspection and comment extraction helpers."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class FileParserError(ValueError):
    """Raised when an uploaded file cannot be inspected or parsed."""


@dataclass(frozen=True)
class ParsedComment:
    row_id: int
    text: str


TEXT_COLUMN_CANDIDATES = (
    "comment_text",
    "comment",
    "comments",
    "text",
    "content",
    "message",
    "body",
    "raw_text",
    "clean_text",
)


def _file_type(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".txt":
        return "txt"
    raise FileParserError("Unsupported file type. Please upload a CSV or TXT file.")


def _ensure_content(content: bytes) -> None:
    if not content or not content.strip():
        raise FileParserError("The uploaded file is empty.")


def _decode_text(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("utf-8", errors="replace")


def _read_csv(content: bytes) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise FileParserError("The uploaded CSV is empty.") from exc
    except Exception as exc:  # pandas gives parser-specific subclasses inconsistently.
        raise FileParserError("The uploaded CSV could not be read.") from exc

    if not len(frame.columns):
        raise FileParserError("The uploaded CSV does not contain any columns.")
    return frame


def _normalise_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _valid_count_for_column(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].map(_normalise_cell).astype(bool).sum())


def _valid_count_any_column(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    normalised = frame.map(_normalise_cell)
    return int(normalised.astype(bool).any(axis=1).sum())


def _guess_text_column(columns: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return columns[0] if len(columns) == 1 else None


def inspect_upload(filename: str | None, content: bytes) -> dict[str, object]:
    _ensure_content(content)
    kind = _file_type(filename)

    if kind == "txt":
        text = _decode_text(content)
        comments = [line.strip() for line in text.splitlines() if line.strip()]
        return {
            "file_type": "txt",
            "columns": [],
            "default_text_column": None,
            "valid_comment_count": len(comments),
        }

    frame = _read_csv(content)
    columns = [str(column) for column in frame.columns]
    default_text_column = _guess_text_column(columns)
    if default_text_column:
        valid_count = _valid_count_for_column(frame, default_text_column)
    else:
        valid_count = _valid_count_any_column(frame)

    return {
        "file_type": "csv",
        "columns": columns,
        "default_text_column": default_text_column,
        "valid_comment_count": valid_count,
    }


def parse_upload_comments(
    filename: str | None,
    content: bytes,
    text_column: str | None = None,
) -> tuple[str, list[ParsedComment]]:
    _ensure_content(content)
    kind = _file_type(filename)

    if kind == "txt":
        text = _decode_text(content)
        comments = [
            ParsedComment(row_id=line_number, text=line.strip())
            for line_number, line in enumerate(text.splitlines(), start=1)
            if line.strip()
        ]
        if not comments:
            raise FileParserError("No valid comments were found in the uploaded TXT file.")
        return kind, comments

    frame = _read_csv(content)
    columns = [str(column) for column in frame.columns]

    if len(columns) == 1:
        selected_column = columns[0]
    else:
        if not text_column:
            raise FileParserError("Please select the CSV column that contains comment text.")
        if text_column not in columns:
            raise FileParserError(
                f"Column '{text_column}' was not found in the uploaded CSV."
            )
        selected_column = text_column

    comments = [
        ParsedComment(row_id=index + 1, text=cleaned)
        for index, value in enumerate(frame[selected_column].tolist())
        if (cleaned := _normalise_cell(value))
    ]
    if not comments:
        raise FileParserError("No valid comments were found in the selected text column.")

    return kind, comments
