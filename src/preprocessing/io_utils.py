"""I/O helpers for metadata, reports, and tabular outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class MissingDependencyError(RuntimeError):
    """Raised when an optional runtime dependency is required for an output."""


def ensure_output_dirs(*paths: Path) -> None:
    """Create output directories if needed."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def to_builtin(value: Any) -> Any:
    """Convert numpy/pandas values into JSON serializable Python values."""

    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) and not isinstance(value, (str, bytes)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable, pretty JSON metadata."""

    path.write_text(
        json.dumps(to_builtin(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(ch in text for ch in ":#[]{}&,*>!|%@`\"'"):
        return json.dumps(text)
    return text


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write YAML metadata, using PyYAML when available and a simple fallback otherwise."""

    serializable = to_builtin(payload)
    try:
        import yaml  # type: ignore

        path.write_text(
            yaml.safe_dump(serializable, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return
    except ModuleNotFoundError:
        LOGGER.warning("PyYAML is not installed; writing a simple YAML fallback.")

    def render(obj: Any, indent: int = 0) -> list[str]:
        pad = " " * indent
        if isinstance(obj, dict):
            lines: list[str] = []
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}{key}:")
                    lines.extend(render(value, indent + 2))
                else:
                    lines.append(f"{pad}{key}: {_yaml_scalar(value)}")
            return lines
        if isinstance(obj, list):
            lines = []
            for value in obj:
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}-")
                    lines.extend(render(value, indent + 2))
                else:
                    lines.append(f"{pad}- {_yaml_scalar(value)}")
            return lines
        return [f"{pad}{_yaml_scalar(obj)}"]

    path.write_text("\n".join(render(serializable)) + "\n", encoding="utf-8")


def require_parquet_engine() -> None:
    """Fail clearly when pandas cannot write parquet files."""

    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "pyarrow is required to write parquet outputs. Install project "
            "dependencies with: pip install -r requirements.txt"
        ) from exc


def save_frame(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    """Save a dataframe as CSV and parquet."""

    require_parquet_engine()
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

