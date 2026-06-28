"""Small IO helpers shared by mass-estimation stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            'pandas is required for mass-estimation stages. Install with: pip install -e ".[mlops]"'
        ) from exc
    return pd


def read_table(path: str | Path):
    pd = require_pandas()
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(table_path, lines=suffix == ".jsonl")
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    raise ValueError(f"Unsupported table format '{suffix}' for {table_path}")


def write_table(df: Any, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif suffix == ".jsonl":
        df.to_json(output_path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported table format '{suffix}' for {output_path}")
    return output_path


def write_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return output_path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_fingerprint(df: Any, *, columns: list[str] | None = None) -> str:
    selected = df[columns] if columns else df
    csv_text = selected.sort_index(axis=1).to_csv(index=False)
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
