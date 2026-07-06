"""Readable console rendering for the long-running edge agent.

The agent already emits machine-readable MQTT/API events.  This module is only
for the human watching a terminal: it groups important runtime messages into
small ASCII panels/tables so inference stages, failures, and host-fallback
decisions are easy to distinguish in a busy console log.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Mapping, Sequence
from typing import Any, TextIO

DEFAULT_WIDTH = 96
MIN_WIDTH = 72
MAX_WIDTH = 132


def print_panel(
    title: str,
    rows: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    *,
    status: str | None = None,
    stream: TextIO | None = None,
) -> None:
    """Print a compact key/value panel to the console."""

    print(
        format_panel(title, rows, status=status),
        file=stream or sys.stdout,
        flush=True,
    )


def print_stage_update(
    payload: Mapping[str, Any],
    *,
    stream: TextIO | None = None,
) -> None:
    """Print one inference-stage update as a small table."""

    print(
        format_stage_update(payload),
        file=stream or sys.stdout,
        flush=True,
    )


def format_panel(
    title: str,
    rows: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    *,
    status: str | None = None,
    width: int | None = None,
) -> str:
    """Return a boxed key/value panel."""

    width = _resolved_width(width)
    normalized_rows = _normalize_rows(rows)
    label_width = min(
        24,
        max([10, *(len(str(key)) for key, _ in normalized_rows)]),
    )
    value_width = max(12, width - label_width - 7)
    heading = _heading(title, status)

    lines = [_border(width), _content_line(heading, width), _border(width)]
    if normalized_rows:
        for key, value in normalized_rows:
            key_text = _clip(str(key), label_width).ljust(label_width)
            value_text = _clip(_stringify(value), value_width).ljust(value_width)
            lines.append(f"| {key_text} | {value_text} |")
    else:
        lines.append(_content_line("-", width))
    lines.append(_border(width))
    return "\n".join(lines)


def format_stage_update(
    payload: Mapping[str, Any],
    *,
    width: int | None = None,
) -> str:
    """Return a one-row stage-progress table."""

    width = _resolved_width(width)
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    title = str(payload.get("title") or payload.get("stage_key") or "stage")
    status = str(payload.get("status") or "-").upper()
    progress = payload.get("progress_percent")
    progress_text = f"{int(progress)}%" if isinstance(progress, int | float) else "-"
    detail = _stage_detail(metadata)

    columns = ["#", "stage", "status", "p%", "details"]
    fixed_widths = [1, 16, 9, 3]
    detail_width = max(
        12,
        width - sum(fixed_widths) - _table_overhead(len(columns)),
    )
    column_widths = [*fixed_widths, detail_width]
    row = [
        payload.get("sequence", "-"),
        payload.get("stage_key", "-"),
        status,
        progress_text,
        detail,
    ]

    lines = [
        _border(width),
        _content_line(f"INFERENCE STAGE :: {title}", width),
        _border(width),
        _table_line(columns, column_widths, width),
        _border(width),
        _table_line(row, column_widths, width),
        _border(width),
    ]
    return "\n".join(lines)


def _normalize_rows(
    rows: Mapping[str, Any] | Sequence[tuple[str, Any]] | None,
) -> list[tuple[str, Any]]:
    if rows is None:
        return []
    if isinstance(rows, Mapping):
        return [(str(key), value) for key, value in rows.items()]
    return [(str(key), value) for key, value in rows]


def _stage_detail(metadata: Mapping[str, Any]) -> str:
    preferred_keys = [
        ("module", "src"),
        ("modules", "src"),
        ("fallback_reason", "reason"),
        ("primary_error", "primary_error"),
        ("host_error", "host_error"),
        ("host_stage", "host"),
        ("object_count", "objects"),
        ("latency_ms", "latency_ms"),
        ("source_type", "source_type"),
        ("source_reference", "source"),
        ("uploaded_count", "uploaded"),
        ("queued_count", "queued"),
        ("error", "error"),
    ]
    parts: list[str] = []
    for key, label in preferred_keys:
        value = metadata.get(key)
        if value is None or value == "":
            continue
        if key == "module":
            value = _compact_module_source(value)
        parts.append(f"{label}={_stringify(value)}")
    if parts:
        return "; ".join(parts)
    if metadata:
        return "; ".join(
            f"{key}={_stringify(value)}"
            for key, value in sorted(metadata.items(), key=lambda item: str(item[0]))
            if value is not None and value != ""
        )
    return "-"


def _heading(title: str, status: str | None) -> str:
    if status:
        return f"{title} [{status.upper()}]"
    return title


def _compact_module_source(value: Any) -> Any:
    if isinstance(value, str) and "." in value:
        return value.rsplit(".", 1)[-1]
    return value


def _resolved_width(width: int | None = None) -> int:
    if width is None:
        width = shutil.get_terminal_size((DEFAULT_WIDTH, 24)).columns
    return max(MIN_WIDTH, min(MAX_WIDTH, int(width)))


def _border(width: int) -> str:
    return "+" + "-" * (width - 2) + "+"


def _content_line(content: Any, width: int) -> str:
    body_width = width - 4
    return f"| {_clip(_stringify(content), body_width).ljust(body_width)} |"


def _table_line(values: Sequence[Any], widths: Sequence[int], total_width: int) -> str:
    cells = [
        _clip(_stringify(value), width).ljust(width)
        for value, width in zip(values, widths, strict=False)
    ]
    line = "| " + " | ".join(cells) + " |"
    if len(line) < total_width:
        return line[:-1] + " " * (total_width - len(line)) + "|"
    if len(line) == total_width:
        return line
    return _clip(line, total_width - 1) + "|"


def _table_overhead(column_count: int) -> int:
    return (3 * column_count) + 1


def _clip(value: str, width: int) -> str:
    if width <= 0:
        return ""
    value = value.replace("\r", " ").replace("\n", " ")
    if len(value) <= width:
        return value
    if width <= 0:
        return ""
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def _stringify(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.1f}" if abs(value) >= 10 else f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, Mapping):
        return ", ".join(
            f"{key}={_stringify(inner)}"
            for key, inner in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_stringify(item) for item in value)
    return str(value)
