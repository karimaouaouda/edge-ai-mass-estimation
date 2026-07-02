"""Device resolution helpers for YOLO training backends."""

from __future__ import annotations

import importlib
from typing import Any


DIRECTML_ALIASES = {"directml", "dml", "torch-directml", "torch_directml"}


def is_directml_device(value: Any) -> bool:
    """Return true when a config value or torch device targets DirectML."""
    text = str(value).strip().lower()
    if text in DIRECTML_ALIASES:
        return True
    if any(text.startswith(f"{alias}:") for alias in DIRECTML_ALIASES):
        return True
    return text.startswith("privateuseone")


def is_cpu_device(value: Any) -> bool:
    """Return true for the explicit CPU device selector."""
    return isinstance(value, str) and value.strip().lower() == "cpu"


def resolve_device_for_ultralytics(value: Any, *, purpose: str) -> Any:
    """Resolve user-friendly device aliases into values accepted by Ultralytics.

    CUDA, CPU, MPS, and numeric device strings are returned unchanged. DirectML
    is resolved to a real ``torch.device`` through ``torch-directml`` because
    Ultralytics accepts torch devices but does not understand the literal
    string ``"directml"``.
    """
    if not is_directml_device(value):
        return value

    torch_directml = _require_torch_directml(purpose=purpose)
    index = _directml_index(value)
    try:
        return torch_directml.device(index)
    except TypeError:
        # Older torch-directml releases expose device() without an index.
        return torch_directml.device()


def resolve_export_device(value: Any, *, purpose: str) -> Any:
    """Return a safe export/pre-export device.

    DirectML accelerates PyTorch execution. It is not an Ultralytics export
    backend in this pipeline, so exports are performed on CPU when the user
    selected DirectML for training or evaluation.
    """
    if is_directml_device(value):
        print(
            f"[device] {purpose}: DirectML requested, using CPU for export/pre-export "
            "because DirectML is a PyTorch execution backend, not an export backend."
        )
        return "cpu"
    return value


def device_to_log_value(value: Any) -> str:
    """Convert a device object to a stable string for tracking metadata."""
    return str(value)


def requires_torch_directml(*values: Any) -> bool:
    """Return true if any configured device requires torch-directml."""
    return any(is_directml_device(value) for value in values if value is not None)


def _require_torch_directml(*, purpose: str) -> Any:
    try:
        return importlib.import_module("torch_directml")
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            f"DirectML was requested for {purpose}, but torch-directml is not available. "
            'Install it with: pip install -e ".[directml]" or pip install torch-directml.'
        ) from exc


def _directml_index(value: Any) -> int:
    text = str(value).strip().lower()
    if ":" not in text:
        return 0
    suffix = text.rsplit(":", 1)[-1]
    if not suffix:
        return 0
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(
            f"Invalid DirectML device selector {value!r}; use 'directml' or 'directml:0'."
        ) from exc
