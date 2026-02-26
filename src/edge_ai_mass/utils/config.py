"""YAML configuration loader with environment variable interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, resolving ``${ENV_VAR}`` placeholders."""
    text = Path(path).read_text()
    # Replace ${VAR} or ${VAR:default} with environment values
    pattern = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")
    text = pattern.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), text)
    return yaml.safe_load(text)


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge multiple config dicts (later values win)."""
    result: dict[str, Any] = {}
    for cfg in configs:
        _deep_update(result, cfg)
    return result


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base
