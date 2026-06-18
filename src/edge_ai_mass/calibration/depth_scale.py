"""Depth-map scaling utilities for monocular depth models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DepthScaleConfig:
    """Configuration for converting raw model depth into metres."""

    output_type: str = "metric"
    scale: float = 1.0
    offset: float = 0.0
    min_depth_m: float = 0.01
    max_depth_m: float | None = None
    invalid_value: float = 0.0
    calibration_id: str = "depth-scale-inline"

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> DepthScaleConfig:
        if not config:
            return cls()
        return cls(
            output_type=str(config.get("output_type", "metric")).lower(),
            scale=float(config.get("scale", 1.0)),
            offset=float(config.get("offset", 0.0)),
            min_depth_m=float(config.get("min_depth_m", 0.01)),
            max_depth_m=(
                float(config["max_depth_m"]) if config.get("max_depth_m") is not None else None
            ),
            invalid_value=float(config.get("invalid_value", 0.0)),
            calibration_id=str(config.get("calibration_id", "depth-scale-inline")),
        )


def raw_depth_to_metric(
    raw_depth: np.ndarray,
    config: DepthScaleConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert model depth output to metres and return ``(depth, valid_mask)``.

    Supported output types:
    - ``metric``: ``depth_m = raw * scale + offset``
    - ``relative``: same transform, but metadata can still record that the source
      needed scene-specific scaling.
    - ``inverse_depth``: ``depth_m = scale / (raw + offset)``
    """
    cfg = config or DepthScaleConfig()
    raw = np.asarray(raw_depth, dtype=np.float32)

    if cfg.output_type in {"metric", "relative"}:
        depth = (raw * cfg.scale) + cfg.offset
    elif cfg.output_type in {"inverse", "inverse_depth", "disparity"}:
        denom = raw + cfg.offset
        depth = np.full_like(raw, cfg.invalid_value, dtype=np.float32)
        valid_denom = np.abs(denom) > 1e-9
        depth[valid_denom] = cfg.scale / denom[valid_denom]
    else:
        raise ValueError(
            "Depth output_type must be 'metric', 'relative', or 'inverse_depth'."
        )

    valid = np.isfinite(depth) & (depth > cfg.min_depth_m)
    if cfg.max_depth_m is not None:
        valid &= depth <= cfg.max_depth_m
        depth = np.minimum(depth, cfg.max_depth_m)

    depth = np.where(valid, depth, cfg.invalid_value).astype(np.float32, copy=False)
    return depth, valid.astype(np.uint8)
