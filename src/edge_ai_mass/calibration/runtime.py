"""Runtime calibration helpers for metric geometry estimation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class RuntimeCalibration:
    """Camera intrinsics used by the inference-time geometry layer."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int] | None = None
    calibration_id: str = "unversioned"
    pixel_to_m_at_1m: float | None = None

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    @classmethod
    def from_file(cls, path: str | Path) -> RuntimeCalibration:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data, calibration_id=data.get("calibration_id") or path.stem)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], calibration_id: str | None = None
    ) -> RuntimeCalibration:
        image_size = data.get("image_size")
        return cls(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float32),
            dist_coeffs=np.asarray(data.get("dist_coeffs", []), dtype=np.float32),
            image_size=tuple(image_size) if image_size else None,
            calibration_id=str(calibration_id or data.get("calibration_id") or "inline"),
            pixel_to_m_at_1m=data.get("pixel_to_m_at_1m"),
        )

    def pixel_area_m2(self, depth_m: np.ndarray | float) -> np.ndarray | float:
        """Return projected area of one pixel at depth ``z`` in square metres."""
        return (np.asarray(depth_m, dtype=np.float32) ** 2) / max(self.fx * self.fy, 1e-9)

    def pixel_width_m(self, pixel_count: float, depth_m: float) -> float:
        return float(pixel_count) * float(depth_m) / max(self.fx, 1e-9)

    def pixel_height_m(self, pixel_count: float, depth_m: float) -> float:
        return float(pixel_count) * float(depth_m) / max(self.fy, 1e-9)


def load_runtime_calibration(config: dict[str, Any] | None) -> RuntimeCalibration | None:
    """Load runtime calibration from a config dict.

    The config may contain ``path`` pointing at a calibration JSON, or inline
    ``camera_matrix``/``dist_coeffs`` fields for tests and simple deployments.
    """
    if not config:
        return None

    path = config.get("path")
    if path:
        path = Path(path)
        if path.exists():
            return RuntimeCalibration.from_file(path)
        if config.get("required", False):
            raise FileNotFoundError(f"Calibration file not found: {path}")

    if config.get("camera_matrix") is not None:
        return RuntimeCalibration.from_dict(config)
    return None


def undistort_if_needed(
    image: np.ndarray,
    calibration: RuntimeCalibration | None,
    *,
    enabled: bool = False,
) -> np.ndarray:
    """Undistort an image when calibration and config request it."""
    if not enabled or calibration is None or calibration.dist_coeffs.size == 0:
        return image
    return cv2.undistort(image, calibration.camera_matrix, calibration.dist_coeffs)
