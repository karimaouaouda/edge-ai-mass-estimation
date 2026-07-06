"""Background-aware object geometry and volume estimation."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edge_ai_mass.calibration.depth_scale import DepthScaleConfig, raw_depth_to_metric
from edge_ai_mass.calibration.runtime import RuntimeCalibration, load_runtime_calibration
from edge_ai_mass.pipeline.pipeline import Detection

logger = logging.getLogger(__name__)


@dataclass
class ObjectGeometry:
    """Metric geometry for one detected object."""

    width_m: float = 0.0
    height_m: float = 0.0
    projected_area_m2: float = 0.0
    mean_object_depth_m: float = 0.0
    mean_background_depth_m: float = 0.0
    mean_height_m: float = 0.0
    max_height_m: float = 0.0
    volume_m3: float = 0.0
    pixel_count: int = 0
    valid_pixel_count: int = 0
    method: str = "unknown"
    calibration_id: str | None = None
    depth_scale_id: str | None = None
    background_id: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeometryEstimator:
    """Estimate object metric geometry from masks, depth, and background."""

    def __init__(
        self,
        *,
        calibration: RuntimeCalibration | None = None,
        depth_scale: DepthScaleConfig | None = None,
        background_depth: np.ndarray | None = None,
        background_depth_m: float | None = None,
        background_id: str | None = None,
        pixel_to_m: float = 0.001,
        default_thickness_m: float = 0.01,
        min_height_m: float = 0.0,
        max_height_m: float | None = 1.0,
        height_mode: str = "background_minus_object",
    ) -> None:
        self.calibration = calibration
        self.depth_scale = depth_scale or DepthScaleConfig()
        self.background_depth = background_depth
        self.background_depth_m = background_depth_m
        self.background_id = background_id
        self.pixel_to_m = float(pixel_to_m)
        self.default_thickness_m = float(default_thickness_m)
        self.min_height_m = float(min_height_m)
        self.max_height_m = max_height_m
        self.height_mode = height_mode

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GeometryEstimator:
        calibration = load_runtime_calibration(config.get("calibration"))
        depth_scale = DepthScaleConfig.from_config(config.get("depth_scale"))
        background_cfg = config.get("background") or {}
        geometry_cfg = config.get("geometry") or {}

        background_depth = None
        background_id = None
        depth_path = background_cfg.get("depth_path")
        if depth_path:
            path = Path(depth_path)
            if path.exists():
                background_depth = _load_depth_file(path)
                if not background_cfg.get("metric", False):
                    background_depth, _ = raw_depth_to_metric(background_depth, depth_scale)
                background_id = str(background_cfg.get("background_id") or path.stem)
            elif background_cfg.get("required", False):
                raise FileNotFoundError(f"Background depth file not found: {path}")
            else:
                logger.warning("Background depth file not found: %s", path)

        background_depth_m = background_cfg.get("depth_m")
        return cls(
            calibration=calibration,
            depth_scale=depth_scale,
            background_depth=background_depth,
            background_depth_m=float(background_depth_m) if background_depth_m is not None else None,
            background_id=background_id or background_cfg.get("background_id"),
            pixel_to_m=float(geometry_cfg.get("pixel_to_m", 0.001)),
            default_thickness_m=float(geometry_cfg.get("default_thickness_m", 0.01)),
            min_height_m=float(geometry_cfg.get("min_height_m", 0.0)),
            max_height_m=(
                float(geometry_cfg["max_height_m"])
                if geometry_cfg.get("max_height_m") is not None
                else None
            ),
            height_mode=str(geometry_cfg.get("height_mode", "background_minus_object")),
        )

    def metric_depth(self, raw_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return raw_depth_to_metric(raw_depth, self.depth_scale)

    def estimate(
        self,
        detection: Detection,
        metric_depth: np.ndarray,
        image_shape: tuple[int, ...],
        valid_depth_mask: np.ndarray | None = None,
    ) -> ObjectGeometry:
        warnings: list[str] = []
        depth = np.asarray(metric_depth, dtype=np.float32)
        mask = _mask_for_depth_shape(depth.shape, detection.mask, detection.bbox)
        pixel_count = int(np.sum(mask > 0))
        if pixel_count == 0:
            return ObjectGeometry(
                method="empty_mask",
                calibration_id=self._calibration_id(),
                depth_scale_id=self.depth_scale.calibration_id,
                background_id=self.background_id,
                warnings=["empty_object_mask"],
            )

        valid = np.isfinite(depth) & (depth > 0)
        if valid_depth_mask is not None:
            valid &= _resize_bool_mask(valid_depth_mask, depth.shape)
        object_region = (mask > 0) & valid
        valid_pixel_count = int(np.sum(object_region))
        has_valid_object_depth = valid_pixel_count > 0
        if not has_valid_object_depth:
            warnings.append("no_valid_object_depth")
            # Keep the mask support for area estimation, but remember that no
            # real object-depth pixels were available.  Later we must not treat
            # raw invalid zero-depth values as real object depths, otherwise a
            # 1 m background becomes a fake 1 m object height.
            object_region = mask > 0
            support_pixel_count = pixel_count
        else:
            support_pixel_count = valid_pixel_count

        object_depth_values = depth[object_region]
        object_depth = float(np.median(object_depth_values[object_depth_values > 0])) if np.any(object_depth_values > 0) else 0.0
        if object_depth <= 0:
            object_depth = float(self.background_depth_m or 1.0)
            warnings.append("using_default_object_depth")

        background = self._background_for(depth.shape)
        if background is None:
            background_values = np.full(support_pixel_count, object_depth + self.default_thickness_m)
            warnings.append("missing_background_depth")
            method = "mask_depth_default_thickness"
        else:
            background_values = background[object_region]
            background_values = background_values[np.isfinite(background_values) & (background_values > 0)]
            if background_values.size == 0:
                background_values = np.full(support_pixel_count, object_depth + self.default_thickness_m)
                warnings.append("invalid_background_depth")
                method = "mask_depth_default_thickness"
            else:
                method = "mask_depth_background"

        if background_values.size != support_pixel_count:
            background_fill = float(np.median(background_values)) if background_values.size else object_depth
            full_background = np.full(support_pixel_count, background_fill, dtype=np.float32)
            full_background[: min(support_pixel_count, background_values.size)] = background_values[
                : min(support_pixel_count, background_values.size)
            ]
            background_values = full_background

        object_values = depth[object_region].astype(np.float32)
        background_values = background_values.astype(np.float32)
        if not has_valid_object_depth:
            heights = np.full_like(background_values, self.default_thickness_m, dtype=np.float32)
            method = "mask_default_thickness_no_object_depth"
            warnings.append("using_default_thickness")
        else:
            heights = self._height_values(background_values, object_values)
        positive = heights > self.min_height_m
        if not np.any(positive):
            heights = np.full_like(object_values, self.default_thickness_m, dtype=np.float32)
            positive = heights > 0
            warnings.append("using_default_thickness")

        if self.max_height_m is not None:
            heights = np.minimum(heights, self.max_height_m)

        pixel_area = self._pixel_area(background_values)
        volume = float(np.sum(pixel_area[positive] * heights[positive]))
        projected_area = float(np.sum(pixel_area[positive]))

        x1, y1, x2, y2 = detection.bbox.astype(float)
        metric_depth_for_size = float(np.median(background_values)) if background_values.size else object_depth
        width_m, height_m = self._bbox_size_m(x2 - x1, y2 - y1, metric_depth_for_size)

        return ObjectGeometry(
            width_m=width_m,
            height_m=height_m,
            projected_area_m2=projected_area,
            mean_object_depth_m=float(np.mean(object_values)) if object_values.size else 0.0,
            mean_background_depth_m=float(np.mean(background_values)) if background_values.size else 0.0,
            mean_height_m=float(np.mean(heights[positive])) if np.any(positive) else 0.0,
            max_height_m=float(np.max(heights[positive])) if np.any(positive) else 0.0,
            volume_m3=volume,
            pixel_count=pixel_count,
            valid_pixel_count=valid_pixel_count,
            method=method,
            calibration_id=self._calibration_id(),
            depth_scale_id=self.depth_scale.calibration_id,
            background_id=self.background_id,
            warnings=warnings,
        )

    def _height_values(self, background_values: np.ndarray, object_values: np.ndarray) -> np.ndarray:
        if self.height_mode in {"object_minus_background", "closer_is_larger"}:
            return object_values - background_values
        return background_values - object_values

    def _pixel_area(self, depth_values: np.ndarray) -> np.ndarray:
        if self.calibration is not None:
            return np.asarray(self.calibration.pixel_area_m2(depth_values), dtype=np.float32)
        return np.full_like(depth_values, self.pixel_to_m**2, dtype=np.float32)

    def _bbox_size_m(self, width_px: float, height_px: float, depth_m: float) -> tuple[float, float]:
        if self.calibration is not None:
            return (
                self.calibration.pixel_width_m(width_px, depth_m),
                self.calibration.pixel_height_m(height_px, depth_m),
            )
        return float(width_px) * self.pixel_to_m, float(height_px) * self.pixel_to_m

    def _background_for(self, depth_shape: tuple[int, ...]) -> np.ndarray | None:
        if self.background_depth is not None:
            return _resize_depth(self.background_depth, depth_shape)
        if self.background_depth_m is not None:
            return np.full(depth_shape[:2], float(self.background_depth_m), dtype=np.float32)
        return None

    def _calibration_id(self) -> str | None:
        return self.calibration.calibration_id if self.calibration is not None else None


def _load_depth_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read background depth file: {path}")
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    return depth.astype(np.float32)


def _resize_depth(depth: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    target_h, target_w = target_shape[:2]
    if depth.shape[:2] == (target_h, target_w):
        return depth.astype(np.float32, copy=False)
    return cv2.resize(depth.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _resize_bool_mask(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    target_h, target_w = target_shape[:2]
    if mask.shape[:2] == (target_h, target_w):
        return mask.astype(bool)
    resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _mask_for_depth_shape(
    depth_shape: tuple[int, ...], mask: np.ndarray | None, bbox: np.ndarray
) -> np.ndarray:
    target_h, target_w = depth_shape[:2]
    if mask is not None:
        if mask.shape[:2] == (target_h, target_w):
            return mask.astype(np.uint8, copy=False)
        resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return resized.astype(np.uint8, copy=False)

    result = np.zeros((target_h, target_w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox.astype(int)
    x1 = max(0, min(target_w, x1))
    x2 = max(0, min(target_w, x2))
    y1 = max(0, min(target_h, y1))
    y2 = max(0, min(target_h, y2))
    if x2 > x1 and y2 > y1:
        result[y1:y2, x1:x2] = 1
    return result
