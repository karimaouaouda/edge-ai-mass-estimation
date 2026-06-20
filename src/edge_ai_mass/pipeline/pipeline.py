"""Main inference pipeline with cascade and fallback architecture.

Design
------
The pipeline runs a sequence of *stages*.  Each stage has:

* A **primary** module (the best-quality model).
* An optional **fallback** module (lighter / simpler) that fires when the
  primary fails or exceeds a latency budget.

The cascade flow for a single frame is::

    Frame ──► Detection ──► Depth ──► Volume ──► Mass
                 │              │         │         │
              fallback       fallback  2-D prior  density-only

The ``Pipeline`` object loads the config once, instantiates modules lazily, and
exposes a single ``run(image)`` method that returns a unified result dict.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from edge_ai_mass.modules.base import BaseModule, ModuleResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------
@dataclass
class Detection:
    bbox: np.ndarray            # (x1, y1, x2, y2)
    mask: np.ndarray | None     # (H, W) binary mask if available
    class_id: int
    class_name: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.astype(float).tolist(),
            "mask_area_px": int(np.sum(self.mask > 0)) if self.mask is not None else None,
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": float(self.confidence),
        }


@dataclass
class ObjectEstimate:
    detection: Detection
    depth_stats: dict[str, float] = field(default_factory=dict)
    geometry: dict[str, Any] | None = None
    volume_m3: float | None = None
    volume_method: str = ""
    material: str | None = None
    mass_kg: float | None = None
    mass_method: str = ""       # "density", "regression", "hybrid", "2d_prior"
    calibration_id: str | None = None
    depth_scale_id: str | None = None
    background_id: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection": self.detection.to_dict(),
            "depth_stats": self.depth_stats,
            "geometry": self.geometry,
            "volume_m3": self.volume_m3,
            "volume_method": self.volume_method,
            "material": self.material,
            "mass_kg": self.mass_kg,
            "mass_method": self.mass_method,
            "calibration_id": self.calibration_id,
            "depth_scale_id": self.depth_scale_id,
            "background_id": self.background_id,
            "warnings": self.warnings,
        }


@dataclass
class PipelineResult:
    objects: list[ObjectEstimate] = field(default_factory=list)
    depth_map: np.ndarray | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    frame_time_ms: float = 0.0

    def to_dict(self, include_depth_map: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "objects": [obj.to_dict() for obj in self.objects],
            "latency_ms": self.latency_ms,
            "frame_time_ms": self.frame_time_ms,
        }
        if include_depth_map and self.depth_map is not None:
            payload["depth_map"] = self.depth_map.astype(float).tolist()
        return payload


@dataclass(frozen=True, slots=True)
class PipelineStageUpdate:
    """Observable state change from one stage of the inference cascade."""

    stage_key: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


PipelineStageCallback = Callable[[PipelineStageUpdate], None]


# ------------------------------------------------------------------
# Stage wrapper
# ------------------------------------------------------------------
class Stage:
    """Wraps a primary module with an optional fallback."""

    def __init__(
        self,
        name: str,
        primary: BaseModule,
        fallback: BaseModule | None = None,
        latency_budget_ms: float = float("inf"),
    ) -> None:
        self.name = name
        self.primary = primary
        self.fallback = fallback
        self.latency_budget_ms = latency_budget_ms
        self.primary_available = True

    def run(self, image: np.ndarray, **kwargs: Any) -> ModuleResult:
        if self.primary_available:
            try:
                result = self.primary.predict(image, **kwargs)
                if result.latency_ms <= self.latency_budget_ms:
                    result.metadata["source"] = f"{self.name}.primary"
                    return result
                logger.warning(
                    "%s primary exceeded budget (%.1f ms > %.1f ms) — trying fallback",
                    self.name,
                    result.latency_ms,
                    self.latency_budget_ms,
                )
            except Exception:
                logger.exception("%s primary failed — trying fallback", self.name)

        if self.fallback is None:
            raise RuntimeError(f"{self.name}: primary failed and no fallback configured")

        result = self.fallback.predict(image, **kwargs)
        result.metadata["source"] = f"{self.name}.fallback"
        return result


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------
class Pipeline:
    """Orchestrates the full frame-level inference cascade."""

    def __init__(self) -> None:
        self.stages: dict[str, Stage] = {}
        self.geometry_estimator: Any | None = None
        self._is_ready = False

    # ------------------------------------------------------------------
    # Construction helpers — called by the factory / config loader
    # ------------------------------------------------------------------
    def add_stage(self, name: str, stage: Stage) -> None:
        self.stages[name] = stage

    def set_geometry_estimator(self, geometry_estimator: Any | None) -> None:
        self.geometry_estimator = geometry_estimator

    def load_all(self) -> None:
        """Pre-load every module so the first frame isn't slow."""
        for name, stage in self.stages.items():
            logger.info("Loading stage %s …", name)
            try:
                stage.primary.load()
            except Exception as exc:
                if stage.fallback is None:
                    raise
                logger.warning("%s primary failed during load; disabling primary: %s", name, exc)
                stage.primary_available = False
            if stage.fallback is not None:
                stage.fallback.load()
        self._is_ready = True

    def warmup(self, input_shape: tuple[int, ...] = (480, 640, 3)) -> None:
        dummy = np.zeros(input_shape, dtype=np.uint8)
        self.run(dummy)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def run(
        self,
        image: np.ndarray,
        *,
        stage_callback: PipelineStageCallback | None = None,
    ) -> PipelineResult:
        t0 = time.perf_counter()
        result = PipelineResult()
        latencies: dict[str, float] = {}

        # 1. Detection ---------------------------------------------------
        _notify_stage(stage_callback, "object-detection", "running")
        try:
            det_result = self.stages["detection"].run(image)
        except Exception as exc:
            _notify_stage(
                stage_callback,
                "object-detection",
                "failed",
                {"error": str(exc)},
            )
            raise
        detections: list[Detection] = det_result.data
        latencies["detection"] = det_result.latency_ms
        _notify_stage(
            stage_callback,
            "object-detection",
            "completed",
            {
                "latency_ms": float(det_result.latency_ms),
                "object_count": len(detections),
                "module": det_result.metadata.get("source"),
            },
        )

        if not detections:
            skipped = {"skipped": True, "reason": "no_objects_detected"}
            _notify_stage(stage_callback, "depth-estimation", "running", skipped)
            _notify_stage(stage_callback, "depth-estimation", "completed", skipped)
            _notify_stage(stage_callback, "mass-estimation", "running", skipped)
            _notify_stage(stage_callback, "mass-estimation", "completed", skipped)
            result.latency_ms = latencies
            result.frame_time_ms = (time.perf_counter() - t0) * 1000
            return result

        # 2. Depth --------------------------------------------------------
        _notify_stage(stage_callback, "depth-estimation", "running")
        try:
            depth_result = self.stages["depth"].run(image)
            raw_depth_map: np.ndarray = depth_result.data
            valid_depth_mask = None
            if self.geometry_estimator is not None:
                depth_map, valid_depth_mask = self.geometry_estimator.metric_depth(raw_depth_map)
            else:
                depth_map = raw_depth_map
            result.depth_map = depth_map
            latencies["depth"] = depth_result.latency_ms
        except Exception as exc:
            _notify_stage(
                stage_callback,
                "depth-estimation",
                "failed",
                {"error": str(exc)},
            )
            raise
        _notify_stage(
            stage_callback,
            "depth-estimation",
            "completed",
            {
                "latency_ms": float(depth_result.latency_ms),
                "module": depth_result.metadata.get("source"),
            },
        )

        # 3. Per-object mass estimation -----------------------------------
        _notify_stage(
            stage_callback,
            "mass-estimation",
            "running",
            {"object_count": len(detections)},
        )
        try:
            mass_stage = self.stages["mass"]
            mass_latency_ms = 0.0
            mass_sources: set[str] = set()
            for det in detections:
                obj = ObjectEstimate(detection=det)

                # Extract depth stats inside the mask / bbox
                obj.depth_stats = _depth_stats_for_detection(depth_map, det)
                if self.geometry_estimator is not None:
                    geometry = self.geometry_estimator.estimate(
                        det,
                        depth_map,
                        image.shape,
                        valid_depth_mask=valid_depth_mask,
                    )
                    obj.geometry = geometry.to_dict()
                    obj.volume_m3 = geometry.volume_m3
                    obj.volume_method = geometry.method
                    obj.calibration_id = geometry.calibration_id
                    obj.depth_scale_id = geometry.depth_scale_id
                    obj.background_id = geometry.background_id
                    obj.warnings.extend(geometry.warnings)

                # Mass module expects a feature dict
                features = {
                    "bbox": det.bbox,
                    "mask": det.mask,
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "depth_stats": obj.depth_stats,
                    "geometry": obj.geometry,
                    "volume_m3": obj.volume_m3,
                    "image_crop": _crop(image, det.bbox),
                }
                mass_result = mass_stage.run(image, features=features)
                obj.mass_kg = mass_result.data.get("mass_kg")
                obj.volume_m3 = mass_result.data.get("volume_m3", obj.volume_m3)
                obj.volume_method = mass_result.data.get("volume_method", obj.volume_method)
                obj.material = mass_result.data.get("material")
                obj.mass_method = mass_result.metadata.get("method", "unknown")
                obj.warnings.extend(mass_result.data.get("warnings", []))
                result.objects.append(obj)
                mass_latency_ms += mass_result.latency_ms
                source = mass_result.metadata.get("source")
                if source:
                    mass_sources.add(str(source))
        except Exception as exc:
            _notify_stage(
                stage_callback,
                "mass-estimation",
                "failed",
                {"error": str(exc), "completed_objects": len(result.objects)},
            )
            raise

        latencies["mass"] = mass_latency_ms
        _notify_stage(
            stage_callback,
            "mass-estimation",
            "completed",
            {
                "latency_ms": float(mass_latency_ms),
                "object_count": len(result.objects),
                "modules": sorted(mass_sources),
            },
        )
        result.latency_ms = latencies
        result.frame_time_ms = (time.perf_counter() - t0) * 1000
        return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _notify_stage(
    callback: PipelineStageCallback | None,
    stage_key: str,
    status: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if callback is not None:
        callback(PipelineStageUpdate(stage_key, status, metadata or {}))


def _crop(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(int)
    return image[y1:y2, x1:x2]


def _depth_stats_for_detection(
    depth_map: np.ndarray, det: Detection
) -> dict[str, float]:
    """Compute mean / median / std depth inside the object region."""
    if det.mask is not None:
        mask = _mask_for_depth_map(depth_map, det.mask)
        pixels = depth_map[mask > 0]
    else:
        x1, y1, x2, y2 = det.bbox.astype(int)
        pixels = depth_map[y1:y2, x1:x2].ravel()

    if pixels.size == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": float(np.mean(pixels)),
        "median": float(np.median(pixels)),
        "std": float(np.std(pixels)),
        "min": float(np.min(pixels)),
        "max": float(np.max(pixels)),
    }


def _mask_for_depth_map(depth_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Resize a detection mask so it can be applied to the depth map safely."""
    if mask.shape == depth_map.shape:
        return mask

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}")

    depth_h, depth_w = depth_map.shape[:2]
    resized = cv2.resize(mask.astype(np.uint8), (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(mask.dtype, copy=False)
