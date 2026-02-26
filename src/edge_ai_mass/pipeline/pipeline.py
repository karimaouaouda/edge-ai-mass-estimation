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
from typing import Any

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


@dataclass
class ObjectEstimate:
    detection: Detection
    depth_stats: dict[str, float] = field(default_factory=dict)
    volume_m3: float | None = None
    material: str | None = None
    mass_kg: float | None = None
    mass_method: str = ""       # "density", "regression", "hybrid", "2d_prior"


@dataclass
class PipelineResult:
    objects: list[ObjectEstimate] = field(default_factory=list)
    depth_map: np.ndarray | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    frame_time_ms: float = 0.0


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

    def run(self, image: np.ndarray, **kwargs: Any) -> ModuleResult:
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
        self._is_ready = False

    # ------------------------------------------------------------------
    # Construction helpers — called by the factory / config loader
    # ------------------------------------------------------------------
    def add_stage(self, name: str, stage: Stage) -> None:
        self.stages[name] = stage

    def load_all(self) -> None:
        """Pre-load every module so the first frame isn't slow."""
        for name, stage in self.stages.items():
            logger.info("Loading stage %s …", name)
            stage.primary.load()
            if stage.fallback is not None:
                stage.fallback.load()
        self._is_ready = True

    def warmup(self, input_shape: tuple[int, ...] = (480, 640, 3)) -> None:
        dummy = np.zeros(input_shape, dtype=np.uint8)
        self.run(dummy)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def run(self, image: np.ndarray) -> PipelineResult:
        t0 = time.perf_counter()
        result = PipelineResult()
        latencies: dict[str, float] = {}

        # 1. Detection ---------------------------------------------------
        det_result = self.stages["detection"].run(image)
        detections: list[Detection] = det_result.data
        latencies["detection"] = det_result.latency_ms

        if not detections:
            result.latency_ms = latencies
            result.frame_time_ms = (time.perf_counter() - t0) * 1000
            return result

        # 2. Depth --------------------------------------------------------
        depth_result = self.stages["depth"].run(image)
        depth_map: np.ndarray = depth_result.data
        result.depth_map = depth_map
        latencies["depth"] = depth_result.latency_ms

        # 3. Per-object mass estimation -----------------------------------
        mass_stage = self.stages["mass"]
        for det in detections:
            obj = ObjectEstimate(detection=det)

            # Extract depth stats inside the mask / bbox
            obj.depth_stats = _depth_stats_for_detection(depth_map, det)

            # Mass module expects a feature dict
            features = {
                "bbox": det.bbox,
                "mask": det.mask,
                "class_id": det.class_id,
                "class_name": det.class_name,
                "depth_stats": obj.depth_stats,
                "image_crop": _crop(image, det.bbox),
            }
            mass_result = mass_stage.run(image, features=features)
            obj.mass_kg = mass_result.data.get("mass_kg")
            obj.volume_m3 = mass_result.data.get("volume_m3")
            obj.material = mass_result.data.get("material")
            obj.mass_method = mass_result.metadata.get("method", "unknown")
            result.objects.append(obj)

        latencies["mass"] = mass_stage.primary.predict.__func__  # placeholder
        result.latency_ms = latencies
        result.frame_time_ms = (time.perf_counter() - t0) * 1000
        return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _crop(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(int)
    return image[y1:y2, x1:x2]


def _depth_stats_for_detection(
    depth_map: np.ndarray, det: Detection
) -> dict[str, float]:
    """Compute mean / median / std depth inside the object region."""
    if det.mask is not None:
        pixels = depth_map[det.mask > 0]
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
