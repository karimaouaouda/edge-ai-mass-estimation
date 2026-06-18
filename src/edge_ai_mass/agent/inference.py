"""Inference command handling and backend result normalization."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from edge_ai_mass.agent.camera import CaptureAdapter, CapturedFrame
from edge_ai_mass.pipeline.pipeline import ObjectEstimate, Pipeline, PipelineResult


@dataclass(frozen=True, slots=True)
class InferenceCommandResult:
    """Normalized inference command output."""

    payload: dict[str, Any]
    captured: CapturedFrame
    pipeline_result: PipelineResult


class InferenceRunner:
    """Run the existing cascade pipeline for backend inference commands."""

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        capture_adapter: CaptureAdapter | None = None,
        active_models: dict[str, str] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.capture_adapter = capture_adapter or CaptureAdapter()
        self.active_models = active_models or {}

    def run_command(
        self,
        payload: dict[str, Any],
        *,
        request_id: str | None,
        correlation_id: str | None,
    ) -> InferenceCommandResult:
        source_type = str(payload.get("source_type") or "camera")
        source_reference = str(payload.get("source_reference") or "camera:0")
        captured = self.capture_adapter.capture(
            source_type=source_type,
            source_reference=source_reference,
        )
        options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
        started = time.perf_counter()
        result = self.pipeline.run(captured.image)
        latency_ms = result.frame_time_ms or ((time.perf_counter() - started) * 1000.0)
        if result.frame_time_ms <= 0:
            result.frame_time_ms = latency_ms
        normalized = normalize_pipeline_result(
            result,
            request_id=request_id,
            correlation_id=correlation_id,
            model_versions=self.active_models,
            max_objects=_max_objects(options),
        )
        return InferenceCommandResult(
            payload=normalized,
            captured=captured,
            pipeline_result=result,
        )


def normalize_pipeline_result(
    result: PipelineResult,
    *,
    request_id: str | None,
    correlation_id: str | None,
    model_versions: dict[str, str] | None = None,
    max_objects: int | None = None,
) -> dict[str, Any]:
    """Convert local pipeline objects into the backend `inference.result` shape."""

    objects = result.objects[:max_objects] if max_objects else result.objects
    normalized_objects = [_normalize_object(obj) for obj in objects]
    masses = [
        item["estimated_mass_grams"]
        for item in normalized_objects
        if item.get("estimated_mass_grams") is not None
    ]
    materials = [
        str(item["material_label"])
        for item in normalized_objects
        if item.get("material_label")
    ]
    frame_time_ms = float(result.frame_time_ms or 0.0)
    return {
        "request_id": request_id,
        "correlation_id": correlation_id,
        "latency_ms": int(round(frame_time_ms)),
        "fps": 1000.0 / frame_time_ms if frame_time_ms > 0 else 0.0,
        "object_count": len(normalized_objects),
        "total_estimated_mass_grams": float(sum(masses)),
        "dominant_material": _dominant_material(materials),
        "model_versions": model_versions or {},
        "objects": normalized_objects,
    }


def _normalize_object(obj: ObjectEstimate) -> dict[str, Any]:
    detection = obj.detection
    x1, y1, x2, y2 = detection.bbox.astype(float)
    payload: dict[str, Any] = {
        "class_label": detection.class_name,
        "material_label": obj.material,
        "confidence": float(detection.confidence),
        "bbox": {
            "x": float(x1),
            "y": float(y1),
            "w": float(max(x2 - x1, 0.0)),
            "h": float(max(y2 - y1, 0.0)),
        },
        "estimated_geometry": obj.geometry or _geometry_from_mask_or_bbox(obj),
        "estimated_mass_grams": (
            float(obj.mass_kg) * 1000.0 if obj.mass_kg is not None else None
        ),
        "mass_method": obj.mass_method or None,
        "features": {
            "volume_m3": obj.volume_m3,
            "volume_method": obj.volume_method,
            "depth_stats": obj.depth_stats,
            "warnings": obj.warnings,
            "calibration_id": obj.calibration_id,
            "depth_scale_id": obj.depth_scale_id,
            "background_id": obj.background_id,
        },
    }
    return {key: value for key, value in payload.items() if value is not None}


def _geometry_from_mask_or_bbox(obj: ObjectEstimate) -> dict[str, Any]:
    detection = obj.detection
    geometry: dict[str, Any] = {}
    if detection.mask is not None:
        geometry["area_px"] = int(np.sum(detection.mask > 0))
    else:
        x1, y1, x2, y2 = detection.bbox.astype(float)
        geometry["area_px"] = float(max(x2 - x1, 0.0) * max(y2 - y1, 0.0))
    if obj.volume_m3 is not None:
        geometry["volume_m3"] = float(obj.volume_m3)
    return geometry


def _dominant_material(materials: list[str]) -> str | None:
    if not materials:
        return None
    counts: dict[str, int] = {}
    for material in materials:
        counts[material] = counts.get(material, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _max_objects(options: dict[str, Any]) -> int | None:
    value = options.get("max_objects")
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
