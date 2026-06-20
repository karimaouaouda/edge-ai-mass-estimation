"""Inference command handling and backend result normalization."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from edge_ai_mass.agent.camera import CaptureAdapter, CapturedFrame
from edge_ai_mass.agent.envelopes import utc_now_iso
from edge_ai_mass.pipeline.pipeline import (
    ObjectEstimate,
    Pipeline,
    PipelineResult,
    PipelineStageUpdate,
)


@dataclass(frozen=True, slots=True)
class InferenceStageDefinition:
    sequence: int
    title: str
    description: str
    running_progress: int
    completed_progress: int


INFERENCE_STAGES = {
    "source-acquisition": InferenceStageDefinition(
        1,
        "Acquiring source",
        "The Jetson is capturing or loading the requested inference frame.",
        5,
        15,
    ),
    "object-detection": InferenceStageDefinition(
        2,
        "Detecting objects",
        "The detector is locating waste objects in the captured frame.",
        20,
        45,
    ),
    "depth-estimation": InferenceStageDefinition(
        3,
        "Estimating depth",
        "The depth model is estimating object distance and geometry.",
        50,
        65,
    ),
    "mass-estimation": InferenceStageDefinition(
        4,
        "Estimating mass",
        "The cascade is estimating material, volume, and mass for each object.",
        70,
        85,
    ),
    "result-normalization": InferenceStageDefinition(
        5,
        "Preparing result",
        "The device is converting model outputs into the backend result format.",
        88,
        94,
    ),
    "media-output": InferenceStageDefinition(
        6,
        "Preparing media",
        "The device is rendering and uploading the requested inference media.",
        96,
        100,
    ),
}


class InferenceStageReporter:
    """Create backend-compatible stage updates for one inference request."""

    def __init__(
        self,
        *,
        request_id: str | None,
        correlation_id: str | None,
        publish: Callable[[dict[str, Any]], None],
    ) -> None:
        self.request_id = request_id
        self.correlation_id = correlation_id
        self.publish = publish
        self._started_at: dict[str, str] = {}
        self._active_stage: str | None = None
        self._terminal_stages: set[str] = set()

    def update(
        self,
        stage_key: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if stage_key not in INFERENCE_STAGES:
            raise ValueError(f"Unknown inference stage: {stage_key}")
        if status not in {"pending", "running", "completed", "failed"}:
            raise ValueError(f"Invalid inference stage status: {status}")
        if stage_key in self._terminal_stages:
            return

        definition = INFERENCE_STAGES[stage_key]
        now = utc_now_iso()
        started_at = self._started_at.setdefault(stage_key, now)
        completed_at = now if status in {"completed", "failed"} else None
        progress = (
            definition.completed_progress
            if status == "completed"
            else definition.running_progress
        )
        payload = {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "stage_key": stage_key,
            "sequence": definition.sequence,
            "status": status,
            "title": definition.title,
            "description": definition.description,
            "progress_percent": int(max(0, min(100, progress))),
            "started_at": started_at,
            "completed_at": completed_at,
            "metadata": dict(metadata or {}),
        }
        self.publish(payload)

        if status == "running":
            self._active_stage = stage_key
        elif status in {"completed", "failed"}:
            self._terminal_stages.add(stage_key)
            if self._active_stage == stage_key:
                self._active_stage = None

    def fail_active(self, error: Exception | str) -> None:
        if self._active_stage is not None:
            self.update(
                self._active_stage,
                "failed",
                {"error": str(error)},
            )

    def handle_pipeline_update(self, update: PipelineStageUpdate) -> None:
        self.update(update.stage_key, update.status, update.metadata)


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
        stage_reporter: InferenceStageReporter | None = None,
    ) -> InferenceCommandResult:
        source_type = str(payload.get("source_type") or "camera")
        source_reference = str(payload.get("source_reference") or "camera:0")
        if stage_reporter is not None:
            stage_reporter.update(
                "source-acquisition",
                "running",
                {"source_type": source_type, "source_reference": source_reference},
            )
        try:
            print(f"Capturing source for inference: type={source_type} reference={source_reference}")
            captured = self.capture_adapter.capture(
                source_type=source_type,
                source_reference=source_reference,
            )
        except Exception as exc:
            print(f"Failed to capture source for inference: {exc}")
            if stage_reporter is not None:
                stage_reporter.fail_active(exc)
            raise
        if stage_reporter is not None:
            stage_reporter.update(
                "source-acquisition",
                "completed",
                {
                    "source_type": source_type,
                    "source_reference": source_reference,
                    "width": int(captured.image.shape[1]),
                    "height": int(captured.image.shape[0]),
                },
            )
        options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
        started = time.perf_counter()
        result = self.pipeline.run(
            captured.image,
            stage_callback=(
                stage_reporter.handle_pipeline_update if stage_reporter is not None else None
            ),
        )
        latency_ms = result.frame_time_ms or ((time.perf_counter() - started) * 1000.0)
        if result.frame_time_ms <= 0:
            result.frame_time_ms = latency_ms
        if stage_reporter is not None:
            stage_reporter.update(
                "result-normalization",
                "running",
                {"object_count": len(result.objects)},
            )
        print(f"Normalizing pipeline result: {len(result.objects)} objects")
        try:
            normalized = normalize_pipeline_result(
                result,
                request_id=request_id,
                correlation_id=correlation_id,
                model_versions=self.active_models,
                max_objects=_max_objects(options),
            )
        except Exception as exc:
            if stage_reporter is not None:
                stage_reporter.fail_active(exc)
            raise
        if stage_reporter is not None:
            stage_reporter.update(
                "result-normalization",
                "completed",
                {
                    "object_count": normalized["object_count"],
                    "latency_ms": normalized["latency_ms"],
                },
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
