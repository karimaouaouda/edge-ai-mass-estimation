"""JSON-safe payload codecs for host-executed inference stages.

The edge pipeline passes rich Python objects between stages: OpenCV images,
NumPy depth maps, segmentation masks, ``Detection`` objects, and nested feature
dictionaries.  The host server keeps the same contracts by wrapping NumPy arrays
as base64-encoded ``.npy`` blobs and reconstructing the domain objects on the
other side.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

from edge_ai_mass.modules.base import ModuleResult
from edge_ai_mass.pipeline.pipeline import Detection, ObjectEstimate, PipelineResult

TYPE_KEY = "__edge_ai_type__"
ARRAY_TYPE = "ndarray"
DETECTION_TYPE = "detection"
OBJECT_ESTIMATE_TYPE = "object_estimate"
PIPELINE_RESULT_TYPE = "pipeline_result"


def encode_value(value: Any) -> Any:
    """Recursively encode pipeline values into JSON-compatible objects."""

    if isinstance(value, np.ndarray):
        return encode_array(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Detection):
        return {
            TYPE_KEY: DETECTION_TYPE,
            "bbox": encode_value(value.bbox),
            "mask": encode_value(value.mask) if value.mask is not None else None,
            "class_id": int(value.class_id),
            "class_name": str(value.class_name),
            "confidence": float(value.confidence),
        }
    if isinstance(value, ObjectEstimate):
        return {
            TYPE_KEY: OBJECT_ESTIMATE_TYPE,
            "detection": encode_value(value.detection),
            "depth_stats": encode_value(value.depth_stats),
            "geometry": encode_value(value.geometry),
            "volume_m3": value.volume_m3,
            "volume_method": value.volume_method,
            "material": value.material,
            "mass_kg": value.mass_kg,
            "mass_method": value.mass_method,
            "mass_features": encode_value(value.mass_features),
            "calibration_id": value.calibration_id,
            "depth_scale_id": value.depth_scale_id,
            "background_id": value.background_id,
            "warnings": encode_value(value.warnings),
        }
    if isinstance(value, PipelineResult):
        return encode_pipeline_result(value)
    if isinstance(value, dict):
        return {str(key): encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]
    return value


def decode_value(value: Any) -> Any:
    """Recursively restore values encoded by :func:`encode_value`."""

    if isinstance(value, list):
        return [decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    payload_type = value.get(TYPE_KEY)
    if payload_type == ARRAY_TYPE:
        return decode_array(value)
    if payload_type == DETECTION_TYPE:
        return Detection(
            bbox=np.asarray(decode_value(value["bbox"]), dtype=np.float32),
            mask=(
                np.asarray(decode_value(value["mask"]), dtype=np.uint8)
                if value.get("mask") is not None
                else None
            ),
            class_id=int(value["class_id"]),
            class_name=str(value["class_name"]),
            confidence=float(value["confidence"]),
        )
    if payload_type == OBJECT_ESTIMATE_TYPE:
        return ObjectEstimate(
            detection=decode_value(value["detection"]),
            depth_stats=decode_value(value.get("depth_stats") or {}),
            geometry=decode_value(value.get("geometry")),
            volume_m3=_optional_float(value.get("volume_m3")),
            volume_method=str(value.get("volume_method") or ""),
            material=value.get("material"),
            mass_kg=_optional_float(value.get("mass_kg")),
            mass_method=str(value.get("mass_method") or ""),
            mass_features=decode_value(value.get("mass_features") or {}),
            calibration_id=value.get("calibration_id"),
            depth_scale_id=value.get("depth_scale_id"),
            background_id=value.get("background_id"),
            warnings=list(decode_value(value.get("warnings") or [])),
        )
    if payload_type == PIPELINE_RESULT_TYPE:
        return decode_pipeline_result(value)
    return {key: decode_value(item) for key, item in value.items()}


def encode_array(array: np.ndarray) -> dict[str, Any]:
    """Encode a NumPy array as an in-memory ``.npy`` payload."""

    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return {
        TYPE_KEY: ARRAY_TYPE,
        "encoding": "npy_base64",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    """Decode an array produced by :func:`encode_array`."""

    if payload.get("encoding") != "npy_base64":
        raise ValueError(f"Unsupported array encoding: {payload.get('encoding')!r}")
    raw = base64.b64decode(str(payload["data"]).encode("ascii"))
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    return np.asarray(array)


def module_result_to_payload(result: ModuleResult) -> dict[str, Any]:
    """Serialize a stage ``ModuleResult`` for an HTTP response."""

    return {
        "data": encode_value(result.data),
        "latency_ms": float(result.latency_ms),
        "metadata": encode_value(result.metadata),
    }


def module_result_from_payload(payload: dict[str, Any]) -> ModuleResult:
    """Restore a stage ``ModuleResult`` from an HTTP response payload."""

    return ModuleResult(
        data=decode_value(payload.get("data")),
        latency_ms=float(payload.get("latency_ms") or 0.0),
        metadata=decode_value(payload.get("metadata") or {}),
    )


def encode_pipeline_result(result: PipelineResult, *, include_depth_maps: bool = True) -> dict[str, Any]:
    """Serialize a full ``PipelineResult``."""

    payload: dict[str, Any] = {
        TYPE_KEY: PIPELINE_RESULT_TYPE,
        "objects": encode_value(result.objects),
        "latency_ms": encode_value(result.latency_ms),
        "frame_time_ms": float(result.frame_time_ms),
    }
    if include_depth_maps:
        payload["depth_map"] = (
            encode_value(result.depth_map) if result.depth_map is not None else None
        )
        payload["raw_depth_map"] = (
            encode_value(result.raw_depth_map)
            if result.raw_depth_map is not None
            else None
        )
    return payload


def decode_pipeline_result(payload: dict[str, Any]) -> PipelineResult:
    """Restore a full ``PipelineResult``."""

    return PipelineResult(
        objects=list(decode_value(payload.get("objects") or [])),
        depth_map=decode_value(payload.get("depth_map")),
        raw_depth_map=decode_value(payload.get("raw_depth_map")),
        latency_ms=decode_value(payload.get("latency_ms") or {}),
        frame_time_ms=float(payload.get("frame_time_ms") or 0.0),
    )


def stage_request_payload(image: np.ndarray, kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a generic stage-inference request body."""

    return {
        "image": encode_value(image),
        "kwargs": encode_value(kwargs or {}),
    }


def decode_stage_request(payload: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode a generic stage-inference request body."""

    if "image" not in payload:
        raise ValueError("stage request requires an 'image' payload")
    image = np.asarray(decode_value(payload["image"]))
    kwargs = decode_value(payload.get("kwargs") or {})
    if not isinstance(kwargs, dict):
        raise ValueError("stage request 'kwargs' must decode to a mapping")
    return image, kwargs


def pipeline_request_payload(image: np.ndarray, *, include_depth_maps: bool = False) -> dict[str, Any]:
    """Create a full-pipeline request body."""

    return {
        "image": encode_value(image),
        "include_depth_maps": bool(include_depth_maps),
    }


def decode_pipeline_request(payload: dict[str, Any]) -> tuple[np.ndarray, bool]:
    """Decode a full-pipeline request body."""

    if "image" not in payload:
        raise ValueError("pipeline request requires an 'image' payload")
    return np.asarray(decode_value(payload["image"])), bool(
        payload.get("include_depth_maps", False)
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
