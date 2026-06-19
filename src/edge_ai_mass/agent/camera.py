"""Camera and image capture adapters for command handlers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class CaptureError(RuntimeError):
    """Raised when a requested source cannot be captured or loaded."""


@dataclass(slots=True)
class CapturedFrame:
    """Frame captured from camera or loaded from disk."""

    image: np.ndarray
    source_type: str
    source_reference: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VideoCaptureSpec:
    """OpenCV source and optional backend derived from a payload reference."""

    source: int | str
    backend: int | None = None


class CaptureAdapter:
    """Capture frames from backend source descriptors."""

    def capture(self, *, source_type: str, source_reference: str) -> CapturedFrame:
        source_type = source_type.lower().strip()
        if source_type == "camera":
            return self._capture_camera(source_reference)
        if source_type in {"file", "image", "uploaded_input"}:
            return self._load_image(source_type, source_reference)
        raise CaptureError(f"Unsupported source_type: {source_type}")

    def is_camera_connected(self, source_reference: str = "camera:0") -> bool:
        try:
            cap = open_video_capture(source_reference)
        except ImportError:
            return False
        try:
            return bool(cap.isOpened())
        finally:
            cap.release()

    def _capture_camera(self, source_reference: str) -> CapturedFrame:
        try:
            import cv2
        except ImportError as exc:
            raise CaptureError("OpenCV is required for camera capture") from exc

        spec = video_capture_spec(source_reference, cv2_module=cv2)
        cap = open_video_capture(source_reference, cv2_module=cv2)
        try:
            if not cap.isOpened():
                raise CaptureError(f"Camera source is not available: {source_reference}")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise CaptureError(f"Camera did not return a frame: {source_reference}")
            return CapturedFrame(
                image=frame,
                source_type="camera",
                source_reference=source_reference,
                metadata={"capture_source": spec.source},
            )
        finally:
            cap.release()

    def _load_image(self, source_type: str, source_reference: str) -> CapturedFrame:
        try:
            import cv2
        except ImportError as exc:
            raise CaptureError("OpenCV is required for image loading") from exc

        path = Path(source_reference)
        image = cv2.imread(str(path))
        if image is None:
            raise CaptureError(f"Could not read image source: {source_reference}")
        return CapturedFrame(
            image=image,
            source_type=source_type,
            source_reference=str(path),
            metadata={"path": str(path)},
        )


def video_capture_spec(
    source_reference: str,
    *,
    cv2_module: Any | None = None,
) -> VideoCaptureSpec:
    """Resolve backend `camera_source` values into OpenCV arguments."""

    source_reference = str(source_reference or "camera:0")
    if source_reference.startswith("camera:"):
        return VideoCaptureSpec(source=int(source_reference.split(":", 1)[1]))
    if source_reference.startswith("gstreamer:"):
        if cv2_module is None:
            import cv2 as cv2_module

        pipeline = source_reference.split(":", 1)[1]
        return VideoCaptureSpec(source=pipeline, backend=cv2_module.CAP_GSTREAMER)
    try:
        return VideoCaptureSpec(source=int(source_reference))
    except ValueError:
        return VideoCaptureSpec(source=source_reference)


def open_video_capture(
    source_reference: str,
    *,
    cv2_module: Any | None = None,
) -> Any:
    """Open an OpenCV capture for a backend camera source reference."""

    if cv2_module is None:
        import cv2 as cv2_module

    spec = video_capture_spec(source_reference, cv2_module=cv2_module)
    if spec.backend is None:
        return cv2_module.VideoCapture(spec.source)
    return cv2_module.VideoCapture(spec.source, spec.backend)


def _camera_index(source_reference: str) -> int:
    """Backward-compatible numeric camera index parser."""

    spec = video_capture_spec(source_reference)
    if not isinstance(spec.source, int):
        raise ValueError(f"Camera source is not a numeric index: {source_reference}")
    return spec.source
