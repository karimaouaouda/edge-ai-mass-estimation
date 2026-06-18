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
            import cv2
        except ImportError:
            return False
        index = _camera_index(source_reference)
        cap = cv2.VideoCapture(index)
        try:
            return bool(cap.isOpened())
        finally:
            cap.release()

    def _capture_camera(self, source_reference: str) -> CapturedFrame:
        try:
            import cv2
        except ImportError as exc:
            raise CaptureError("OpenCV is required for camera capture") from exc

        index = _camera_index(source_reference)
        cap = cv2.VideoCapture(index)
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
                metadata={"camera_index": index},
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


def _camera_index(source_reference: str) -> int:
    source_reference = str(source_reference or "camera:0")
    if source_reference.startswith("camera:"):
        return int(source_reference.split(":", 1)[1])
    return int(source_reference)
