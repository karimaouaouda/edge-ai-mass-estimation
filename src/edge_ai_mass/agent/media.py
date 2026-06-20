"""Media rendering helpers for backend uploads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edge_ai_mass.pipeline.pipeline import PipelineResult


@dataclass(frozen=True, slots=True)
class MediaUploadJob:
    """Local media file ready to upload through the backend HTTP endpoint."""

    path: Path
    media_type: str
    request_id: str | None
    correlation_id: str | None
    metadata: dict[str, object]


class MediaRenderer:
    """Create original, annotated, and depth-preview files for workflows."""

    def __init__(self, media_dir: str | Path) -> None:
        self.media_dir = Path(media_dir)

    def render_requested(
        self,
        *,
        image: np.ndarray,
        result: PipelineResult,
        request_id: str | None,
        correlation_id: str | None,
        return_annotated_image: bool = False,
        return_depth_preview: bool = False,
    ) -> list[MediaUploadJob]:
        self.media_dir.mkdir(parents=True, exist_ok=True)
        jobs: list[MediaUploadJob] = []
        workflow_id = request_id or correlation_id or "manual"

        if return_annotated_image:
            path = self.media_dir / f"{workflow_id}-annotated.jpg"
            self.write_annotated(image, result, path)
            jobs.append(
                MediaUploadJob(
                    path=path,
                    media_type="annotated",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    metadata={"object_count": len(result.objects)},
                )
            )

        if return_depth_preview and result.depth_map is not None:
            path = self.media_dir / f"{workflow_id}-depth-preview.jpg"
            self.write_depth_preview(result.depth_map, path)
            jobs.append(
                MediaUploadJob(
                    path=path,
                    media_type="depth_preview",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    metadata={},
                )
            )

        return jobs

    def write_annotated(self, image: np.ndarray, result: PipelineResult, path: str | Path) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to render annotated images") from exc

        output = image.copy()
        palette = (
            (0, 255, 0),
            (255, 191, 0),
            (255, 0, 255),
            (0, 165, 255),
            (255, 255, 0),
        )
        mask_alpha = 0.35
        for index, obj in enumerate(result.objects):
            detection = obj.detection
            color = palette[index % len(palette)]

            if detection.mask is not None:
                mask = np.asarray(detection.mask)
                mask = np.squeeze(mask)
                if mask.ndim == 2:
                    if mask.shape != output.shape[:2]:
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (output.shape[1], output.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    mask_pixels = mask > 0
                    if np.any(mask_pixels):
                        color_array = np.asarray(color, dtype=np.float32)
                        blended = (
                            output[mask_pixels].astype(np.float32) * (1.0 - mask_alpha)
                            + color_array * mask_alpha
                        )
                        output[mask_pixels] = np.clip(blended, 0, 255).astype(output.dtype)
                        contours, _ = cv2.findContours(
                            mask_pixels.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        cv2.drawContours(output, contours, -1, color, 2)

            x1, y1, x2, y2 = detection.bbox.astype(int)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            grams = (obj.mass_kg or 0.0) * 1000.0
            label = f"{detection.class_name} {grams:.0f}g"
            cv2.putText(
                output,
                label,
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), output)

    def write_depth_preview(self, depth_map: np.ndarray, path: str | Path) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to render depth previews") from exc

        finite = np.asarray(depth_map, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            preview = np.zeros(depth_map.shape[:2], dtype=np.uint8)
        else:
            min_value = float(np.min(finite))
            max_value = float(np.max(finite))
            denom = max(max_value - min_value, 1e-6)
            preview = np.clip((depth_map - min_value) / denom * 255.0, 0, 255).astype(np.uint8)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), preview)
