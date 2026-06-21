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
                    metadata={
                        "object_count": len(result.objects),
                        "segmentation_mask_count": sum(
                            obj.detection.mask is not None for obj in result.objects
                        ),
                    },
                )
            )

        if return_depth_preview and result.depth_map is not None:
            path = self.media_dir / f"{workflow_id}-depth-preview.jpg"
            preview_source = (
                result.raw_depth_map
                if result.raw_depth_map is not None
                else result.depth_map
            )
            depth_metadata = self.write_depth_preview(preview_source, path)
            depth_metadata["depth_source"] = (
                "raw_model" if result.raw_depth_map is not None else "metric"
            )
            jobs.append(
                MediaUploadJob(
                    path=path,
                    media_type="depth_preview",
                    request_id=request_id,
                    correlation_id=correlation_id,
                    metadata=depth_metadata,
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

    def write_depth_preview(
        self,
        depth_map: np.ndarray,
        path: str | Path,
    ) -> dict[str, object]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to render depth previews") from exc

        depth = np.squeeze(np.asarray(depth_map, dtype=np.float32))
        if depth.ndim != 2:
            raise ValueError(f"Depth preview expects a 2-D map, got shape {depth.shape}")

        valid_mask = np.isfinite(depth) & (depth > 0)
        valid_values = depth[valid_mask]

        normalized = np.zeros(depth.shape, dtype=np.uint8)
        if valid_values.size:
            low, high = np.percentile(valid_values, (2.0, 98.0)).astype(float)
            dynamic_range = high - low
            if dynamic_range <= max(abs(high), 1.0) * 1e-6:
                normalized[valid_mask] = 127
            else:
                scaled = (depth[valid_mask] - low) / dynamic_range
                normalized[valid_mask] = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
            preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
            preview[~valid_mask] = 0
            min_value = float(np.min(valid_values))
            max_value = float(np.max(valid_values))
        else:
            low = high = min_value = max_value = 0.0
            preview = np.full((*depth.shape, 3), 32, dtype=np.uint8)
            cv2.putText(
                preview,
                "NO VALID DEPTH",
                (10, max(24, depth.shape[0] // 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), preview):
            raise RuntimeError(f"OpenCV could not encode depth preview: {output_path}")

        return {
            "min_depth_value": min_value,
            "max_depth_value": max_value,
            "display_percentile_low": float(low),
            "display_percentile_high": float(high),
            "valid_pixel_percent": float(np.mean(valid_mask) * 100.0),
            "visualization": "turbo_percentile_2_98",
        }
