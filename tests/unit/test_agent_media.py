"""Tests for media rendered before backend upload."""

from __future__ import annotations

import cv2
import numpy as np

from edge_ai_mass.agent.media import MediaRenderer
from edge_ai_mass.pipeline.pipeline import Detection, ObjectEstimate, PipelineResult


def test_write_annotated_overlays_segmentation_mask(tmp_path):
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    result = PipelineResult(
        objects=[
            ObjectEstimate(
                detection=Detection(
                    bbox=np.array([5, 5, 30, 30], dtype=np.float32),
                    mask=mask,
                    class_id=0,
                    class_name="trash",
                    confidence=0.95,
                ),
                mass_kg=0.1,
            )
        ]
    )
    output_path = tmp_path / "annotated.png"

    MediaRenderer(tmp_path).write_annotated(image, result, output_path)

    rendered = cv2.imread(str(output_path))
    assert rendered is not None
    assert rendered[20, 20].tolist() == [0, 89, 0]
    assert rendered[35, 35].tolist() == [0, 0, 0]


def test_write_depth_preview_renders_constant_map_as_visible_color(tmp_path):
    output_path = tmp_path / "constant-depth.jpg"

    metadata = MediaRenderer(tmp_path).write_depth_preview(
        np.full((40, 40), 10.0, dtype=np.float32),
        output_path,
    )

    rendered = cv2.imread(str(output_path))
    assert rendered is not None
    assert rendered.mean() > 20
    assert metadata["min_depth_value"] == 10.0
    assert metadata["max_depth_value"] == 10.0
    assert metadata["valid_pixel_percent"] == 100.0


def test_render_requested_prefers_raw_depth_map_over_clipped_metric_map(tmp_path):
    raw_depth = np.tile(np.linspace(1.0, 255.0, 40, dtype=np.float32), (40, 1))
    result = PipelineResult(
        depth_map=np.full((40, 40), 10.0, dtype=np.float32),
        raw_depth_map=raw_depth,
    )

    jobs = MediaRenderer(tmp_path).render_requested(
        image=np.zeros((40, 40, 3), dtype=np.uint8),
        result=result,
        request_id="request-1",
        correlation_id="correlation-1",
        return_depth_preview=True,
    )

    rendered = cv2.imread(str(jobs[0].path))
    assert rendered is not None
    assert float(np.std(rendered)) > 20.0
    assert jobs[0].metadata["depth_source"] == "raw_model"
    assert jobs[0].metadata["max_depth_value"] == 255.0


def test_invalid_depth_map_writes_visible_diagnostic_instead_of_black_image(tmp_path):
    output_path = tmp_path / "invalid-depth.jpg"
    invalid_depth = np.full((40, 160), np.nan, dtype=np.float32)

    metadata = MediaRenderer(tmp_path).write_depth_preview(invalid_depth, output_path)

    rendered = cv2.imread(str(output_path))
    assert rendered is not None
    assert rendered.mean() > 20
    assert metadata["valid_pixel_percent"] == 0.0
