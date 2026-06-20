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
