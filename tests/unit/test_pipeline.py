"""Unit tests for pipeline data structures and helpers."""

import numpy as np

from edge_ai_mass.pipeline.pipeline import Detection, _crop, _depth_stats_for_detection


def test_crop():
    image = np.arange(100).reshape(10, 10).astype(np.uint8)
    bbox = np.array([2, 3, 7, 8])
    cropped = _crop(image, bbox)
    assert cropped.shape == (5, 5)


def test_depth_stats_with_mask():
    depth = np.ones((10, 10), dtype=np.float32) * 2.0
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 1
    det = Detection(
        bbox=np.array([3, 3, 7, 7]),
        mask=mask,
        class_id=0,
        class_name="plastic",
        confidence=0.9,
    )
    stats = _depth_stats_for_detection(depth, det)
    assert stats["mean"] == 2.0
    assert stats["std"] == 0.0


def test_depth_stats_bbox_only():
    depth = np.random.rand(10, 10).astype(np.float32)
    det = Detection(
        bbox=np.array([0, 0, 5, 5]),
        mask=None,
        class_id=1,
        class_name="glass",
        confidence=0.8,
    )
    stats = _depth_stats_for_detection(depth, det)
    assert "mean" in stats
    assert "median" in stats
