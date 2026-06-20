"""Unit tests for pipeline data structures and helpers."""

import numpy as np

from edge_ai_mass.modules.base import ModuleResult
from edge_ai_mass.pipeline.pipeline import (
    Detection,
    Pipeline,
    _crop,
    _depth_stats_for_detection,
)


class FixedStage:
    def __init__(self, result):
        self.result = result

    def run(self, _image, **_kwargs):
        return self.result


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


def test_depth_stats_resizes_mask_to_depth_shape():
    depth = np.ones((20, 20), dtype=np.float32) * 3.0
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    det = Detection(
        bbox=np.array([2, 2, 8, 8]),
        mask=mask,
        class_id=0,
        class_name="plastic",
        confidence=0.9,
    )

    stats = _depth_stats_for_detection(depth, det)

    assert stats["mean"] == 3.0
    assert stats["median"] == 3.0


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


def test_pipeline_reports_completed_skipped_stages_when_no_objects_are_detected():
    pipeline = Pipeline()
    pipeline.stages["detection"] = FixedStage(
        ModuleResult([], latency_ms=3.5, metadata={"source": "detection.primary"})
    )
    updates = []

    result = pipeline.run(
        np.zeros((8, 8, 3), dtype=np.uint8),
        stage_callback=updates.append,
    )

    assert result.objects == []
    assert [(item.stage_key, item.status) for item in updates] == [
        ("object-detection", "running"),
        ("object-detection", "completed"),
        ("depth-estimation", "running"),
        ("depth-estimation", "completed"),
        ("mass-estimation", "running"),
        ("mass-estimation", "completed"),
    ]
    assert updates[1].metadata["object_count"] == 0
    assert updates[-1].metadata == {
        "skipped": True,
        "reason": "no_objects_detected",
    }


def test_pipeline_reports_detection_failure_before_raising():
    class FailingStage:
        def run(self, _image, **_kwargs):
            raise RuntimeError("detector unavailable")

    pipeline = Pipeline()
    pipeline.stages["detection"] = FailingStage()
    updates = []

    try:
        pipeline.run(
            np.zeros((8, 8, 3), dtype=np.uint8),
            stage_callback=updates.append,
        )
    except RuntimeError as exc:
        assert str(exc) == "detector unavailable"
    else:
        raise AssertionError("Expected detection failure")

    assert [(item.stage_key, item.status) for item in updates] == [
        ("object-detection", "running"),
        ("object-detection", "failed"),
    ]
    assert updates[-1].metadata["error"] == "detector unavailable"
