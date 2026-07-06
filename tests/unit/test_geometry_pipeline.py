"""Tests for calibration-aware geometry and mass flow."""

from __future__ import annotations

import numpy as np
import pytest

from edge_ai_mass.calibration.depth_scale import DepthScaleConfig, raw_depth_to_metric
from edge_ai_mass.calibration.runtime import RuntimeCalibration
from edge_ai_mass.modules.base import BaseModule
from edge_ai_mass.modules.geometry import GeometryEstimator
from edge_ai_mass.modules.mass.density_estimator import DensityMassEstimator
from edge_ai_mass.pipeline.pipeline import Detection, Pipeline, Stage


def _calibration() -> RuntimeCalibration:
    return RuntimeCalibration(
        camera_matrix=np.array(
            [
                [100.0, 0.0, 5.0],
                [0.0, 100.0, 5.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        dist_coeffs=np.zeros(5, dtype=np.float32),
        image_size=(10, 10),
        calibration_id="test-calibration",
    )


def _detection() -> Detection:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    return Detection(
        bbox=np.array([2, 2, 8, 8], dtype=np.float32),
        mask=mask,
        class_id=0,
        class_name="plastic_bottle",
        confidence=0.99,
    )


def test_depth_scale_inverse_depth():
    raw = np.array([[2.0, 4.0]], dtype=np.float32)
    depth, valid = raw_depth_to_metric(
        raw,
        DepthScaleConfig(output_type="inverse_depth", scale=2.0, min_depth_m=0.0),
    )
    assert np.allclose(depth, [[1.0, 0.5]])
    assert valid.sum() == 2


def test_geometry_estimator_integrates_background_volume():
    depth = np.ones((10, 10), dtype=np.float32)
    depth[2:8, 2:8] = 0.8
    estimator = GeometryEstimator(
        calibration=_calibration(),
        background_depth_m=1.0,
        background_id="flat-bg",
    )

    geometry = estimator.estimate(_detection(), depth, (10, 10, 3))

    assert geometry.method == "mask_depth_background"
    assert geometry.pixel_count == 36
    assert geometry.valid_pixel_count == 36
    assert geometry.width_m == pytest.approx(0.06)
    assert geometry.height_m == pytest.approx(0.06)
    assert geometry.projected_area_m2 == pytest.approx(36 * 0.0001)
    assert geometry.mean_height_m == pytest.approx(0.2)
    assert geometry.volume_m3 == pytest.approx(36 * 0.0001 * 0.2)


def test_density_estimator_uses_geometry_volume_and_material_mapping():
    estimator = DensityMassEstimator(
        {
            "densities": {"plastic": 10.0, "other": 1.0},
            "class_to_material": {"plastic_bottle": "plastic"},
        }
    )
    estimator.load()
    result = estimator.predict(
        np.zeros((10, 10, 3), dtype=np.uint8),
        features={
            "class_name": "plastic_bottle",
            "geometry": {"volume_m3": 0.25, "method": "unit-test"},
            "volume_m3": 0.25,
            "depth_stats": {"mean": 1.0},
        },
    )

    assert result.metadata["method"] == "density"
    assert result.data["material"] == "plastic"
    assert result.data["mass_kg"] == pytest.approx(2.5)
    assert result.data["volume_method"] == "geometry"


class FakeDetector(BaseModule):
    def load(self) -> None:
        self._is_loaded = True

    def _forward(self, image: np.ndarray, **kwargs):
        return [_detection()]


class FakeTypoDetector(BaseModule):
    def load(self) -> None:
        self._is_loaded = True

    def _forward(self, image: np.ndarray, **kwargs):
        detection = _detection()
        detection.class_name = "plastic_battle"
        return [detection]


class FakeDepth(BaseModule):
    def load(self) -> None:
        self._is_loaded = True

    def _forward(self, image: np.ndarray, **kwargs):
        depth = np.ones(image.shape[:2], dtype=np.float32)
        depth[2:8, 2:8] = 0.8
        return depth


class FailingModule(BaseModule):
    def load(self) -> None:
        raise FileNotFoundError("missing checkpoint")

    def _forward(self, image: np.ndarray, **kwargs):
        raise AssertionError("disabled primary should not run")


def test_pipeline_runs_geometry_then_mass():
    pipeline = Pipeline()
    pipeline.add_stage("detection", Stage("detection", FakeDetector({})))
    pipeline.add_stage("depth", Stage("depth", FakeDepth({})))
    pipeline.add_stage(
        "mass",
        Stage(
            "mass",
            DensityMassEstimator(
                {
                    "densities": {"plastic": 10.0, "other": 1.0},
                    "class_to_material": {"plastic_bottle": "plastic"},
                }
            ),
        ),
    )
    pipeline.set_geometry_estimator(
        GeometryEstimator(
            calibration=_calibration(),
            background_depth_m=1.0,
            background_id="flat-bg",
        )
    )

    result = pipeline.run(np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(result.objects) == 1
    obj = result.objects[0]
    assert obj.volume_m3 == pytest.approx(36 * 0.0001 * 0.2)
    assert obj.mass_kg == pytest.approx(obj.volume_m3 * 10.0)
    assert obj.mass_method == "density"
    assert obj.geometry["background_id"] == "flat-bg"
    assert result.to_dict()["objects"][0]["geometry"]["volume_m3"] == pytest.approx(obj.volume_m3)


def test_pipeline_normalizes_detection_class_typo_before_mass_stage():
    pipeline = Pipeline()
    pipeline.add_stage("detection", Stage("detection", FakeTypoDetector({})))
    pipeline.add_stage("depth", Stage("depth", FakeDepth({})))
    pipeline.add_stage(
        "mass",
        Stage(
            "mass",
            DensityMassEstimator(
                {
                    "densities": {"plastic": 10.0, "other": 1.0},
                    "class_to_material": {"plastic_bottle": "plastic"},
                }
            ),
        ),
    )
    pipeline.set_geometry_estimator(
        GeometryEstimator(calibration=_calibration(), background_depth_m=1.0)
    )
    stage_updates = []

    result = pipeline.run(
        np.zeros((10, 10, 3), dtype=np.uint8),
        stage_callback=stage_updates.append,
    )

    obj = result.objects[0]
    assert obj.detection.class_name == "plastic_bottle"
    assert obj.mass_kg == pytest.approx(obj.volume_m3 * 10.0)
    detection_completed = [
        update
        for update in stage_updates
        if update.stage_key == "object-detection" and update.status == "completed"
    ][0]
    assert detection_completed.metadata["class_name_corrections"] == {
        "plastic_battle->plastic_bottle": 1
    }


def test_pipeline_load_all_disables_failed_primary_and_uses_fallback():
    pipeline = Pipeline()
    pipeline.add_stage("detection", Stage("detection", FakeDetector({})))
    pipeline.add_stage("depth", Stage("depth", FakeDepth({})))
    pipeline.add_stage(
        "mass",
        Stage(
            "mass",
            FailingModule({}),
            fallback=DensityMassEstimator(
                {
                    "densities": {"plastic": 10.0, "other": 1.0},
                    "class_to_material": {"plastic_bottle": "plastic"},
                }
            ),
        ),
    )
    pipeline.set_geometry_estimator(
        GeometryEstimator(calibration=_calibration(), background_depth_m=1.0)
    )

    pipeline.load_all()
    result = pipeline.run(np.zeros((10, 10, 3), dtype=np.uint8))

    assert result.objects[0].mass_method == "density"
    assert pipeline.stages["mass"].primary_available is False
