"""Runtime contracts for the deployed residual mass estimator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from edge_ai_mass.mass_estimation.models.residual import ResidualMassModel
from edge_ai_mass.modules.mass.residual_estimator import ResidualMassEstimator


def test_residual_mass_estimator_builds_model_feature_row(tmp_path: Path):
    schema = json.loads(
        Path("mass_model/data/processed/mass_estimation/feature_schema.json").read_text(
            encoding="utf-8"
        )
    )
    model = ResidualMassModel(
        name="physics-only-runtime-test",
        model_type="physics_only",
        estimator=None,
        feature_columns=list(schema["feature_columns"]),
        numeric_columns=list(schema["numeric_columns"]),
        categorical_columns=list(schema["categorical_columns"]),
        mass_base_column=str(schema["mass_base_column"]),
    )
    model_path = model.save(tmp_path / "best_model.joblib")

    estimator = ResidualMassEstimator(
        {
            "model_path": str(model_path),
            "class_to_material": {"plastic_bottle": "plastic"},
            "density_priors_kg_m3": {"plastic": 40.0, "other": 100.0},
            "default_pixel_to_m": 0.01,
            "default_thickness_m": 0.02,
        }
    )
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1

    result = estimator.predict(
        np.zeros((10, 10, 3), dtype=np.uint8),
        features={
            "bbox": np.array([2, 2, 8, 8], dtype=np.float32),
            "mask": mask,
            "class_id": 4,
            "class_name": "plastic_bottle",
            "confidence": 0.91,
            "depth_stats": {
                "mean": 0.8,
                "median": 0.8,
                "std": 0.01,
                "min": 0.78,
                "max": 0.82,
                "range": 0.04,
                "p10": 0.79,
                "p90": 0.81,
                "iqr": 0.02,
                "valid_ratio": 1.0,
            },
            "geometry": {
                "width_m": 0.06,
                "height_m": 0.06,
                "mean_object_depth_m": 0.8,
                "mean_background_depth_m": 1.0,
                "mean_height_m": 0.02,
                "volume_m3": 72e-6,
                "method": "mask_depth_background",
                "calibration_id": "cal-1",
                "depth_scale_id": "depth-1",
                "background_id": "bg-1",
            },
            "volume_m3": 72e-6,
        },
    )

    row = result.data["mass_features"]

    assert result.metadata["method"] == "residual_mass_model"
    assert set(schema["feature_columns"]).issubset(row)
    assert row["material"] == "plastic"
    assert row["selected_volume_cm3"] == pytest.approx(72.0)
    assert row["mass_base_g"] == pytest.approx(2.88)
    assert result.data["predicted_correction_g"] == pytest.approx(0.0)
    assert result.data["mass_kg"] == pytest.approx(0.00288)
