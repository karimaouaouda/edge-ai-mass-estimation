"""Focused tests for the governed mass-estimation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edge_ai_mass.mass_estimation import MassEstimationConfig, MassEstimationPipeline
from edge_ai_mass.mass_estimation.evaluation.metrics import regression_metrics
from edge_ai_mass.mass_estimation.features import validate_no_target_leakage
from edge_ai_mass.mass_estimation.models import ResidualMassModel


def _write_measurements(path: Path) -> None:
    import pandas as pd

    rows = []
    classes = ["plastic_bottle", "metal_can", "paper_cardboard"]
    densities = {"plastic_bottle": 40.0, "metal_can": 160.0, "paper_cardboard": 60.0}
    for object_index in range(6):
        class_name = classes[object_index % len(classes)]
        density = densities[class_name]
        for repeat in range(2):
            volume = 0.0004 + object_index * 0.00008 + repeat * 0.00001
            mass_base = volume * density * 1000.0
            residual = 5.0 + object_index * 1.5
            rows.append(
                {
                    "sample_id": f"s{object_index}_{repeat}",
                    "object_id": f"obj{object_index}",
                    "class_name": class_name,
                    "estimated_volume_m3": volume,
                    "effective_density_kg_m3": density,
                    "bbox_width_px": 20 + object_index,
                    "bbox_height_px": 10 + repeat,
                    "mask_area_px": 200 + object_index * 10,
                    "depth_valid_ratio": 0.90,
                    "real_mass_g": mass_base + residual,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _config(tmp_path: Path) -> MassEstimationConfig:
    measurements = tmp_path / "mass_measurements.csv"
    _write_measurements(measurements)
    payload = {
        "project": {
            "name": "test",
            "root": str(tmp_path),
            "pipeline": "mass_estimation",
            "model_stage": "mass",
        },
        "orchestration": {"zenml": {"enabled": False}},
        "data": {
            "measurements": str(measurements),
            "output_dir": "processed_mass",
            "columns": {
                "sample_id": "sample_id",
                "object_id": "object_id",
                "class_name": "class_name",
                "real_mass_g": "real_mass_g",
                "volume": "estimated_volume_m3",
                "volume_unit": "m3",
                "density": "effective_density_kg_m3",
            },
            "material": {
                "class_to_material": {
                    "plastic_bottle": "plastic",
                    "metal_can": "metal",
                    "paper_cardboard": "cardboard",
                }
            },
            "optional_inputs": {},
        },
        "features": {
            "output_format": "csv",
            "require_mass_base": True,
            "mass_base_column": "mass_base_g",
            "categorical_columns": ["class_name", "material"],
            "numeric_columns": [
                "estimated_volume_m3",
                "effective_density_kg_m3",
                "mass_base_g",
                "bbox_width_px",
                "bbox_height_px",
                "mask_area_px",
                "depth_valid_ratio",
            ],
            "auto_include_numeric": True,
            "strict_columns": True,
            "exclude_columns": ["real_mass_g", "residual_g", "correction_g"],
        },
        "split": {
            "train": 0.5,
            "val": 0.25,
            "test": 0.25,
            "seed": 3,
            "group_column": "object_id",
            "unsafe_random_fallback": False,
        },
        "model": {
            "target": "residual_g",
            "candidates": [
                {"name": "physics", "type": "physics_only", "enabled": True, "params": {}},
                {
                    "name": "ridge",
                    "type": "ridge",
                    "enabled": True,
                    "params": {"alpha": 0.1},
                },
            ],
        },
        "training": {
            "run_name": "unit-mass",
            "artifacts_dir": "artifacts/mass_estimation",
            "seed": 7,
            "selection": {"split": "val", "metric": "hybrid.mae_g", "mode": "minimize"},
        },
        "evaluation": {"splits": ["val", "test"], "small_mass_epsilon_g": 1.0},
        "tracking": {
            "enabled": False,
            "uri": "mlruns",
            "registry": {"enabled": False},
        },
    }
    return MassEstimationConfig(payload, tmp_path / "mass_config.yaml")


def test_feature_schema_rejects_target_leakage():
    validate_no_target_leakage(["mass_base_g", "estimated_volume_m3"])

    with pytest.raises(ValueError, match="Target leakage"):
        validate_no_target_leakage(["mass_base_g", "real_mass_g", "correction_g"])


def test_regression_metrics_are_reported_in_grams():
    metrics = regression_metrics([10.0, 20.0], [12.0, 18.0])

    assert metrics["mae_g"] == pytest.approx(2.0)
    assert metrics["rmse_g"] == pytest.approx(2.0)
    assert metrics["median_absolute_error_g"] == pytest.approx(2.0)
    assert metrics["r2"] == pytest.approx(0.84)


def test_mass_pipeline_runs_end_to_end_on_synthetic_dataset(tmp_path: Path):
    config = _config(tmp_path)
    pipeline = MassEstimationPipeline(config)

    result = pipeline.run_native("preprocess,features,split,train,evaluate")

    assert (config.processed_dir / "preprocessed_objects.csv").is_file()
    assert (config.processed_dir / "feature_schema.json").is_file()
    assert (config.processed_dir / "features_train.csv").is_file()
    assert Path(result["train"]["best_model"]).is_file()
    assert (config.artifacts_dir / "evaluation" / "metrics.json").is_file()
    schema = json.loads((config.processed_dir / "feature_schema.json").read_text())
    assert "real_mass_g" not in schema["feature_columns"]
    assert "correction_g" not in schema["feature_columns"]
    assert "mass_base_g" in schema["feature_columns"]


def test_split_keeps_object_groups_disjoint(tmp_path: Path):
    config = _config(tmp_path)
    pipeline = MassEstimationPipeline(config)
    result = pipeline.run_native("preprocess,features,split")

    assert result["split"]["group_overlap"] == {
        "train_val": [],
        "train_test": [],
        "val_test": [],
    }
    assert sum(result["split"]["groups"].values()) == 6


def test_saved_residual_model_loads_and_predicts(tmp_path: Path):
    config = _config(tmp_path)
    pipeline = MassEstimationPipeline(config)
    result = pipeline.run_native("preprocess,features,split,train")

    model = ResidualMassModel.load(result["train"]["best_model"])
    frame = __import__("pandas").read_csv(config.processed_dir / "features_val.csv")
    predictions = model.predict_mass(frame)

    assert len(predictions) == len(frame)
    assert predictions.dtype.kind == "f"
