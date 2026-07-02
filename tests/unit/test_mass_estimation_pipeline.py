"""Focused tests for the governed mass-estimation pipeline."""

from __future__ import annotations

import json
import warnings
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


def test_preprocess_preserves_object_id_when_used_as_sample_id(tmp_path: Path):
    import pandas as pd

    measurements = tmp_path / "object_only_features.csv"
    pd.DataFrame(
        [
            {
                "object_id": "obj-a",
                "class": "plastic_bottle",
                "material": "plastic",
                "selected_volume_cm3": 50.0,
                "effective_density_g_cm3": 0.04,
                "real_mass_g": 8.0,
            },
            {
                "object_id": "obj-b",
                "class": "metal_can",
                "material": "metal",
                "selected_volume_cm3": 30.0,
                "effective_density_g_cm3": 0.16,
                "real_mass_g": 12.0,
            },
        ]
    ).to_csv(measurements, index=False)
    config = _config(tmp_path)
    config.payload["data"]["measurements"] = str(measurements)
    config.payload["data"]["columns"].update(
        {
            "sample_id": "object_id",
            "object_id": "object_id",
            "class_name": "class",
            "volume": "selected_volume_cm3",
            "volume_unit": "cm3",
            "density": "effective_density_g_cm3",
        }
    )

    result = MassEstimationPipeline(config).run_native("preprocess")
    processed = pd.read_csv(config.processed_dir / "preprocessed_objects.csv")

    assert result["preprocess"]["generated_object_ids"] is False
    assert processed["sample_id"].tolist() == ["obj-a", "obj-b"]
    assert processed["object_id"].tolist() == ["obj-a", "obj-b"]
    assert processed["mass_base_g"].tolist() == pytest.approx([2.0, 4.8])


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


def test_lightgbm_feature_name_warning_is_suppressed():
    class WarningEstimator:
        def predict(self, features):
            warnings.warn(
                "X does not have valid feature names, but LGBMRegressor "
                "was fitted with feature names",
                UserWarning,
            )
            return [1.5] * len(features)

    frame = __import__("pandas").DataFrame({"mass_base_g": [10.0], "feature": [2.0]})
    model = ResidualMassModel(
        name="lightgbm",
        model_type="lightgbm",
        estimator=WarningEstimator(),
        feature_columns=["feature"],
        numeric_columns=["feature"],
        categorical_columns=[],
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        prediction = model.predict_mass(frame)

    assert prediction.tolist() == pytest.approx([11.5])
    assert captured == []


def test_training_mode_single_trains_only_selected_candidate(tmp_path: Path):
    config = _config(tmp_path)
    config.payload["training"]["mode"] = "single"
    config.payload["training"]["single_model"] = "ridge"

    result = MassEstimationPipeline(config).run_native("preprocess,features,split,train")

    assert result["train"]["training_mode"] == "single"
    assert result["train"]["selected_candidate"]["name"] == "ridge"
    assert [candidate["name"] for candidate in result["train"]["candidates"]] == ["ridge"]


def test_sklearn_boosting_candidate_trains_in_single_mode(tmp_path: Path):
    config = _config(tmp_path)
    config.payload["model"]["candidates"].append(
        {
            "name": "gradient_boosting",
            "type": "gradient_boosting",
            "enabled": True,
            "params": {"n_estimators": 20, "learning_rate": 0.05, "max_depth": 2},
        }
    )
    config.payload["training"]["mode"] = "single"
    config.payload["training"]["single_model"] = "gradient_boosting"

    result = MassEstimationPipeline(config).run_native("preprocess,features,split,train")

    assert result["train"]["selected_candidate"]["name"] == "gradient_boosting"
    assert result["train"]["selected_candidate"]["type"] == "gradient_boosting"


def test_plan_reports_missing_optional_booster_package(tmp_path: Path, monkeypatch):
    from edge_ai_mass.mass_estimation import pipeline as mass_pipeline_module

    config = _config(tmp_path)
    config.payload["model"]["candidates"].append(
        {
            "name": "xgboost",
            "type": "xgboost",
            "enabled": True,
            "params": {},
        }
    )
    original_find_spec = mass_pipeline_module.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "xgboost":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(mass_pipeline_module.importlib.util, "find_spec", fake_find_spec)

    plan = MassEstimationPipeline(config).plan("train")

    assert plan["ready"] is False
    assert "missing Python package: xgboost" in plan["blocking_issues"]
