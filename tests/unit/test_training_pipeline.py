"""Focused contracts for the governed training pipeline without GPU work."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from edge_ai_mass.training.config import TrainingConfig, TrainingConfigError, normalize_stages
from edge_ai_mass.training.pipeline import TrainingPipeline
from edge_ai_mass.training.preprocessing import DatasetBuildError, build_yolo_dataset
from edge_ai_mass.training.state import PipelineState
from edge_ai_mass.training.tracking import normalize_uri
from edge_ai_mass.training.visualization import create_dataset_visualizations
from edge_ai_mass.training.yolo import (
    YOLOTrainer,
    _organize_export,
    _path_digest,
    normalize_metrics,
)


def _source(root: Path, name: str, label: str, *, image_label_field: bool = False) -> dict:
    source_dir = root / name
    source_dir.mkdir()
    images = []
    annotations = []
    for index in range(4):
        file_name = f"{index}.jpg"
        Image.new("RGB", (32, 24), color=(index * 30, 20, 40)).save(source_dir / file_name)
        image = {"id": index + 1, "file_name": file_name, "width": 32, "height": 24}
        if image_label_field:
            image["source_class_name"] = label
        images.append(image)
        annotations.append(
            {
                "id": index + 1,
                "image_id": index + 1,
                "category_id": 1,
                "bbox": [2, 3, 12, 10],
                "segmentation": [[2, 3, 14, 3, 14, 13, 2, 13]],
            }
        )
    annotations_path = source_dir / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": label}],
            }
        ),
        encoding="utf-8",
    )
    result = {
        "name": name,
        "annotations": str(annotations_path),
        "images": str(source_dir),
        "resolver": "direct",
        "category_mapping": {label: "mixed_waste"},
    }
    if image_label_field:
        result["category_from_image_field"] = "source_class_name"
    return result


def _realwaste_source(root: Path) -> dict:
    """Model the raw nested tree plus a separately uploaded segmentation JSON."""
    images_root = root / "realwaste-main" / "RealWaste"
    original_class = images_root / "Foreign Folder Label"
    original_class.mkdir(parents=True)
    images = []
    annotations = []
    for index in range(4):
        source_name = f"Foreign Folder Label/{index}.jpg"
        Image.new("RGB", (32, 24), color=(index * 30, 20, 40)).save(images_root / source_name)
        images.append(
            {
                "id": index + 1,
                # The flattened output does not exist in the raw dataset; the
                # source path is resolved after stripping the dataset prefix.
                "file_name": f"flattened-{index}.jpg",
                "source_file_name": f"realwaste-main/RealWaste/{source_name}",
                "width": 32,
                "height": 24,
            }
        )
        annotations.append(
            {
                "id": index + 1,
                "image_id": index + 1,
                "category_id": 1,
                "bbox": [2, 3, 12, 10],
                "segmentation": [[2, 3, 14, 3, 14, 13, 2, 13]],
            }
        )
    annotations_dir = root / "realwaste-annotations"
    annotations_dir.mkdir()
    annotations_path = annotations_dir / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                # This project label—not "Foreign Folder Label"—must control
                # every generated YOLO class id.
                "categories": [{"id": 1, "name": "mixed_waste"}],
            }
        ),
        encoding="utf-8",
    )
    return {
        "name": "realwaste",
        "annotations": str(annotations_path),
        # Intentionally provide the parent mount, matching common Kaggle usage.
        "images": str(root),
        "images_root_candidates": ["realwaste-main/RealWaste", "RealWaste", "."],
        "file_name_fields": ["source_file_name", "file_name"],
        "strip_path_prefixes": ["realwaste-main/RealWaste", "RealWaste"],
        "recursive_basename_fallback": True,
        "resolver": "direct",
        "require_category_names_in_data_classes": True,
    }


def _config(tmp_path: Path) -> TrainingConfig:
    sources = [
        _source(tmp_path, "taco", "raw_a"),
        _source(tmp_path, "aquatrash", "raw_a"),
        _realwaste_source(tmp_path),
    ]
    payload = {
        "project": {
            "name": "test",
            "root": str(tmp_path),
            "pipeline": "yolo",
            "model_stage": "detection",
        },
        "data": {
            "output_dir": "processed",
            "classes": ["mixed_waste"],
            "materialize": "copy",
            "split": {"train": 0.5, "val": 0.25, "test": 0.25, "seed": 7},
            "quality": {"min_images": 1, "min_instances": 1, "require_all_classes": True},
            "sources": sources,
        },
        "model": {"task": "segment", "checkpoint": "any-checkpoint.pt"},
        "training": {"run_name": "unit", "artifacts_dir": "artifacts"},
        "tuning": {"enabled": False},
        "tracking": {"uri": "mlruns", "registry": {"enabled": False}},
    }
    return TrainingConfig(payload, tmp_path / "config.yaml")


def test_preprocessing_merges_three_coco_sources(tmp_path: Path):
    config = _config(tmp_path)
    manifest = build_yolo_dataset(config)

    assert manifest["total_images"] == 12
    assert manifest["total_instances"] == 12
    assert set(manifest["sources"]) == {"taco", "aquatrash", "realwaste"}
    assert manifest["sources"]["realwaste"]["classes"] == {"mixed_waste": 4}
    assert manifest["sources"]["realwaste"]["image_resolution"] == {
        "prefix_stripped:source_file_name": 4
    }
    assert Path(manifest["sources"]["realwaste"]["images_root_resolved"]).name == "RealWaste"
    assert sum(item["images"] for item in manifest["splits"].values()) == 12
    assert (config.dataset_dir / "dataset.yaml").is_file()
    label = next((config.dataset_dir / "labels" / "train").rglob("*.txt"))
    assert label.read_text(encoding="utf-8").startswith("0 ")


def test_realwaste_rejects_legacy_non_project_categories(tmp_path: Path):
    config = _config(tmp_path)
    source = next(item for item in config.payload["data"]["sources"] if item["name"] == "realwaste")
    annotations_path = Path(source["annotations"])
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    payload["categories"] = [{"id": 1, "name": "trash"}]
    annotations_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetBuildError, match="categories must be names from data.classes"):
        build_yolo_dataset(config)


def test_realwaste_failure_prints_bounded_structured_debug(tmp_path: Path, capsys):
    config = _config(tmp_path)
    empty_mount = tmp_path / "empty-realwaste-mount"
    empty_mount.mkdir()
    source = next(item for item in config.payload["data"]["sources"] if item["name"] == "realwaste")
    source["images"] = str(empty_mount)
    config.payload["data"]["debug"] = {
        "enabled": True,
        "sample_limit": 1,
        "progress_every": 4,
        "root_entry_limit": 5,
    }

    with pytest.raises(DatasetBuildError, match="missing image fraction 100.000%"):
        build_yolo_dataset(config)

    output = capsys.readouterr().out
    assert '"event": "image_root.before"' in output
    assert '"event": "image_root.candidate"' in output
    assert '"event": "image.resolve.before"' in output
    assert '"event": "image.resolve.after"' in output
    assert '"event": "source.progress"' in output
    assert '"event": "quality.failed"' in output
    assert output.count('"event": "image.resolve.before"') == 3


def test_plan_and_stage_aliases_do_not_start_training(tmp_path: Path):
    config = _config(tmp_path)
    plan = TrainingPipeline(config).plan("detection", skip_optuna=True)

    assert plan["execution_stages"] == [
        "preprocess",
        "train",
        "evaluate",
        "export",
        "register",
    ]
    assert normalize_stages("preprocess,train") == ["preprocess", "train"]
    assert plan["checkpoint"] == "any-checkpoint.pt"


def test_metric_normalization_exposes_stable_names():
    metrics = {"metrics/mAP50-95(M)": 0.42, "metrics/mAP50(B)": 0.71}
    normalized = normalize_metrics(metrics)

    assert normalized["mask_map50_95"] == 0.42
    assert normalized["box_map50"] == 0.71


def test_sqlite_tracking_uri_is_resolved_inside_project(tmp_path: Path):
    uri = normalize_uri("sqlite:///artifacts/mlflow/mlflow.db", tmp_path)

    assert uri == f"sqlite:///{(tmp_path / 'artifacts/mlflow/mlflow.db').as_posix()}"
    assert (tmp_path / "artifacts" / "mlflow").is_dir()


def test_dataset_visualizations_render_annotated_mosaics(tmp_path: Path):
    config = _config(tmp_path)
    build_yolo_dataset(config)

    report = create_dataset_visualizations(config)

    assert report["enabled"] is True
    assert report["splits"]["train"]["rendered_samples"] > 0
    assert Path(report["splits"]["train"]["mosaic"]).is_file()


def test_export_artifacts_are_moved_beside_format_manifest(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    best = models_dir / "best.pt"
    exported = models_dir / "best.onnx"
    best.write_bytes(b"best-model")
    exported.write_bytes(b"onnx-model")
    destination = tmp_path / "exports" / "onnx"
    destination.mkdir(parents=True)

    organized = _organize_export(exported, destination, best)

    assert organized == (destination / "best.onnx").resolve()
    assert organized.is_file()
    assert not exported.exists()
    assert _path_digest(organized) == _path_digest(destination / "best.onnx")


def test_default_optuna_space_covers_training_loss_and_augmentation_groups():
    config = TrainingConfig.load("configs/training/yolo_segmentation.yaml")
    search_space = config.payload["tuning"]["search_space"]

    assert len(search_space) >= 35
    assert {
        "batch",
        "imgsz",
        "optimizer",
        "lr0",
        "momentum",
        "box",
        "cls",
        "dfl",
        "mask_ratio",
        "mosaic",
        "copy_paste",
        "multi_scale",
    } <= set(search_space)


def test_training_curves_and_annotated_batches_are_organized(tmp_path: Path):
    config = _config(tmp_path)
    trainer = YOLOTrainer(config, PipelineState(config.artifacts_dir / "state.json"))
    run_dir = tmp_path / "ultralytics-run"
    run_dir.mkdir()
    for name in ("results.png", "results.csv", "PR_curve.png", "train_batch0.jpg"):
        (run_dir / name).write_bytes(name.encode())

    artifacts = trainer._collect_training_artifacts(run_dir)

    assert len(artifacts["curves"]) == 3
    assert len(artifacts["annotated_batches"]) == 1
    assert all(Path(path).is_file() for path in artifacts["curves"])


def test_tuning_guardrail_rejects_configured_params_without_distributions(tmp_path: Path):
    payload = _config(tmp_path).payload
    payload["training"]["hyperparameters"] = {"lr0": 0.001}
    payload["tuning"] = {
        "enabled": True,
        "enforce_complete_space": True,
        "search_space": {},
    }

    with pytest.raises(TrainingConfigError, match="lr0"):
        TrainingConfig(payload, tmp_path / "config.yaml")
