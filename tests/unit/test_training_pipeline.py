"""Focused contracts for the governed training pipeline without GPU work."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from edge_ai_mass.training.checkpoints import (
    CheckpointStore,
    ModelSource,
    resolve_model_source,
    resume_target_epochs,
)
from edge_ai_mass.training.config import TrainingConfig, TrainingConfigError, normalize_stages
from edge_ai_mass.training.devices import (
    is_directml_device,
    requires_torch_directml,
    resolve_export_device,
)
from edge_ai_mass.training.pipeline import TrainingPipeline
from edge_ai_mass.training.preprocessing import DatasetBuildError, build_yolo_dataset
from edge_ai_mass.training.publication import (
    TrainingOutputPublisher,
    collect_reusable_outputs,
    restore_training_outputs,
)
from edge_ai_mass.training.state import PipelineState
from edge_ai_mass.training.taxonomy import resolve_category
from edge_ai_mass.training.tracking import normalize_uri
from edge_ai_mass.training.visualization import create_dataset_visualizations
from edge_ai_mass.training.yolo import (
    YOLOTrainer,
    _accelerator_out_of_memory_kind,
    _is_cuda_runtime_import_error,
    _is_cuda_out_of_memory,
    _next_smaller_evaluation_batch,
    _organize_export,
    _path_digest,
    _set_evaluation_mode,
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


def _trashnet_source(root: Path) -> dict:
    """Model TrashNet raw images plus the separately uploaded COCO masks JSON."""
    images_root = root / "trashnet" / "dataset-resized" / "trash"
    images_root.mkdir(parents=True)
    images = []
    annotations = []
    genuine_mask = [2, 3, 14, 4, 12, 13, 7, 11, 3, 8]
    for index in range(4):
        file_name = f"trash-{index}.jpg"
        Image.new("RGB", (32, 24), color=(index * 20, 30, 50)).save(
            images_root / file_name
        )
        images.append(
            {
                "id": index + 1,
                # Kaggle can mount the raw dataset above dataset-resized; the
                # configured source strips that prefix before resolving.
                "file_name": f"dataset-resized/trash/{file_name}",
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
                "segmentation": [genuine_mask],
            }
        )
    annotations_dir = root / "trashnet-segmentations" / "annotations"
    annotations_dir.mkdir(parents=True)
    annotations_path = annotations_dir / "instances_default.json"
    annotations_path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "trash"}],
            }
        ),
        encoding="utf-8",
    )
    return {
        "name": "trashnet",
        "annotations": str(annotations_path),
        # Intentionally provide the parent mount, matching Kaggle usage.
        "images": str(root / "trashnet"),
        "images_root_candidates": ["dataset-resized", "."],
        "file_name_fields": ["file_name", "source_file_name"],
        "strip_path_prefixes": ["dataset-resized"],
        "recursive_basename_fallback": True,
        "resolver": "trashnet",
        "unknown_category": "error",
        "allow_bbox_fallback": False,
        "box_polygon_policy": "drop_annotation",
        "box_polygon_match": "full_image",
        "box_polygon_tolerance_pixels": 2.0,
        "keep_empty_images": False,
    }


def _config(tmp_path: Path) -> TrainingConfig:
    sources = [
        _source(tmp_path, "taco", "raw_a"),
        _source(tmp_path, "aquatrash", "raw_a"),
        _trashnet_source(tmp_path),
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


def test_preprocessing_merges_coco_sources_including_trashnet(tmp_path: Path):
    config = _config(tmp_path)
    manifest = build_yolo_dataset(config)

    assert manifest["total_images"] == 16
    assert manifest["total_instances"] == 16
    assert set(manifest["sources"]) == {"taco", "aquatrash", "trashnet", "realwaste"}
    assert manifest["sources"]["trashnet"]["classes"] == {"mixed_waste": 4}
    assert manifest["sources"]["trashnet"]["image_resolution"] == {
        "prefix_stripped:file_name": 4
    }
    assert Path(manifest["sources"]["trashnet"]["images_root_resolved"]).name == (
        "dataset-resized"
    )
    assert manifest["sources"]["realwaste"]["classes"] == {"mixed_waste": 4}
    assert manifest["sources"]["realwaste"]["image_resolution"] == {
        "prefix_stripped:source_file_name": 4
    }
    assert Path(manifest["sources"]["realwaste"]["images_root_resolved"]).name == "RealWaste"
    assert sum(item["images"] for item in manifest["splits"].values()) == 16
    assert (config.dataset_dir / "dataset.yaml").is_file()
    assert (config.dataset_dir / manifest["clean_coco"]).is_file()
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


def test_realwaste_removes_box_polygon_annotations_and_their_images(tmp_path: Path):
    config = _config(tmp_path)
    source = next(item for item in config.payload["data"]["sources"] if item["name"] == "realwaste")
    source["drop_images_with_box_polygons"] = True
    source["box_polygon_tolerance_pixels"] = 0.5
    annotations_path = Path(source["annotations"])
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))

    # Keep three genuine, non-rectangular masks. The fourth annotation remains
    # the box-shaped polygon created by the fixture and must remove its image.
    genuine_mask = [2, 3, 14, 4, 12, 13, 7, 11, 3, 8]
    for annotation in payload["annotations"][:3]:
        annotation["segmentation"] = [genuine_mask]
    annotations_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = build_yolo_dataset(config)
    report = manifest["sources"]["realwaste"]

    assert manifest["total_images"] == 15
    assert manifest["total_instances"] == 15
    assert report["images"] == 3
    assert report["instances"] == 3
    assert report["box_polygon_annotations"] == 1
    assert report["images_removed_box_polygons"] == 1
    assert report["box_polygon_image_examples"][0]["image_id"] == 4

    materialized_realwaste = list(
        (config.dataset_dir / "images").glob("*/realwaste/*")
    )
    assert len(materialized_realwaste) == 3


def test_trashnet_removes_full_image_box_masks_and_empty_images(tmp_path: Path):
    config = _config(tmp_path)
    source = next(item for item in config.payload["data"]["sources"] if item["name"] == "trashnet")
    annotations_path = Path(source["annotations"])
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    full_image_box = [0, 0, 32, 0, 32, 24, 0, 24]
    genuine_mask = [2, 3, 14, 4, 12, 13, 7, 11, 3, 8]
    payload["annotations"] = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 32, 24],
            "segmentation": [full_image_box],
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [2, 3, 12, 10],
            "segmentation": [genuine_mask],
        },
        {
            "id": 3,
            "image_id": 2,
            "category_id": 1,
            "bbox": [0, 0, 32, 24],
            "segmentation": [full_image_box],
        },
        # image 3 intentionally has no segmentation and must be removed.
        {
            "id": 4,
            "image_id": 4,
            "category_id": 1,
            "bbox": [2, 3, 12, 10],
            "segmentation": [genuine_mask],
        },
    ]
    annotations_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = build_yolo_dataset(config)
    report = manifest["sources"]["trashnet"]

    assert manifest["total_images"] == 14
    assert manifest["total_instances"] == 14
    assert report["images"] == 2
    assert report["instances"] == 2
    assert report["box_polygon_policy"] == "drop_annotation"
    assert report["box_polygon_match"] == "full_image"
    assert report["box_polygon_annotations"] == 2
    assert report["images_removed_box_polygons"] == 0
    assert report["empty_images"] == 2
    assert report["empty_images_removed"] == 2
    assert report["classes"] == {"mixed_waste": 2}
    materialized_trashnet = list((config.dataset_dir / "images").glob("*/trashnet/*"))
    assert len(materialized_trashnet) == 2
    clean_coco = json.loads(
        (config.dataset_dir / manifest["clean_coco"]).read_text(encoding="utf-8")
    )
    clean_trashnet_images = [
        image for image in clean_coco["images"] if image["source"] == "trashnet"
    ]
    assert {Path(image["source_name"]).name for image in clean_trashnet_images} == {
        "trash-0.jpg",
        "trash-3.jpg",
    }
    clean_trashnet_image_ids = {image["id"] for image in clean_trashnet_images}
    assert sum(
        annotation["image_id"] in clean_trashnet_image_ids
        for annotation in clean_coco["annotations"]
    ) == 2


def test_trashnet_taxonomy_resolver_maps_source_labels_to_project_classes():
    assert resolve_category({"name": "cardboard"}, resolver="trashnet") == "paper_cardboard"
    assert resolve_category({"name": "paper"}, resolver="trashnet") == "paper_cardboard"
    assert resolve_category({"name": "metal"}, resolver="trashnet") == "metal_can"
    assert resolve_category({"name": "plastic"}, resolver="trashnet") == "rigid_plastic"
    assert resolve_category({"name": "trash"}, resolver="trashnet") == "mixed_waste"


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
    assert output.count('"event": "image.resolve.before"') == 4


def test_plan_and_stage_aliases_do_not_start_training(tmp_path: Path):
    config = _config(tmp_path)
    plan = TrainingPipeline(config).plan("detection", skip_optuna=True)

    assert plan["execution_stages"] == [
        "preprocess",
        "train",
        "evaluate",
        "export",
        "register",
        "publish",
    ]
    assert normalize_stages("preprocess,train") == ["preprocess", "train"]
    assert plan["checkpoint"] == "any-checkpoint.pt"


def test_plan_checks_tensorrt_for_evaluation_engine_pre_export(
    tmp_path: Path,
    monkeypatch,
):
    import edge_ai_mass.training.pipeline as pipeline_module

    config = _config(tmp_path)
    config.payload["evaluation"] = {
        "pre_export": {"enabled": True, "format": "engine"}
    }
    original_find_spec = pipeline_module.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "tensorrt":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(pipeline_module.importlib.util, "find_spec", fake_find_spec)

    plan = TrainingPipeline(config).plan("evaluate")

    assert "missing Python package: tensorrt" in plan["blocking_issues"]


def test_plan_checks_onnxruntime_for_evaluation_onnx_pre_export(
    tmp_path: Path,
    monkeypatch,
):
    import edge_ai_mass.training.pipeline as pipeline_module

    config = _config(tmp_path)
    config.payload["evaluation"] = {
        "pre_export": {"enabled": True, "format": "onnx"}
    }
    original_find_spec = pipeline_module.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "onnxruntime":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(pipeline_module.importlib.util, "find_spec", fake_find_spec)

    plan = TrainingPipeline(config).plan("evaluate")

    assert "missing Python package: onnxruntime" in plan["blocking_issues"]
    assert "missing Python package: tensorrt" not in plan["blocking_issues"]


def test_plan_reports_broken_onnxruntime_cuda_runtime_import(
    tmp_path: Path,
    monkeypatch,
):
    import edge_ai_mass.training.pipeline as pipeline_module

    config = _config(tmp_path)
    config.payload["evaluation"] = {
        "pre_export": {"enabled": True, "format": "onnx"}
    }

    def fake_package_status(name: str):
        if name == "onnxruntime":
            return {
                "available": False,
                "error": (
                    "ImportError: libcudart.so.13: cannot open shared object file: "
                    "No such file or directory"
                ),
            }
        return {"available": True, "error": None}

    monkeypatch.setattr(pipeline_module, "_package_status", fake_package_status)

    plan = TrainingPipeline(config).plan("evaluate")

    assert plan["packages"]["onnxruntime"] is False
    assert "libcudart.so.13" in plan["package_errors"]["onnxruntime"]
    assert any(
        issue.startswith("broken Python package: onnxruntime:")
        for issue in plan["blocking_issues"]
    )


def test_plan_requires_torch_directml_for_directml_training(
    tmp_path: Path,
    monkeypatch,
):
    import edge_ai_mass.training.pipeline as pipeline_module

    config = _config(tmp_path)
    config.payload["training"]["device"] = "directml"

    def fake_package_status(name: str):
        if name == "torch-directml":
            return {"available": False, "error": None}
        return {"available": True, "error": None}

    monkeypatch.setattr(pipeline_module, "_package_status", fake_package_status)

    plan = TrainingPipeline(config).plan("train")

    assert plan["packages"]["torch-directml"] is False
    assert "missing Python package: torch-directml" in plan["blocking_issues"]


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


def test_evaluation_precision_casts_model_parameters_to_fp16(tmp_path: Path):
    import torch

    config = _config(tmp_path)
    trainer = YOLOTrainer(config, PipelineState(config.artifacts_dir / "state.json"))
    module = torch.nn.Linear(4, 2)
    model = SimpleNamespace(model=module)
    precision = trainer._evaluation_precision_config(
        {"precision": "fp16", "cast_model_to_half": True}
    )

    report = trainer._prepare_evaluation_precision(model, precision=precision)

    assert report["precision"] == "fp16"
    assert report["half"] is True
    assert report["cast_status"] == "converted_to_fp16"
    assert report["parameter_footprint_after"]["bytes"] == (
        report["parameter_footprint_before"]["bytes"] // 2
    )
    assert {parameter.dtype for parameter in module.parameters()} == {torch.float16}


def test_evaluation_mode_skips_non_torch_engine_backend():
    model = SimpleNamespace(model="best.engine")

    report = _set_evaluation_mode(model)

    assert report == {
        "status": "skipped_non_torch_backend",
        "backend_type": "str",
    }


def test_cuda_oom_detection_handles_wrapped_runtime_errors():
    root = RuntimeError("CUDA out of memory. Tried to allocate 256.00 MiB.")
    wrapped = RuntimeError(f"Failed to run validation: {root}")
    wrapped.__cause__ = root

    assert _is_cuda_out_of_memory(wrapped) is True
    assert _is_cuda_out_of_memory(RuntimeError("validation metric is missing")) is False


def test_cuda_runtime_import_error_detection_handles_onnxruntime_gpu_mismatch():
    error = ImportError(
        "libcudart.so.13: cannot open shared object file: No such file or directory"
    )
    wrapped = RuntimeError("Failed to load ONNX backend")
    wrapped.__cause__ = error

    assert _is_cuda_runtime_import_error(wrapped) is True
    assert _is_cuda_runtime_import_error(ImportError("onnxruntime missing")) is False


def test_directml_device_aliases_and_export_fallback():
    assert is_directml_device("directml") is True
    assert is_directml_device("dml:1") is True
    assert is_directml_device("privateuseone:0") is True
    assert is_directml_device("cpu") is False
    assert requires_torch_directml("cpu", "directml") is True
    assert resolve_export_device("directml", purpose="unit test") == "cpu"


def test_directml_oom_detection_handles_wrapped_runtime_errors():
    root = RuntimeError("DirectML device ran out of memory on D3D12.")
    wrapped = RuntimeError("Failed to run validation")
    wrapped.__cause__ = root

    assert _accelerator_out_of_memory_kind(wrapped) == "directml"
    assert _accelerator_out_of_memory_kind(RuntimeError("metric is missing")) is None


def test_cuda_oom_retry_halves_batch_until_one():
    assert _next_smaller_evaluation_batch(16) == 8
    assert _next_smaller_evaluation_batch(3) == 2
    assert _next_smaller_evaluation_batch(2) == 1
    assert _next_smaller_evaluation_batch(1) == 1


def test_gpu_evaluation_retry_updates_batch_plots_and_pre_export_options(
    tmp_path: Path,
):
    config = _config(tmp_path)
    trainer = YOLOTrainer(config, PipelineState(config.artifacts_dir / "state.json"))

    retry = trainer._evaluation_for_device(
        {
            "precision": "fp16",
            "batch": 16,
            "plots": True,
            "device": 0,
            "pre_export": {
                "enabled": True,
                "format": "onnx",
                "use_for_evaluation": True,
                "options": {"half": True, "batch": 16, "device": 0},
            },
        },
        device=0,
        batch=8,
        plots=False,
    )

    assert retry["device"] == 0
    assert retry["batch"] == 8
    assert retry["plots"] is False
    assert retry["precision"] == "fp16"
    assert retry["pre_export"]["enabled"] is True
    assert retry["pre_export"]["options"]["device"] == 0
    assert retry["pre_export"]["options"]["batch"] == 8
    assert retry["pre_export"]["options"]["half"] is True


def test_cpu_evaluation_fallback_disables_cuda_precision_and_engine_export(
    tmp_path: Path,
):
    config = _config(tmp_path)
    trainer = YOLOTrainer(config, PipelineState(config.artifacts_dir / "state.json"))

    fallback = trainer._evaluation_for_device(
        {
            "precision": "fp16",
            "half": True,
            "cast_model_to_half": True,
            "device": 0,
            "pre_export": {
                "enabled": True,
                "format": "engine",
                "use_for_evaluation": True,
                "options": {"half": True, "device": 0},
            },
        },
        device="cpu",
        cuda_oom_fallback=True,
    )

    assert fallback["device"] == "cpu"
    assert fallback["precision"] == "fp32"
    assert fallback["half"] is False
    assert fallback["cast_model_to_half"] is False
    assert fallback["pre_export"]["enabled"] is False
    assert fallback["pre_export"]["use_for_evaluation"] is False
    assert fallback["pre_export"]["options"]["device"] == "cpu"
    assert fallback["pre_export"]["options"]["half"] is False
    assert (
        fallback["pre_export"]["disabled_reason"]
        == "cpu_fallback_skips_tensorrt_engine_pre_export"
    )


def test_directml_evaluation_disables_pre_export_and_fp16_casting(
    tmp_path: Path,
):
    config = _config(tmp_path)
    trainer = YOLOTrainer(config, PipelineState(config.artifacts_dir / "state.json"))

    evaluation = trainer._evaluation_for_device(
        {
            "precision": "fp16",
            "half": True,
            "cast_model_to_half": True,
            "batch": 4,
            "device": 0,
            "pre_export": {
                "enabled": True,
                "format": "onnx",
                "use_for_evaluation": True,
                "options": {"half": True, "device": 0, "batch": 4},
            },
        },
        device="privateuseone:0",
        batch=2,
    )

    assert evaluation["device"] == "privateuseone:0"
    assert evaluation["precision"] == "fp32"
    assert evaluation["half"] is False
    assert evaluation["cast_model_to_half"] is False
    assert evaluation["pre_export"]["enabled"] is False
    assert evaluation["pre_export"]["use_for_evaluation"] is False
    assert evaluation["pre_export"]["disabled_reason"] == (
        "directml_evaluation_uses_pytorch_backend"
    )
    assert evaluation["pre_export"]["options"]["device"] == "cpu"
    assert evaluation["pre_export"]["options"]["half"] is False
    assert evaluation["pre_export"]["options"]["batch"] == 2


def test_directml_training_device_resolves_and_disables_amp(
    tmp_path: Path,
    monkeypatch,
):
    import edge_ai_mass.training.devices as devices_module

    config = _config(tmp_path)
    config.payload["training"].update({"device": "directml:1", "amp": True})
    trainer = YOLOTrainer(config, PipelineState(config.artifacts_dir / "state.json"))

    class FakeTorchDirectML:
        @staticmethod
        def device(index=0):
            return f"privateuseone:{index}"

    def fake_import_module(name: str):
        assert name == "torch_directml"
        return FakeTorchDirectML

    monkeypatch.setattr(
        devices_module.importlib,
        "import_module",
        fake_import_module,
    )

    args = trainer._train_args({}, tuning=False, run_name="directml-unit")

    assert args["device"] == "privateuseone:1"
    assert args["amp"] is False


def test_evaluation_validation_args_pass_half_precision_without_hidden_device(
    tmp_path: Path,
):
    config = _config(tmp_path)
    config.payload["evaluation"] = {
        "precision": "fp16",
        "batch": 2,
        "device": 0,
        "workers": 0,
    }
    trainer = YOLOTrainer(config, PipelineState(config.artifacts_dir / "state.json"))
    precision = trainer._evaluation_precision_config(config.payload["evaluation"])

    args = trainer._evaluation_validation_args(
        config.payload["evaluation"],
        split="val",
        precision=precision,
        output_root=config.artifacts_dir / "evaluation",
    )

    assert args["half"] is True
    assert args["batch"] == 2
    assert args["device"] == 0
    assert args["workers"] == 0


def test_evaluation_can_pre_export_fp16_engine_for_validation(tmp_path: Path):
    config = _config(tmp_path)
    trainer = YOLOTrainer(config, PipelineState(config.artifacts_dir / "state.json"))
    best = tmp_path / "best.pt"
    best.write_bytes(b"best")

    class FakeYOLO:
        calls: list[dict] = []

        def __init__(self, weights: str, *, task: str):
            self.weights = weights
            self.task = task

        def export(self, *, format: str, **options):
            self.calls.append({"format": format, "options": options})
            exported = tmp_path / f"best.{format}"
            exported.write_bytes(b"engine")
            return str(exported)

    tracker = SimpleNamespace(
        mlflow=SimpleNamespace(log_artifacts=lambda *args, **kwargs: None)
    )
    report = trainer._prepare_evaluation_pre_export(
        FakeYOLO,
        source_best_weights=best,
        evaluation={
            "imgsz": 320,
            "batch": 1,
            "device": 0,
            "pre_export": {
                "enabled": True,
                "format": "engine",
                "use_for_evaluation": True,
                "options": {"half": True, "int8": False},
            },
        },
        tracker=tracker,
    )

    assert report["status"] == "complete"
    assert report["quantization_bits"] == 16
    assert report["evaluation_weights"].endswith(".engine")
    assert Path(report["evaluation_weights"]).is_file()
    assert FakeYOLO.calls[0]["format"] == "engine"
    assert FakeYOLO.calls[0]["options"]["half"] is True
    assert FakeYOLO.calls[0]["options"]["int8"] is False


def test_evaluation_can_pre_export_fp16_onnx_for_validation(tmp_path: Path):
    config = _config(tmp_path)
    trainer = YOLOTrainer(config, PipelineState(config.artifacts_dir / "state.json"))
    best = tmp_path / "best.pt"
    best.write_bytes(b"best")

    class FakeYOLO:
        calls: list[dict] = []

        def __init__(self, weights: str, *, task: str):
            self.weights = weights
            self.task = task

        def export(self, *, format: str, **options):
            self.calls.append({"format": format, "options": options})
            exported = tmp_path / f"best.{format}"
            exported.write_bytes(b"onnx")
            return str(exported)

    tracker = SimpleNamespace(
        mlflow=SimpleNamespace(log_artifacts=lambda *args, **kwargs: None)
    )
    report = trainer._prepare_evaluation_pre_export(
        FakeYOLO,
        source_best_weights=best,
        evaluation={
            "precision": "fp16",
            "imgsz": 320,
            "batch": 1,
            "device": 0,
            "pre_export": {
                "enabled": True,
                "format": "onnx",
                "use_for_evaluation": True,
                "options": {"half": True, "dynamic": False},
            },
        },
        tracker=tracker,
    )

    assert report["status"] == "complete"
    assert report["format"] == "onnx"
    assert report["quantization_bits"] == 16
    assert report["evaluation_weights"].endswith(".onnx")
    assert Path(report["evaluation_weights"]).is_file()
    assert FakeYOLO.calls[0]["format"] == "onnx"
    assert FakeYOLO.calls[0]["options"]["half"] is True
    assert "int8" not in FakeYOLO.calls[0]["options"]
    assert "workspace" not in FakeYOLO.calls[0]["options"]


def test_evaluation_pre_export_cuda_runtime_error_falls_back_to_pt_weights(
    tmp_path: Path,
):
    config = _config(tmp_path)
    trainer = YOLOTrainer(config, PipelineState(config.artifacts_dir / "state.json"))
    best = tmp_path / "best.pt"
    best.write_bytes(b"best")

    class BrokenOnnxYOLO:
        def __init__(self, weights: str, *, task: str):
            self.weights = weights
            self.task = task

        def export(self, *, format: str, **options):
            raise ImportError(
                "libcudart.so.13: cannot open shared object file: "
                "No such file or directory"
            )

    tracker = SimpleNamespace(
        mlflow=SimpleNamespace(log_artifacts=lambda *args, **kwargs: None)
    )
    report = trainer._prepare_evaluation_pre_export(
        BrokenOnnxYOLO,
        source_best_weights=best,
        evaluation={
            "precision": "fp16",
            "batch": 1,
            "device": 0,
            "pre_export": {
                "enabled": True,
                "format": "onnx",
                "use_for_evaluation": True,
                "continue_on_error": False,
                "options": {"half": True},
            },
        },
        tracker=tracker,
    )

    assert report["status"] == "failed"
    assert report["runtime_error_kind"] == "cuda_runtime_import_error"
    assert report["use_for_evaluation"] is False
    assert report["evaluation_weights"] is None


def test_evaluation_precision_must_be_fp32_or_fp16(tmp_path: Path):
    payload = _config(tmp_path).payload
    payload["evaluation"] = {"precision": "bf16"}

    with pytest.raises(TrainingConfigError, match="evaluation.precision"):
        TrainingConfig(payload, tmp_path / "config.yaml")


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


def test_periodic_checkpoint_stores_weights_metrics_curves_and_latest(tmp_path: Path):
    config = _config(tmp_path)
    config.payload["training"]["checkpointing"] = {
        "enabled": True,
        "interval_epochs": 5,
        "save_final": True,
        "keep_last": 2,
        "resume": {"mode": "auto", "checkpoint": None},
    }
    state = PipelineState(config.artifacts_dir / "state.json")
    run_dir = tmp_path / "ultralytics-run"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    last = weights_dir / "last.pt"
    best = weights_dir / "best.pt"
    last.write_bytes(b"resumable-optimizer-state")
    best.write_bytes(b"best-model-state")
    results_csv = run_dir / "results.csv"
    results_csv.write_text(
        "epoch,train/box_loss,metrics/mAP50-95(M)\n"
        "0,1.5,0.10\n"
        "4,0.8,0.42\n",
        encoding="utf-8",
    )

    trainer = SimpleNamespace(
        epoch=4,
        epochs=20,
        stop=False,
        last=last,
        best=best,
        csv=results_csv,
        save_dir=run_dir,
        metrics={"metrics/mAP50-95(M)": 0.42},
        fitness=0.42,
        best_fitness=0.42,
        lr={"lr/pg0": 0.001},
        tloss=[0.8],
        label_loss_items=lambda values: {"train/box_loss": values[0]},
    )
    store = CheckpointStore(
        config,
        state,
        dataset_fingerprint="dataset-v1",
        model_source=ModelSource("yolo26n-seg.pt", "base_model", False),
    )

    store.callback(trainer)

    checkpoint_dir = config.artifacts_dir / "checkpoints" / "epoch_000005"
    assert (checkpoint_dir / "weights.pt").read_bytes() == b"resumable-optimizer-state"
    assert (checkpoint_dir / "best.pt").read_bytes() == b"best-model-state"
    assert (checkpoint_dir / "results.csv").is_file()
    assert (checkpoint_dir / "metrics_history.json").is_file()
    assert (checkpoint_dir / "metrics_curves.png").is_file()
    latest = json.loads(
        (config.artifacts_dir / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    assert latest["completed_epochs"] == 5
    assert latest["weights"] == "epoch_000005/weights.pt"
    assert Path(state.data["latest_checkpoint"]) == (checkpoint_dir / "weights.pt").resolve()

    resolved = resolve_model_source(config, artifacts_dir=config.artifacts_dir)
    assert resolved.resume is True
    assert resolved.kind == "automatic_checkpoint"
    assert resolved.completed_epochs == 5
    assert Path(resolved.path) == (checkpoint_dir / "weights.pt").resolve()


def test_selected_checkpoint_resume_prunes_newer_checkpoints_and_rewrites_latest(
    tmp_path: Path,
):
    config = _config(tmp_path)
    root = config.artifacts_dir / "checkpoints"

    def write_checkpoint(epoch: int) -> None:
        checkpoint_dir = root / f"epoch_{epoch:06d}"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "weights.pt").write_bytes(f"weights-{epoch}".encode())
        (checkpoint_dir / "best.pt").write_bytes(f"best-{epoch}".encode())
        (checkpoint_dir / "checkpoint_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "epoch": epoch - 1,
                    "completed_epochs": epoch,
                    "target_epochs": 100,
                    "weights": "weights.pt",
                    "best_weights": "best.pt",
                    "metrics": {},
                }
            ),
            encoding="utf-8",
        )

    for epoch in (54, 55, 60):
        write_checkpoint(epoch)
    (root / "latest.json").write_text(
        json.dumps(
            {
                "completed_epochs": 60,
                "weights": "epoch_000060/weights.pt",
                "manifest": "epoch_000060/checkpoint_manifest.json",
            }
        ),
        encoding="utf-8",
    )
    config.payload["training"]["checkpointing"] = {
        "enabled": True,
        "resume": {
            "mode": "auto",
            "selected_epoch": 55,
            "prune_after_selected": True,
        },
    }

    source = resolve_model_source(
        config,
        artifacts_dir=config.artifacts_dir,
        rollback_to_selected=True,
    )
    latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))

    assert source.kind == "selected_checkpoint"
    assert source.selected_epoch == 55
    assert source.completed_epochs == 55
    assert Path(source.path) == (root / "epoch_000055" / "weights.pt").resolve()
    assert (root / "epoch_000054").is_dir()
    assert (root / "epoch_000055").is_dir()
    assert not (root / "epoch_000060").exists()
    assert latest["weights"] == "epoch_000055/weights.pt"
    assert latest["rollback"]["selected_epoch"] == 55
    assert source.rollback["deleted_checkpoints"] == ["epoch_000060"]


def test_training_configs_default_selected_resume_epoch_is_54():
    for path in (
        "configs/training/yolo_segmentation.yaml",
        "configs/training/yolo_segmentation_pretrain.yaml",
        "configs/training/yolo_segmentation_fine_tune.yaml",
    ):
        config = TrainingConfig.load(path)
        resume = config.payload["training"]["checkpointing"]["resume"]
        assert resume["selected_epoch"] == 54
        assert resume["prune_after_selected"] is True


def test_selected_checkpoint_resume_refuses_different_checkpoint_when_missing(
    tmp_path: Path,
):
    config = _config(tmp_path)
    root = config.artifacts_dir / "checkpoints"
    checkpoint_dir = root / "epoch_000060"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "weights.pt").write_bytes(b"weights")
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps({"completed_epochs": 60, "weights": "weights.pt"}),
        encoding="utf-8",
    )
    config.payload["training"]["checkpointing"] = {
        "enabled": True,
        "resume": {"mode": "auto", "selected_epoch": 54},
    }

    with pytest.raises(FileNotFoundError, match="refusing to resume"):
        resolve_model_source(config, artifacts_dir=config.artifacts_dir)


def test_train_args_enable_checkpoint_callback_without_duplicate_epoch_weights(
    tmp_path: Path,
):
    config = _config(tmp_path)
    config.payload["training"]["checkpointing"] = {
        "enabled": True,
        "interval_epochs": 5,
        "resume": {"mode": "never"},
    }
    trainer = YOLOTrainer(config, PipelineState(config.artifacts_dir / "state.json"))

    args = trainer._train_args({}, tuning=False, run_name=config.run_name)

    assert args["save"] is True
    assert args["save_period"] == -1
    assert resolve_model_source(config).kind == "base_model"


def test_train_args_use_configurable_early_stopping(tmp_path: Path):
    config = _config(tmp_path)
    config.payload["training"]["early_stopping"] = {
        "enabled": True,
        "patience": 7,
    }
    trainer = YOLOTrainer(config, PipelineState(config.artifacts_dir / "state.json"))

    args = trainer._train_args({}, tuning=False, run_name=config.run_name)
    tuned_args = trainer._train_args(
        {"patience": 3},
        tuning=True,
        run_name=f"{config.run_name}-trial",
    )

    assert args["patience"] == 7
    assert tuned_args["patience"] == 3


def test_train_args_can_disable_early_stopping(tmp_path: Path):
    config = _config(tmp_path)
    config.payload["training"]["early_stopping"] = {
        "enabled": False,
        "patience": 7,
    }
    trainer = YOLOTrainer(config, PipelineState(config.artifacts_dir / "state.json"))

    args = trainer._train_args({}, tuning=False, run_name=config.run_name)

    assert args["patience"] == 0


def test_early_stopping_patience_must_be_positive_when_enabled(tmp_path: Path):
    payload = _config(tmp_path).payload
    payload["training"]["early_stopping"] = {
        "enabled": True,
        "patience": 0,
    }

    with pytest.raises(TrainingConfigError, match="early_stopping.patience"):
        TrainingConfig(payload, tmp_path / "config.yaml")


def test_early_stopping_summary_reports_pre_target_stop(tmp_path: Path):
    config = _config(tmp_path)
    trainer = YOLOTrainer(config, PipelineState(config.artifacts_dir / "state.json"))
    yolo_trainer = SimpleNamespace(
        epoch=8,
        stop=True,
        stopper=SimpleNamespace(best_epoch=5, best_fitness=0.42),
    )

    report = trainer._early_stopping_report(
        yolo_trainer,
        train_args={"patience": 3},
        target_epochs=100,
    )

    assert report["enabled"] is True
    assert report["stopped_early"] is True
    assert report["completed_epochs"] == 9
    assert report["best_epoch"] == 5
    assert report["best_fitness"] == 0.42
    assert report["epochs_without_improvement"] == 4


def test_resume_target_supports_fixed_size_training_chunks(tmp_path: Path):
    config = _config(tmp_path)
    config.payload["training"]["epochs"] = 20
    config.payload["training"]["checkpointing"] = {
        "enabled": True,
        "interval_epochs": 5,
        "resume": {"mode": "auto", "additional_epochs": 20},
    }
    source = ModelSource(
        path=str(tmp_path / "checkpoint.pt"),
        kind="automatic_checkpoint",
        resume=True,
        completed_epochs=40,
    )

    assert resume_target_epochs(config, source) == 60
    assert resume_target_epochs(
        config,
        ModelSource("yolo26n-seg.pt", "base_model", False),
    ) == 20


def test_checkpoint_interval_must_be_positive(tmp_path: Path):
    payload = _config(tmp_path).payload
    payload["training"]["checkpointing"] = {
        "enabled": True,
        "interval_epochs": 0,
        "resume": {"mode": "auto"},
    }

    with pytest.raises(TrainingConfigError, match="interval_epochs"):
        TrainingConfig(payload, tmp_path / "config.yaml")


def test_selected_checkpoint_epoch_must_be_positive(tmp_path: Path):
    payload = _config(tmp_path).payload
    payload["training"]["checkpointing"] = {
        "enabled": True,
        "resume": {"mode": "auto", "selected_epoch": 0},
    }

    with pytest.raises(TrainingConfigError, match="selected_epoch"):
        TrainingConfig(payload, tmp_path / "config.yaml")


def test_training_output_bundle_excludes_data_images_and_restores_checkpoints(
    tmp_path: Path,
):
    config = _config(tmp_path)
    config.payload["publication"] = {
        "enabled": True,
        "provider": "kaggle",
        "dataset": "owner/training-outputs",
        "bundle_dir": str(tmp_path / "publication"),
        "require_checkpoint": True,
    }
    artifacts = config.artifacts_dir
    (artifacts / "models").mkdir(parents=True)
    (artifacts / "models" / "best.pt").write_bytes(b"best")
    checkpoint = artifacts / "checkpoints" / "epoch_000005"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.pt").write_bytes(b"resume")
    (checkpoint / "metrics_history.json").write_text("{}", encoding="utf-8")
    (checkpoint / "metrics_curves.png").write_bytes(b"plot")
    (artifacts / "checkpoints" / "latest.json").write_text(
        json.dumps({"weights": "epoch_000005/weights.pt"}),
        encoding="utf-8",
    )
    curves = artifacts / "evaluation" / "test" / "curves"
    curves.mkdir(parents=True)
    (curves / "PR_curve.png").write_bytes(b"curve")
    annotated = artifacts / "evaluation" / "test" / "annotated_samples"
    annotated.mkdir(parents=True)
    (annotated / "sample.jpg").write_bytes(b"raw-image")
    batches = artifacts / "training" / "annotated_batches"
    batches.mkdir(parents=True)
    (batches / "train_batch0.jpg").write_bytes(b"batch")
    config.dataset_dir.mkdir(parents=True)
    (config.dataset_dir / "dataset_manifest.json").write_text(
        json.dumps({"dataset_fingerprint": "data-v1"}),
        encoding="utf-8",
    )

    selected = collect_reusable_outputs(config)
    selected_names = {relative.as_posix() for _, relative in selected}
    assert "models/best.pt" in selected_names
    assert "checkpoints/epoch_000005/weights.pt" in selected_names
    assert "checkpoints/epoch_000005/metrics_curves.png" in selected_names
    assert "evaluation/test/curves/PR_curve.png" in selected_names
    assert not any("annotated_samples" in name for name in selected_names)
    assert not any("annotated_batches" in name for name in selected_names)
    assert not any(name.endswith(".jpg") for name in selected_names)

    publisher = TrainingOutputPublisher(
        config,
        PipelineState(artifacts / "pipeline_state.json"),
    )
    bundle = publisher.prepare_bundle()
    archive = Path(bundle["archive"])
    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
    assert "models/best.pt" in names
    assert "metadata/resolved-config.yaml" in names
    assert "metadata/dataset-manifest.json" in names
    assert not any("images/" in name or "labels/" in name for name in names)
    assert not any(name.endswith(".jpg") for name in names)

    restored = tmp_path / "restored-artifacts"
    report = restore_training_outputs(archive, restored)
    assert "checkpoints/epoch_000005/weights.pt" in report["restored"]
    assert (restored / "checkpoints" / "epoch_000005" / "weights.pt").is_file()
    assert not (restored / "metadata").exists()


def test_output_publisher_creates_or_versions_kaggle_dataset(tmp_path: Path, monkeypatch):
    import edge_ai_mass.training.publication as publication_module

    config = _config(tmp_path)
    config.payload["publication"] = {
        "enabled": True,
        "provider": "kaggle",
        "dataset": "owner/training-outputs",
        "bundle_dir": str(tmp_path / "publication"),
        "require_checkpoint": True,
        "version_notes": "unit-test",
    }
    models = config.artifacts_dir / "models"
    models.mkdir(parents=True)
    (models / "best.pt").write_bytes(b"model")
    state = PipelineState(config.artifacts_dir / "pipeline_state.json")

    class FakeApi:
        def __init__(self):
            self.created = 0
            self.versioned = 0

        def dataset_create_new(self, *args, **kwargs):
            self.created += 1
            return {"status": "ok"}

        def dataset_create_version(self, *args, **kwargs):
            self.versioned += 1
            return {"status": "ok"}

    api = FakeApi()
    monkeypatch.setattr(publication_module, "_authenticate_kaggle", lambda: api)
    monkeypatch.setattr(publication_module, "_wait_for_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        publication_module,
        "_owned_dataset",
        lambda *args, **kwargs: SimpleNamespace(current_version_number=3),
    )
    monkeypatch.setattr(publication_module, "_dataset_exists", lambda *args: False)

    created = TrainingOutputPublisher(config, state).publish()
    assert created["operation"] == "created"
    assert created["version"] == 3
    assert api.created == 1

    monkeypatch.setattr(publication_module, "_dataset_exists", lambda *args: True)
    versioned = TrainingOutputPublisher(config, state).publish()
    assert versioned["operation"] == "versioned"
    assert api.versioned == 1


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
