"""Build a thesis-ready evidence pack from installed YOLO training artifacts.

This script is intentionally read-only with respect to training artifacts. It
does not import Ultralytics, allocate GPU resources, or start training. It only
parses exported CSV/JSON/YAML files and indexes image artifacts that already
exist under the supplied artifact root.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MISSING = "[NOT_FOUND_IN_ARTIFACTS]"
REAL_REQUIRED = "[REAL_VALUE_REQUIRED]"

CLASS_NAMES = [
    "organic_waste",
    "plastic_bottle",
    "plastic_bag",
    "rigid_plastic",
    "metal_can",
    "glass",
    "paper_cardboard",
    "mixed_waste",
]

# Image-derived AP@0.5 values read from the exported Ultralytics PR curve
# legends. These are useful thesis evidence, but they are not a substitute for
# a structured per-class metrics export.
BOX_PR_IMAGE_AP50 = {
    "organic_waste": 0.981,
    "plastic_bottle": 0.640,
    "plastic_bag": 0.356,
    "rigid_plastic": 0.531,
    "metal_can": 0.660,
    "glass": 0.712,
    "paper_cardboard": 0.814,
    "mixed_waste": 0.498,
}

MASK_PR_IMAGE_AP50 = {
    "organic_waste": 0.981,
    "plastic_bottle": 0.629,
    "plastic_bag": 0.328,
    "rigid_plastic": 0.464,
    "metal_can": 0.650,
    "glass": 0.713,
    "paper_cardboard": 0.811,
    "mixed_waste": 0.447,
}

CONFUSION_DIAGONAL_IMAGE = {
    "organic_waste": 0.96,
    "plastic_bottle": 0.53,
    "plastic_bag": 0.30,
    "rigid_plastic": 0.49,
    "metal_can": 0.59,
    "glass": 0.62,
    "paper_cardboard": 0.79,
    "mixed_waste": 0.43,
}


@dataclass
class Context:
    artifact_root: Path
    evidence_root: Path
    output_root: Path
    run_root: Path
    mlflow_run_id: str | None
    files: list[Path]
    rel_files: list[str]
    pipeline_state_path: Path | None
    training_summary_path: Path | None
    latest_checkpoint_path: Path | None
    dataset_manifest_path: Path | None
    optuna_best_path: Path | None
    args_yaml_path: Path | None
    resolved_config_path: Path | None
    results_csv_path: Path | None
    pipeline_state: dict[str, Any]
    training_summary: dict[str, Any]
    latest_checkpoint: dict[str, Any]
    dataset_manifest: dict[str, Any]
    optuna_best: dict[str, Any]
    args_yaml: dict[str, Any]
    resolved_config: dict[str, Any]
    results_rows: list[dict[str, str]]


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml_flat(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        data: dict[str, Any] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = parse_scalar(value.strip())
        return data


def parse_scalar(value: str) -> Any:
    if value in {"null", "None", ""}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_csv_dicts(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return fmt(value)
    return str(value)


def fmt(value: Any) -> str:
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.6f}"
        return text.rstrip("0").rstrip(".")
    return str(value)


def rel(ctx: Context, path: Path | None) -> str:
    if path is None:
        return MISSING
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        try:
            return path.relative_to(ctx.artifact_root.parent).as_posix()
        except ValueError:
            return path.as_posix()


def existing_local_from_kaggle(ctx: Context, kaggle_path: str | None) -> str:
    if not kaggle_path:
        return MISSING
    prefix = "/kaggle/working/"
    if kaggle_path.startswith(prefix):
        local = Path(kaggle_path.replace(prefix, "kaggle-output/", 1))
        if local.exists():
            return local.as_posix()
        alt = Path(kaggle_path.replace(prefix, "", 1))
        if alt.exists():
            return alt.as_posix()
    return kaggle_path


def first_existing(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def find_first(root: Path, pattern: str, preferred: str | None = None) -> Path | None:
    paths = sorted(root.rglob(pattern))
    if preferred:
        preferred_paths = [path for path in paths if preferred.replace("\\", "/") in path.as_posix()]
        if preferred_paths:
            return preferred_paths[0]
    return paths[0] if paths else None


def discover_context(artifact_root: Path, evidence_root: Path) -> Context:
    artifact_root = artifact_root.resolve()
    evidence_root = evidence_root.resolve()
    output_root = evidence_root / "results_analysis"

    run_root = artifact_root / "artifacts" / "training" / "yolo" / "waste-seg-yolo"
    if not run_root.exists():
        candidates = sorted((artifact_root / "artifacts" / "training" / "yolo").glob("*"))
        run_root = candidates[0] if candidates else artifact_root

    files = sorted(path for path in artifact_root.rglob("*") if path.is_file())
    rel_files = [path.relative_to(Path.cwd()).as_posix() for path in files]

    pipeline_state_path = first_existing([run_root / "pipeline_state.json"])
    pipeline_state = read_json(pipeline_state_path)
    mlflow_run_id = pipeline_state.get("mlflow_run_id")
    if not mlflow_run_id and isinstance(pipeline_state.get("training"), dict):
        mlflow_run_id = pipeline_state["training"].get("mlflow_run_id")

    training_summary_path = first_existing(
        [
            run_root / "reports" / "training_summary.json",
            find_first(artifact_root, "training_summary.json", "artifacts/reports"),
        ]
    )
    latest_checkpoint_path = first_existing([run_root / "checkpoints" / "latest.json"])
    dataset_manifest_path = find_first(
        artifact_root, "dataset_manifest.json", "artifacts/lineage/dataset_manifest.json"
    )
    optuna_best_path = first_existing(
        [
            run_root / "optimization" / "optuna_best.json",
            find_first(artifact_root, "optuna_best.json", "artifacts/optimization"),
        ]
    )
    args_yaml_path = first_existing(
        [
            run_root / "training_runs" / "waste-seg-yolo" / "args.yaml",
            find_first(artifact_root, "args.yaml", "training_runs/waste-seg-yolo"),
        ]
    )
    resolved_config_path = find_first(artifact_root, "resolved_config.json")
    results_csv_path = first_existing(
        [
            run_root / "training" / "curves" / "results.csv",
            run_root / "training_runs" / "waste-seg-yolo" / "results.csv",
            find_first(artifact_root, "results.csv", "artifacts/training/curves"),
        ]
    )

    return Context(
        artifact_root=artifact_root,
        evidence_root=evidence_root,
        output_root=output_root,
        run_root=run_root,
        mlflow_run_id=mlflow_run_id,
        files=files,
        rel_files=rel_files,
        pipeline_state_path=pipeline_state_path,
        training_summary_path=training_summary_path,
        latest_checkpoint_path=latest_checkpoint_path,
        dataset_manifest_path=dataset_manifest_path,
        optuna_best_path=optuna_best_path,
        args_yaml_path=args_yaml_path,
        resolved_config_path=resolved_config_path,
        results_csv_path=results_csv_path,
        pipeline_state=read_json(pipeline_state_path),
        training_summary=read_json(training_summary_path),
        latest_checkpoint=read_json(latest_checkpoint_path),
        dataset_manifest=read_json(dataset_manifest_path),
        optuna_best=read_json(optuna_best_path),
        args_yaml=read_yaml_flat(args_yaml_path),
        resolved_config=read_json(resolved_config_path),
        results_rows=read_csv_dicts(results_csv_path),
    )


def nested(data: dict[str, Any], *keys: str, default: Any = MISSING) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def to_float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def final_row(ctx: Context) -> dict[str, str]:
    return ctx.results_rows[-1] if ctx.results_rows else {}


def best_row(ctx: Context, metric: str) -> dict[str, str]:
    candidates = [row for row in ctx.results_rows if to_float(row, metric) is not None]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: float(row[metric]))


def total_instances_by_class(ctx: Context) -> dict[str, int]:
    totals = {class_name: 0 for class_name in CLASS_NAMES}
    splits = ctx.dataset_manifest.get("splits", {})
    if isinstance(splits, dict):
        for split in splits.values():
            counts = split.get("instances_by_class", {}) if isinstance(split, dict) else {}
            for class_name, count in counts.items():
                totals[class_name] = totals.get(class_name, 0) + int(count)
    return totals


def artifact_kind(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower().lstrip(".") or "no_extension"
    if suffix in {"png", "jpg", "jpeg"}:
        return "image"
    if suffix in {"json", "yaml", "yml", "csv", "db"}:
        return suffix
    if suffix in {"pt", "onnx", "engine", "torchscript"}:
        return "model/checkpoint"
    if suffix == "html":
        return "html visualization"
    if name.endswith(".log"):
        return "log"
    return suffix


def artifact_purpose(path: Path) -> str:
    low = path.as_posix().lower()
    name = path.name.lower()
    if name == "pipeline_state.json":
        return "pipeline state and run lineage"
    if name == "training_summary.json":
        return "final training summary and best-model metrics"
    if name == "dataset_manifest.json":
        return "dataset lineage, splits, sources, and class distribution"
    if name == "results.csv":
        return "epoch-level training and validation metrics"
    if name == "metrics.json":
        return "checkpoint metric snapshot"
    if name == "metrics_history.json":
        return "checkpoint metric history"
    if name == "checkpoint_manifest.json":
        return "checkpoint checksum and epoch metadata"
    if name == "latest.json":
        return "latest managed checkpoint pointer"
    if name == "args.yaml":
        return "Ultralytics training arguments"
    if name == "optuna_best.json":
        return "Optuna best trial and tuned hyperparameters"
    if name in {"best.pt", "last.pt", "weights.pt"}:
        return "YOLO checkpoint weights"
    if "confusion_matrix" in name:
        return "confusion-matrix visual evidence"
    if "curve" in name or name == "results.png":
        return "metric curve visual evidence"
    if name == "labels.jpg":
        return "label distribution and box geometry diagnostics"
    if "batch" in name or "samples" in low or "dataset_mosaic" in name:
        return "qualitative dataset or prediction visual evidence"
    if "optimization" in low and name.endswith(".html"):
        return "Optuna visualization"
    if name.endswith(".db"):
        return "tracking or optimization database"
    return "supporting artifact"


def artifact_useful_for_thesis(path: Path) -> str:
    low = path.as_posix().lower()
    name = path.name.lower()
    useful_names = {
        "pipeline_state.json",
        "training_summary.json",
        "dataset_manifest.json",
        "results.csv",
        "metrics.json",
        "metrics_history.json",
        "checkpoint_manifest.json",
        "latest.json",
        "args.yaml",
        "optuna_best.json",
        "results.png",
        "labels.jpg",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "boxpr_curve.png",
        "maskpr_curve.png",
        "boxf1_curve.png",
        "maskf1_curve.png",
    }
    if name in useful_names:
        return "yes"
    if "dataset_visualizations" in low or "annotated_batches" in low or "val_batch" in name:
        return "yes, qualitative"
    if name in {"best.pt", "last.pt", "weights.pt"}:
        return "yes, traceability"
    if "mlflow.db" in low or "yolo.db" in low:
        return "supporting"
    return "supporting"


def artifact_numeric(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".csv", ".json", ".yaml", ".yml", ".db"}:
        return "yes"
    if name in {
        "boxpr_curve.png",
        "maskpr_curve.png",
        "boxf1_curve.png",
        "maskf1_curve.png",
        "confusion_matrix_normalized.png",
        "labels.jpg",
        "results.png",
    }:
        return "image-derived"
    return "no"


def artifact_notes(ctx: Context, path: Path) -> str:
    low = path.as_posix().lower()
    notes: list[str] = []
    if ctx.run_root in path.parents:
        notes.append("canonical run artifact")
    if "\\mlruns\\" in str(path).lower() or "/mlruns/" in low:
        notes.append("MLflow mirror/copy")
    if path.name.lower() in {"best.pt", "last.pt", "weights.pt"}:
        notes.append(f"size={path.stat().st_size} bytes")
    if path.name.lower() == "yolo-train-pipeline.log" and path.stat().st_size == 0:
        notes.append("empty log file")
    return "; ".join(notes) if notes else ""


def build_inventory_rows(ctx: Context) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in ctx.files:
        rows.append(
            {
                "relative_path": path.relative_to(Path.cwd()).as_posix(),
                "file_type": artifact_kind(path),
                "likely_purpose": artifact_purpose(path),
                "useful_for_thesis": artifact_useful_for_thesis(path),
                "contains_extractable_numeric_results": artifact_numeric(path),
                "notes": artifact_notes(ctx, path),
            }
        )
    return rows


def curve_rows(ctx: Context) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    image_exts = {".png", ".jpg", ".jpeg"}
    for path in ctx.files:
        low = path.as_posix().lower()
        name = path.name.lower()
        if path.suffix.lower() not in image_exts and path.suffix.lower() != ".html":
            continue
        if not (
            "curve" in name
            or "confusion" in name
            or name == "results.png"
            or name == "labels.jpg"
            or "batch" in name
            or "dataset_mosaic" in name
            or "samples" in low
            or "optimization" in low
        ):
            continue
        rows.append(
            {
                "artifact_path": path.relative_to(Path.cwd()).as_posix(),
                "artifact_type": artifact_kind(path),
                "likely_purpose": artifact_purpose(path),
                "thesis_use": curve_thesis_use(path),
                "contains_numeric_results": artifact_numeric(path),
                "notes": artifact_notes(ctx, path),
            }
        )
    return rows


def curve_thesis_use(path: Path) -> str:
    name = path.name.lower()
    low = path.as_posix().lower()
    if name == "results.png":
        return "show training/validation loss and overall metric trends"
    if "pr_curve" in name or "pr_curve" in low or "pr" in name:
        return "show precision-recall behavior and image-derived AP@0.5"
    if "f1_curve" in name:
        return "show confidence threshold behavior"
    if "p_curve" in name:
        return "show precision versus confidence"
    if "r_curve" in name:
        return "show recall versus confidence"
    if "confusion_matrix" in name:
        return "show class confusion and background errors"
    if name == "labels.jpg":
        return "show class imbalance and annotation geometry"
    if "batch" in name:
        return "show qualitative prediction/label examples"
    if "dataset_mosaic" in name or "samples" in low:
        return "show dataset visual diversity and annotation examples"
    if path.suffix.lower() == ".html":
        return "show Optuna tuning behavior"
    return "supporting figure"


def training_run_summary_rows(ctx: Context) -> list[dict[str, Any]]:
    summary = ctx.training_summary
    args = ctx.args_yaml
    latest = ctx.latest_checkpoint
    state = ctx.pipeline_state
    best_params = summary.get("best_params", {}) if isinstance(summary.get("best_params"), dict) else {}
    config_export = nested(ctx.resolved_config, "export", "enabled", default=MISSING)
    rows = [
        row("artifact_root", ctx.artifact_root.relative_to(Path.cwd()).as_posix(), ctx.artifact_root, "Installed artifact root supplied by prompt."),
        row("canonical_run_root", ctx.run_root.relative_to(Path.cwd()).as_posix(), ctx.run_root, "Primary run directory analyzed."),
        row("mlflow_run_id", ctx.mlflow_run_id or MISSING, ctx.pipeline_state_path, "Run id from pipeline_state.json."),
        row("task_type", args.get("task") or ctx.dataset_manifest.get("task") or nested(ctx.resolved_config, "model", "task", default=MISSING), ctx.args_yaml_path, "YOLO task."),
        row("model_checkpoint", args.get("model") or summary.get("checkpoint") or MISSING, ctx.args_yaml_path, "Base model/checkpoint used to start training."),
        row("dataset_yaml_path", args.get("data") or ctx.dataset_manifest.get("dataset_yaml") or MISSING, ctx.args_yaml_path, "Dataset YAML consumed by Ultralytics."),
        row("target_epochs", args.get("epochs") or summary.get("target_epochs") or MISSING, ctx.args_yaml_path, "Configured target epochs."),
        row("completed_epochs", latest.get("completed_epochs") or nested(summary, "latest_checkpoint", "completed_epochs", default=MISSING), ctx.latest_checkpoint_path, "Installed final checkpoint completed epochs."),
        row("image_size", args.get("imgsz") or best_params.get("imgsz") or MISSING, ctx.args_yaml_path, "Training image size."),
        row("batch_size", args.get("batch") or best_params.get("batch") or MISSING, ctx.args_yaml_path, "Final training batch size."),
        row("optimizer", args.get("optimizer") or best_params.get("optimizer") or MISSING, ctx.args_yaml_path, "Final training optimizer."),
        row("learning_rate_lr0", args.get("lr0") or best_params.get("lr0") or MISSING, ctx.args_yaml_path, "Initial learning rate."),
        row("learning_rate_lrf", args.get("lrf") or best_params.get("lrf") or MISSING, ctx.args_yaml_path, "Final LR multiplier."),
        row("device", args.get("device") or MISSING, ctx.args_yaml_path, "Training device."),
        row("run_name", args.get("name") or nested(ctx.resolved_config, "training", "run_name", default=MISSING), ctx.args_yaml_path, "Ultralytics run name."),
        row("start_timestamp", MISSING, None, "No explicit run start timestamp found."),
        row("end_timestamp", MISSING, None, "No explicit run end timestamp found."),
        row("checkpoint_created_at", latest.get("created_at", MISSING), ctx.latest_checkpoint_path, "Timestamp recorded in latest checkpoint manifest."),
        row("best_checkpoint_path", "kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/best.pt", ctx.training_summary_path, "Best model path from training_summary.json, localized to installed artifacts."),
        row("last_checkpoint_path", "kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/last.pt", ctx.training_summary_path, "Last model path from training_summary.json, localized to installed artifacts."),
        row("export_enabled", config_export, ctx.resolved_config_path, "Resolved config says export is disabled in the installed run." if config_export is False else "Export config value."),
        row("export_artifacts", MISSING, ctx.resolved_config_path, "No ONNX/TensorRT export files or exports_manifest.json were discovered."),
        row("optuna_best_metric", ctx.optuna_best.get("metric", MISSING), ctx.optuna_best_path, "Optuna optimization objective."),
        row("optuna_best_value", ctx.optuna_best.get("best_value", MISSING), ctx.optuna_best_path, "Tuning objective value, not a final test metric."),
        row("optuna_completed_trials", ctx.optuna_best.get("completed_trials", MISSING), ctx.optuna_best_path, "Trials attempted in installed artifact set."),
    ]
    for key in [
        "augment",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "copy_paste_mode",
        "multi_scale",
        "freeze",
        "mask_ratio",
        "overlap_mask",
    ]:
        rows.append(row(f"augmentation_or_training_param_{key}", args.get(key, best_params.get(key, MISSING)), ctx.args_yaml_path, "Tuned final training parameter."))
    return rows


def row(item: str, value: Any, source: Path | None, notes: str) -> dict[str, Any]:
    return {
        "item": item,
        "value": fmt(value),
        "source_file": source.relative_to(Path.cwd()).as_posix() if source else MISSING,
        "notes": notes,
    }


def dataset_distribution_rows(ctx: Context) -> list[dict[str, Any]]:
    manifest = ctx.dataset_manifest
    source_file = rel(ctx, ctx.dataset_manifest_path)
    rows: list[dict[str, Any]] = []

    totals = total_instances_by_class(ctx)
    for class_name in CLASS_NAMES:
        rows.append(
            {
                "scope": "overall",
                "split": "all",
                "source": "all",
                "class_name": class_name,
                "images": manifest.get("total_images", MISSING),
                "instances": totals.get(class_name, MISSING),
                "source_file": source_file,
                "notes": "Class total summed from split instances_by_class.",
            }
        )

    splits = manifest.get("splits", {})
    if isinstance(splits, dict):
        for split_name, split in splits.items():
            if not isinstance(split, dict):
                continue
            rows.append(
                {
                    "scope": "split_summary",
                    "split": split_name,
                    "source": "all",
                    "class_name": "all",
                    "images": split.get("images", MISSING),
                    "instances": split.get("instances", MISSING),
                    "source_file": source_file,
                    "notes": "Split summary from dataset_manifest.json.",
                }
            )
            for source_name, count in split.get("images_by_source", {}).items():
                rows.append(
                    {
                        "scope": "split_by_source",
                        "split": split_name,
                        "source": source_name,
                        "class_name": "all",
                        "images": count,
                        "instances": "",
                        "source_file": source_file,
                        "notes": "Image count by source in split.",
                    }
                )
            for class_name, count in split.get("instances_by_class", {}).items():
                rows.append(
                    {
                        "scope": "split_by_class",
                        "split": split_name,
                        "source": "all",
                        "class_name": class_name,
                        "images": "",
                        "instances": count,
                        "source_file": source_file,
                        "notes": "Instance count by class in split.",
                    }
                )

    sources = manifest.get("sources", {})
    if isinstance(sources, dict):
        for source_name, source in sources.items():
            if not isinstance(source, dict):
                continue
            rows.append(
                {
                    "scope": "source_summary",
                    "split": "all",
                    "source": source_name,
                    "class_name": "all",
                    "images": source.get("images", MISSING),
                    "instances": source.get("instances", MISSING),
                    "source_file": source_file,
                    "notes": "Source summary from dataset_manifest.json.",
                }
            )
            for class_name, count in source.get("classes", {}).items():
                rows.append(
                    {
                        "scope": "source_by_class",
                        "split": "all",
                        "source": source_name,
                        "class_name": class_name,
                        "images": "",
                        "instances": count,
                        "source_file": source_file,
                        "notes": "Source instance count by class.",
                    }
                )
    return rows


def overall_metrics_rows(ctx: Context) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    final = final_row(ctx)
    results_file = rel(ctx, ctx.results_csv_path)
    if final:
        epoch = final.get("epoch", MISSING)
        for metric in [
            "train/box_loss",
            "train/seg_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "train/sem_loss",
            "val/box_loss",
            "val/seg_loss",
            "val/cls_loss",
            "val/dfl_loss",
            "val/sem_loss",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
            "lr/pg0",
            "lr/pg1",
            "lr/pg2",
        ]:
            rows.append(
                {
                    "metric": f"final_epoch:{metric}",
                    "value": final.get(metric, MISSING),
                    "source_file": results_file,
                    "epoch_or_step": epoch,
                    "notes": "Final row in installed results.csv.",
                }
            )

    for metric in [
        "metrics/mAP50-95(M)",
        "metrics/mAP50(M)",
        "metrics/precision(M)",
        "metrics/recall(M)",
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
    ]:
        best = best_row(ctx, metric)
        if best:
            rows.append(
                {
                    "metric": f"best_csv_epoch:{metric}",
                    "value": best.get(metric, MISSING),
                    "source_file": results_file,
                    "epoch_or_step": best.get("epoch", MISSING),
                    "notes": "Best value for this metric across results.csv rows.",
                }
            )

    summary_metrics = ctx.training_summary.get("metrics", {})
    if isinstance(summary_metrics, dict):
        for metric, value in summary_metrics.items():
            rows.append(
                {
                    "metric": f"training_summary:{metric}",
                    "value": value,
                    "source_file": rel(ctx, ctx.training_summary_path),
                    "epoch_or_step": "best-model summary",
                    "notes": "Metric from training_summary.json; treat as validation unless Karim confirms otherwise.",
                }
            )

    latest_metrics = nested(ctx.latest_checkpoint, "metrics", "metrics", default={})
    if isinstance(latest_metrics, dict):
        for metric, value in latest_metrics.items():
            rows.append(
                {
                    "metric": f"latest_checkpoint:{metric}",
                    "value": value,
                    "source_file": rel(ctx, ctx.latest_checkpoint_path),
                    "epoch_or_step": ctx.latest_checkpoint.get("completed_epochs", MISSING),
                    "notes": "Metric snapshot in latest checkpoint pointer.",
                }
            )

    if not rows:
        rows.append(
            {
                "metric": "missing",
                "value": REAL_REQUIRED,
                "source_file": MISSING,
                "epoch_or_step": "missing",
                "notes": "No results.csv, metrics.json, or training_summary.json found.",
            }
        )
    return rows


def class_level_metrics_rows(ctx: Context) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    box_pr_source = "kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/BoxPR_curve.png"
    mask_pr_source = "kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png"
    for class_name in CLASS_NAMES:
        rows.append(
            {
                "class_name": class_name,
                "precision": REAL_REQUIRED,
                "recall": REAL_REQUIRED,
                "mAP50": fmt(BOX_PR_IMAGE_AP50[class_name]),
                "mAP50_95": REAL_REQUIRED,
                "mask_mAP50": fmt(MASK_PR_IMAGE_AP50[class_name]),
                "mask_mAP50_95": REAL_REQUIRED,
                "source_file": f"{box_pr_source}; {mask_pr_source}",
                "notes": "AP@0.5 values read from PR curve legends; structured per-class precision/recall and mAP@50:95 were not found, needs confirmation.",
            }
        )
    return rows


def checkpoint_rows(ctx: Context) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in ctx.files:
        suffix = path.suffix.lower()
        name = path.name.lower()
        low = path.as_posix().lower()
        if suffix not in {".pt", ".onnx", ".engine", ".torchscript", ".tflite"} and "exports_manifest" not in name:
            continue
        rel_path = path.relative_to(Path.cwd()).as_posix()
        artifact_type = "checkpoint"
        purpose = "training/resume/debug"
        thesis = "no"
        deployment = "no"
        notes = artifact_notes(ctx, path)
        if name == "best.pt" and "/models/" in low:
            purpose = "best model checkpoint for evaluation/export"
            thesis = "yes"
            deployment = "candidate, after export and benchmarking"
        elif name == "last.pt" and "/models/" in low:
            purpose = "last epoch checkpoint for resume/debugging"
            thesis = "supporting"
            deployment = "no"
        elif name == "best.pt":
            purpose = "best weights snapshot inside managed checkpoint"
            thesis = "supporting"
            deployment = "no, prefer models/best.pt unless repository owner confirms otherwise"
        elif name == "weights.pt":
            purpose = "managed checkpoint weights for resume/debug"
            thesis = "supporting"
            deployment = "no"
        elif suffix in {".onnx", ".engine", ".torchscript", ".tflite"}:
            artifact_type = "export"
            purpose = "deployable model export"
            thesis = "yes"
            deployment = "yes, if benchmarked on target device"
        rows.append(
            {
                "artifact_path": rel_path,
                "type": artifact_type,
                "purpose": purpose,
                "should_use_for_thesis": thesis,
                "should_use_for_deployment": deployment,
                "notes": notes,
            }
        )
    if not rows:
        rows.append(
            {
                "artifact_path": REAL_REQUIRED,
                "type": "missing",
                "purpose": "checkpoint/export artifact",
                "should_use_for_thesis": "missing",
                "should_use_for_deployment": "missing",
                "notes": "No checkpoint or export artifact found.",
            }
        )
    return rows


def missing_rows(ctx: Context) -> list[dict[str, Any]]:
    return [
        missing("final_run_confirmation", "Is the installed epoch-54 artifact the final thesis run or an intermediate output?", ctx.latest_checkpoint_path),
        missing("test_set_metrics", "No structured test metrics were found, despite a test split in dataset_manifest.json.", ctx.dataset_manifest_path),
        missing("class_level_precision_recall", "Per-class precision and recall were not found in structured CSV/JSON artifacts.", None),
        missing("class_level_mAP50_95", "Per-class mAP@50:95 values were not found in structured artifacts.", None),
        missing("evaluation_json", "No reports/evaluation.json artifact was found.", None),
        missing("export_artifacts", "No ONNX, TensorRT engine, or exports_manifest.json found; resolved config has export.enabled=false.", ctx.resolved_config_path),
        missing("jetson_latency_fps", "No Jetson Nano FPS, latency, memory, or power benchmark artifacts were found.", None),
        missing("mass_estimation_metrics", "No mass-estimation MAE/RMSE/R2 or end-to-end mass evaluation artifacts were found.", None),
        missing("run_start_end_timestamps", "No explicit run start/end timestamps were found in the installed artifacts.", None),
        missing("separate_inference_evaluation", "No full end-to-end inference or deployment evaluation report was found.", None),
    ]


def missing(item: str, notes: str, source: Path | None) -> dict[str, Any]:
    return {
        "item": item,
        "status": "missing_or_ambiguous",
        "source_or_expected_artifact": source.relative_to(Path.cwd()).as_posix() if source else REAL_REQUIRED,
        "notes": notes,
    }


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def clean(value: Any) -> str:
        text = csv_value(value)
        return text.replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(clean(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def bullet(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def classes_text(ctx: Context) -> str:
    classes = ctx.dataset_manifest.get("classes") or CLASS_NAMES
    return ", ".join(str(item) for item in classes)


def metric_value(ctx: Context, metric: str, source: str = "summary") -> str:
    if source == "summary":
        value = ctx.training_summary.get("metrics", {}).get(metric)
        return fmt(value) if value is not None else REAL_REQUIRED
    final = final_row(ctx)
    return final.get(metric, REAL_REQUIRED)


def write_readme(ctx: Context) -> None:
    missing = missing_rows(ctx)
    extracted = [
        f"Dataset size: {ctx.dataset_manifest.get('total_images', MISSING)} images and {ctx.dataset_manifest.get('total_instances', MISSING)} instances ({rel(ctx, ctx.dataset_manifest_path)}).",
        f"Final installed epoch row: epoch {final_row(ctx).get('epoch', MISSING)} ({rel(ctx, ctx.results_csv_path)}).",
        f"Best-model validation mask mAP@50:95: {metric_value(ctx, 'mask_map50_95')} ({rel(ctx, ctx.training_summary_path)}).",
        f"Best checkpoint: kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/best.pt ({rel(ctx, ctx.training_summary_path)}).",
        f"Optuna objective: {ctx.optuna_best.get('metric', MISSING)} = {fmt(ctx.optuna_best.get('best_value'))} ({rel(ctx, ctx.optuna_best_path)}).",
    ]
    content = f"""# Results Analysis Evidence Pack

This folder contains a thesis-ready extension for the installed YOLO segmentation training artifacts. It was generated from:

`{ctx.artifact_root.relative_to(Path.cwd()).as_posix()}`

The analysis uses only values found in the installed artifacts. Missing values remain explicit placeholders such as `{REAL_REQUIRED}` or `{MISSING}`. Prism AI must not invent missing training, test, deployment, or mass-estimation results.

## Extracted Values

{bullet(extracted)}

## Remaining Gaps

{md_table(missing, ["item", "status", "source_or_expected_artifact", "notes"])}

## How Prism AI Should Use This Folder

- Use `03_training_metrics_analysis.md`, `04_validation_and_segmentation_metrics.md`, and `tables/overall_metrics.csv` for quantitative validation metrics.
- Use `02_dataset_manifest_analysis.md` and `tables/dataset_distribution.csv` for dataset counts, splits, source lineage, and class distribution.
- Use `05_curves_and_visual_evidence.md`, `figures_index/figure_index.md`, and `figures_index/selected_figures_for_thesis.md` for figure choices.
- Use `09_thesis_ready_results_section.md` as a paste-ready draft, but keep placeholders until Karim confirms missing values.
- Treat YOLO segmentation validation results separately from full end-to-end mass-estimation results.
"""
    write_text(ctx.output_root / "README.md", content)


def write_artifact_inventory(ctx: Context, inventory: list[dict[str, Any]]) -> None:
    content = f"""# Artifact Inventory

Artifact root analyzed: `{ctx.artifact_root.relative_to(Path.cwd()).as_posix()}`

This inventory lists every discovered file under the artifact root. `MLflow mirror/copy` notes indicate duplicated artifacts stored under the MLflow run directory; thesis claims should generally cite the canonical run artifact unless the artifact only exists in MLflow lineage.

{md_table(inventory, ["relative_path", "file_type", "likely_purpose", "useful_for_thesis", "contains_extractable_numeric_results", "notes"])}
"""
    write_text(ctx.output_root / "00_artifact_inventory.md", content)


def write_training_run_summary(ctx: Context, rows: list[dict[str, Any]]) -> None:
    params = [row for row in rows if row["item"].startswith("augmentation_or_training_param_")]
    core = [row for row in rows if not row["item"].startswith("augmentation_or_training_param_")]
    content = f"""# Training Run Summary

This summary describes the installed YOLO segmentation run. The artifacts show a configured target of 60 epochs and an installed final checkpoint with 54 completed epochs, so the final-run status needs Karim confirmation.

## Core Run Metadata

{md_table(core, ["item", "value", "source_file", "notes"])}

## Tuned Training and Augmentation Parameters

{md_table(params, ["item", "value", "source_file", "notes"])}

## Interpretation

- The task is segmentation (`task: segment`) and the base model/checkpoint is `yolo26m-seg.pt`.
- The final training configuration used batch size 32, image size 640, optimizer SGD, and device `0`.
- Optuna was enabled with `mask_map50_95` as the objective. The Optuna best value belongs to the tuning study and should not be reported as the final model validation/test score.
- No export artifacts were found, and the resolved configuration records `export.enabled=false`.
"""
    write_text(ctx.output_root / "01_training_run_summary.md", content)


def write_dataset_analysis(ctx: Context) -> None:
    manifest = ctx.dataset_manifest
    source_file = rel(ctx, ctx.dataset_manifest_path)
    source_rows: list[dict[str, Any]] = []
    for name, src in manifest.get("sources", {}).items():
        source_rows.append(
            {
                "source": name,
                "images": src.get("images", MISSING),
                "instances": src.get("instances", MISSING),
                "invalid_annotations": src.get("invalid_annotations", MISSING),
                "empty_images": src.get("empty_images", MISSING),
                "box_polygon_annotations": src.get("box_polygon_annotations", MISSING),
                "notes": f"resolved root: {src.get('images_root_resolved', MISSING)}",
            }
        )
    split_rows: list[dict[str, Any]] = []
    for name, split in manifest.get("splits", {}).items():
        split_rows.append(
            {
                "split": name,
                "images": split.get("images", MISSING),
                "instances": split.get("instances", MISSING),
                "images_by_source": json.dumps(split.get("images_by_source", {}), sort_keys=True),
                "instances_by_class": json.dumps(split.get("instances_by_class", {}), sort_keys=True),
            }
        )
    class_rows = [
        {"class_name": class_name, "total_instances": count}
        for class_name, count in total_instances_by_class(ctx).items()
    ]
    content = f"""# Dataset Manifest Analysis

Source artifact: `{source_file}`

## Overview

- Task: `{manifest.get('task', MISSING)}`
- Dataset YAML: `{manifest.get('dataset_yaml', MISSING)}`
- Dataset fingerprint: `{manifest.get('dataset_fingerprint', MISSING)}`
- Data config digest: `{manifest.get('data_config_digest', MISSING)}`
- Total images: {manifest.get('total_images', MISSING)}
- Total annotations/instances: {manifest.get('total_instances', MISSING)}
- Duplicate images recorded: {manifest.get('duplicate_images', MISSING)}
- Classes: {classes_text(ctx)}

## Dataset Sources

{md_table(source_rows, ["source", "images", "instances", "invalid_annotations", "empty_images", "box_polygon_annotations", "notes"])}

## Split Sizes

{md_table(split_rows, ["split", "images", "instances", "images_by_source", "instances_by_class"])}

## Overall Class Distribution

{md_table(class_rows, ["class_name", "total_instances"])}

## Data Quality Notes

- RealWaste records 160 box-polygon annotations and 160 removed images, which supports mentioning a mask-quality/data-cleaning step.
- TACO records 155 invalid annotations, 167 dimension mismatches, 19 empty images, and 234 multipart reductions.
- AquaTrash records no invalid annotations, missing images, or bbox fallbacks in the manifest.
- Exact duplicate handling is recorded as 0 duplicate images.
"""
    write_text(ctx.output_root / "02_dataset_manifest_analysis.md", content)


def write_metrics_analysis(ctx: Context, overall_rows: list[dict[str, Any]]) -> None:
    best_metrics = []
    for metric in ["metrics/mAP50-95(M)", "metrics/mAP50(M)", "metrics/mAP50-95(B)", "metrics/mAP50(B)"]:
        best = best_row(ctx, metric)
        best_metrics.append(
            {
                "metric": metric,
                "best_value": best.get(metric, MISSING),
                "epoch": best.get("epoch", MISSING),
                "source": rel(ctx, ctx.results_csv_path),
            }
        )
    final = final_row(ctx)
    final_metrics = [
        {"metric": key, "value": final.get(key, MISSING), "source": rel(ctx, ctx.results_csv_path), "epoch": final.get("epoch", MISSING)}
        for key in [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
            "train/box_loss",
            "train/seg_loss",
            "train/cls_loss",
            "val/box_loss",
            "val/seg_loss",
            "val/cls_loss",
        ]
    ]
    content = f"""# Training Metrics Analysis

Primary metric source: `{rel(ctx, ctx.results_csv_path)}`  
Best-model summary source: `{rel(ctx, ctx.training_summary_path)}`

## Final Installed Epoch Metrics

{md_table(final_metrics, ["metric", "value", "source", "epoch"])}

## Best Epochs by Selected Metrics

{md_table(best_metrics, ["metric", "best_value", "epoch", "source"])}

## Best-Model Summary Metrics

- Box mAP@50: {metric_value(ctx, 'box_map50')} (`{rel(ctx, ctx.training_summary_path)}`)
- Box mAP@50:95: {metric_value(ctx, 'box_map50_95')} (`{rel(ctx, ctx.training_summary_path)}`)
- Mask mAP@50: {metric_value(ctx, 'mask_map50')} (`{rel(ctx, ctx.training_summary_path)}`)
- Mask mAP@50:95: {metric_value(ctx, 'mask_map50_95')} (`{rel(ctx, ctx.training_summary_path)}`)
- Box precision: {metric_value(ctx, 'metrics/precision(B)')} (`{rel(ctx, ctx.training_summary_path)}`)
- Box recall: {metric_value(ctx, 'metrics/recall(B)')} (`{rel(ctx, ctx.training_summary_path)}`)
- Mask precision: {metric_value(ctx, 'metrics/precision(M)')} (`{rel(ctx, ctx.training_summary_path)}`)
- Mask recall: {metric_value(ctx, 'metrics/recall(M)')} (`{rel(ctx, ctx.training_summary_path)}`)

## Metric Meaning Notes

- Box metrics evaluate object localization using bounding boxes.
- Mask metrics evaluate instance segmentation masks and are the more relevant results for this thesis section.
- mAP@50 uses an IoU threshold of 0.50 and is more permissive.
- mAP@50:95 averages AP over IoU thresholds from 0.50 to 0.95 and is stricter.
- Precision indicates how many predicted positives were correct.
- Recall indicates how many ground-truth objects were detected.

The table `tables/overall_metrics.csv` contains the extracted metric rows with source paths and epochs.
"""
    write_text(ctx.output_root / "03_training_metrics_analysis.md", content)


def write_validation_metrics(ctx: Context) -> None:
    content = f"""# Validation and Segmentation Metrics

The installed artifacts contain validation metrics from the YOLO training run. A test split exists in `dataset_manifest.json`, but no structured test-set evaluation report or `reports/evaluation.json` was found.

## Bounding-Box Detection Performance

- Best-model box precision: {metric_value(ctx, 'metrics/precision(B)')} (`{rel(ctx, ctx.training_summary_path)}`)
- Best-model box recall: {metric_value(ctx, 'metrics/recall(B)')} (`{rel(ctx, ctx.training_summary_path)}`)
- Best-model box mAP@50: {metric_value(ctx, 'box_map50')} (`{rel(ctx, ctx.training_summary_path)}`)
- Best-model box mAP@50:95: {metric_value(ctx, 'box_map50_95')} (`{rel(ctx, ctx.training_summary_path)}`)

## Mask/Segmentation Performance

- Best-model mask precision: {metric_value(ctx, 'metrics/precision(M)')} (`{rel(ctx, ctx.training_summary_path)}`)
- Best-model mask recall: {metric_value(ctx, 'metrics/recall(M)')} (`{rel(ctx, ctx.training_summary_path)}`)
- Best-model mask mAP@50: {metric_value(ctx, 'mask_map50')} (`{rel(ctx, ctx.training_summary_path)}`)
- Best-model mask mAP@50:95: {metric_value(ctx, 'mask_map50_95')} (`{rel(ctx, ctx.training_summary_path)}`)
- Final installed epoch mask mAP@50: {metric_value(ctx, 'metrics/mAP50(M)', source='final')} (`{rel(ctx, ctx.results_csv_path)}`, epoch {final_row(ctx).get('epoch', MISSING)})
- Final installed epoch mask mAP@50:95: {metric_value(ctx, 'metrics/mAP50-95(M)', source='final')} (`{rel(ctx, ctx.results_csv_path)}`, epoch {final_row(ctx).get('epoch', MISSING)})

## Validation Versus Test Status

- Validation metrics are present in `results.csv`, `training_summary.json`, and checkpoint metric files.
- Test metrics are missing from the installed artifacts and must remain `{REAL_REQUIRED}` until a test evaluation artifact is supplied.
- The presence of `dataset_manifest.json` test split counts does not prove test-set model performance.
"""
    write_text(ctx.output_root / "04_validation_and_segmentation_metrics.md", content)


def write_curves_analysis(ctx: Context) -> None:
    figure_rows = [
        {
            "figure": "results.png",
            "artifact_path": "kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.png",
            "what_it_shows": "training/validation losses and aggregate box/mask metrics across epochs",
            "thesis_use": "supports learning-curve discussion",
            "concerns": "losses trend downward and mAP trends upward; recall and validation losses show some jitter; overfitting claim needs confirmation",
        },
        {
            "figure": "MaskPR_curve.png",
            "artifact_path": "kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png",
            "what_it_shows": "mask precision-recall curves and image-derived class AP@0.5 values",
            "thesis_use": "best figure for segmentation class-level discussion",
            "concerns": "plastic_bag, mixed_waste, and rigid_plastic have weaker AP@0.5 than organic_waste and paper_cardboard",
        },
        {
            "figure": "BoxPR_curve.png",
            "artifact_path": "kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/BoxPR_curve.png",
            "what_it_shows": "box precision-recall curves and image-derived class AP@0.5 values",
            "thesis_use": "compare detection versus segmentation behavior",
            "concerns": "class ranking is similar to mask curve; image-derived values need confirmation",
        },
        {
            "figure": "confusion_matrix_normalized.png",
            "artifact_path": "kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/confusion_matrix_normalized.png",
            "what_it_shows": "normalized class confusion including background",
            "thesis_use": "supports class confusion and false-negative discussion",
            "concerns": "background row is prominent for plastic_bag, rigid_plastic, glass, and mixed_waste; visual interpretation needs confirmation",
        },
        {
            "figure": "labels.jpg",
            "artifact_path": "kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/diagnostics/labels.jpg",
            "what_it_shows": "training-set class distribution and bounding-box geometry",
            "thesis_use": "supports class imbalance discussion",
            "concerns": "mixed_waste has more training instances than glass or plastic_bottle",
        },
        {
            "figure": "val_batch0_pred.jpg and val_batch0_labels.jpg",
            "artifact_path": "kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/annotated_batches/val_batch0_pred.jpg; kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/annotated_batches/val_batch0_labels.jpg",
            "what_it_shows": "side-by-side qualitative prediction and label examples",
            "thesis_use": "illustrate successes, missed detections, class substitutions, and cluttered scenes",
            "concerns": "small visual sample; do not generalize without a larger qualitative audit",
        },
    ]
    content = f"""# Curves and Visual Evidence

The installed run includes quantitative curves, confusion matrices, label diagnostics, dataset mosaics, and annotated train/validation batches. Visual interpretation is secondary evidence; numeric claims should cite CSV/JSON artifacts when available.

{md_table(figure_rows, ["figure", "artifact_path", "what_it_shows", "thesis_use", "concerns"])}

## Supported Visual Observations

- The loss and mAP curves in `results.png` are consistent with learning over the installed epochs.
- `labels.jpg` and `dataset_manifest.json` show class imbalance, with mixed_waste and rigid_plastic having more instances than glass or plastic_bottle.
- `MaskPR_curve.png` and `BoxPR_curve.png` show stronger AP@0.5 for organic_waste and paper_cardboard, and weaker AP@0.5 for plastic_bag and mixed_waste.
- The normalized confusion matrix suggests background-related misses for several classes, especially plastic_bag, rigid_plastic, glass, and mixed_waste; this is image-derived and should be confirmed if a numeric confusion matrix becomes available.
"""
    write_text(ctx.output_root / "05_curves_and_visual_evidence.md", content)


def write_class_analysis(ctx: Context) -> None:
    rows = class_level_metrics_rows(ctx)
    ranked = sorted(rows, key=lambda row: float(row["mask_mAP50"]))
    weak = ", ".join(row["class_name"] for row in ranked[:3])
    strong = ", ".join(row["class_name"] for row in ranked[-3:][::-1])
    content = f"""# Class-Level Performance Analysis

Structured per-class metric CSV/JSON artifacts were not found. However, the exported PR curve images include readable class AP@0.5 values in their legends. The values below are therefore image-derived and marked as needing confirmation.

{md_table(rows, ["class_name", "precision", "recall", "mAP50", "mAP50_95", "mask_mAP50", "mask_mAP50_95", "source_file", "notes"])}

## Strong and Weak Classes

- Stronger mask AP@0.5 classes from `MaskPR_curve.png`: {strong}.
- Weaker mask AP@0.5 classes from `MaskPR_curve.png`: {weak}.
- The normalized confusion matrix diagonal appears strongest for organic_waste and paper_cardboard, and weaker for plastic_bag, mixed_waste, and rigid_plastic (`confusion_matrix_normalized.png`).

## Possible Reasons

- Class imbalance is supported by `dataset_manifest.json` and `labels.jpg`.
- Plastic bags and mixed waste may be visually heterogeneous or deformable; this is a hypothesis and needs Karim confirmation.
- Background confusion and missed objects are visible in `confusion_matrix_normalized.png` and validation batch examples, but exact false-positive/false-negative counts were not found.
"""
    write_text(ctx.output_root / "06_class_level_performance_analysis.md", content)


def write_error_analysis(ctx: Context) -> None:
    rows = [
        issue("Background/false-negative errors for plastic_bag", "confusion_matrix_normalized.png shows high background value for plastic_bag; val_batch examples include missed plastic_bag labels", "small/deformable objects, clutter, annotation difficulty", "Discuss as a key validation failure mode, not as a test-set result.", "more plastic-bag examples, targeted augmentation, inspect labels", "medium"),
        issue("Mixed waste lower class performance", "MaskPR_curve.png shows mixed_waste AP@0.5 = 0.447; confusion_matrix_normalized.png diagonal about 0.43", "heterogeneous class definition, visual similarity to other debris, background clutter", "Mixed waste is broad and may be difficult for a compact segmentation model.", "review taxonomy, split subclasses if needed, collect examples", "medium"),
        issue("Rigid plastic moderate/weak segmentation", "MaskPR_curve.png shows rigid_plastic AP@0.5 = 0.464; confusion_matrix_normalized.png diagonal about 0.49", "visual similarity with plastic bottle/bag or background; needs confirmation", "Report as a weaker class supported by visual curve evidence.", "class-specific validation audit and additional examples", "medium"),
        issue("Class imbalance", "dataset_manifest.json class counts and labels.jpg", "dataset composition: mixed_waste and rigid_plastic have more instances than glass or plastic_bottle", "May influence learned decision boundaries and thesis discussion of dataset limitations.", "rebalance sampling, class-aware augmentation, collect underrepresented classes", "high"),
        issue("Qualitative missed detections and substitutions", "val_batch0_labels.jpg compared with val_batch0_pred.jpg", "small/occluded objects, cluttered outdoor scenes, confidence threshold behavior", "Use as qualitative examples only.", "inspect more batches, tune confidence threshold, add hard negatives", "low"),
        issue("No test-set performance artifact", "dataset_manifest.json has a test split, but no evaluation.json/test metrics file was found", "evaluation stage may not have run or was not included in installed artifacts", "Do not claim test performance.", "run/export a test evaluation report later, without changing these artifact claims", "high"),
    ]
    content = f"""# Error Analysis and Failure Modes

The table below combines structured metrics with visual evidence. Confidence reflects evidence strength in the installed artifacts, not model certainty.

{md_table(rows, ["Observed issue", "Evidence artifact", "Possible cause", "Thesis interpretation", "Suggested mitigation", "Confidence"])}
"""
    write_text(ctx.output_root / "07_error_analysis_and_failure_modes.md", content)


def issue(observed: str, evidence: str, cause: str, interpretation: str, mitigation: str, confidence: str) -> dict[str, str]:
    return {
        "Observed issue": observed,
        "Evidence artifact": evidence,
        "Possible cause": cause,
        "Thesis interpretation": interpretation,
        "Suggested mitigation": mitigation,
        "Confidence": confidence,
    }


def write_model_selection(ctx: Context, checkpoints: list[dict[str, Any]]) -> None:
    content = f"""# Model Selection and Best Checkpoint

Best checkpoint identified: `kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/best.pt` (`{rel(ctx, ctx.training_summary_path)}`)  
Last checkpoint identified: `kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/last.pt` (`{rel(ctx, ctx.training_summary_path)}`)

## Checkpoint and Export Artifacts

{md_table(checkpoints, ["artifact_path", "type", "purpose", "should_use_for_thesis", "should_use_for_deployment", "notes"])}

## Recommendation

- Use `models/best.pt` as the thesis-reported model checkpoint because it is the repository's best model artifact and the training docs state that evaluation/export should use the best checkpoint.
- Use `last.pt` mainly for resume/debugging unless Karim confirms a repository-specific reason to prefer it.
- No ONNX or TensorRT export artifacts were found. Deployment claims on Jetson Nano should remain placeholders until export and benchmark artifacts exist.
- The large `checkpoints/epoch_000054/weights.pt` and `checkpoints/epoch_000054/best.pt` files are managed checkpoint snapshots; the smaller `models/best.pt` and `models/last.pt` are the final model artifacts copied from the Ultralytics run.
"""
    write_text(ctx.output_root / "08_model_selection_and_best_checkpoint.md", content)


def write_thesis_section(ctx: Context) -> None:
    content = f"""# Experimental Results: YOLO Segmentation Training

## Experimental Setup

A YOLO segmentation model was trained for the waste-characterization stage of the Edge AI mass-estimation pipeline. The installed artifacts identify the task as `segment`, the base checkpoint as `yolo26m-seg.pt`, the dataset configuration as `/kaggle/working/data/processed/waste_seg_yolo/dataset.yaml`, image size as 640, batch size as 32, optimizer as SGD, and training device as `0` (`{rel(ctx, ctx.args_yaml_path)}`). The run name was `waste-seg-yolo` (`{rel(ctx, ctx.args_yaml_path)}`). The configured target was 60 epochs, while the installed final checkpoint records 54 completed epochs (`{rel(ctx, ctx.latest_checkpoint_path)}`); this run-finality status requires confirmation.

## Dataset Summary

The training dataset merged TACO, AquaTrash, and RealWaste into eight classes: {classes_text(ctx)} (`{rel(ctx, ctx.dataset_manifest_path)}`). The manifest records {ctx.dataset_manifest.get('total_images', MISSING)} images and {ctx.dataset_manifest.get('total_instances', MISSING)} object instances (`{rel(ctx, ctx.dataset_manifest_path)}`). The split sizes were {nested(ctx.dataset_manifest, 'splits', 'train', 'images')} training images, {nested(ctx.dataset_manifest, 'splits', 'val', 'images')} validation images, and {nested(ctx.dataset_manifest, 'splits', 'test', 'images')} test images (`{rel(ctx, ctx.dataset_manifest_path)}`). No duplicate images were recorded (`{rel(ctx, ctx.dataset_manifest_path)}`).

## Quantitative Validation Results

| Metric | Value | Source |
| --- | --- | --- |
| Box precision | {metric_value(ctx, 'metrics/precision(B)')} | `{rel(ctx, ctx.training_summary_path)}` |
| Box recall | {metric_value(ctx, 'metrics/recall(B)')} | `{rel(ctx, ctx.training_summary_path)}` |
| Box mAP@50 | {metric_value(ctx, 'box_map50')} | `{rel(ctx, ctx.training_summary_path)}` |
| Box mAP@50:95 | {metric_value(ctx, 'box_map50_95')} | `{rel(ctx, ctx.training_summary_path)}` |
| Mask precision | {metric_value(ctx, 'metrics/precision(M)')} | `{rel(ctx, ctx.training_summary_path)}` |
| Mask recall | {metric_value(ctx, 'metrics/recall(M)')} | `{rel(ctx, ctx.training_summary_path)}` |
| Mask mAP@50 | {metric_value(ctx, 'mask_map50')} | `{rel(ctx, ctx.training_summary_path)}` |
| Mask mAP@50:95 | {metric_value(ctx, 'mask_map50_95')} | `{rel(ctx, ctx.training_summary_path)}` |

The final installed epoch row reports mask mAP@50 = {metric_value(ctx, 'metrics/mAP50(M)', source='final')} and mask mAP@50:95 = {metric_value(ctx, 'metrics/mAP50-95(M)', source='final')} at epoch {final_row(ctx).get('epoch', MISSING)} (`{rel(ctx, ctx.results_csv_path)}`). The best CSV epoch for mask mAP@50:95 was epoch {best_row(ctx, 'metrics/mAP50-95(M)').get('epoch', MISSING)} with value {best_row(ctx, 'metrics/mAP50-95(M)').get('metrics/mAP50-95(M)', MISSING)} (`{rel(ctx, ctx.results_csv_path)}`).

## Curve and Qualitative Analysis

The loss and metric curves show decreasing training/validation losses and increasing mAP over the installed epochs (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.png`). The Mask PR curve gives image-derived AP@0.5 values, with organic_waste and paper_cardboard visually strongest and plastic_bag/mixed_waste weaker (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png`). The normalized confusion matrix suggests background-related misses for several classes (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/confusion_matrix_normalized.png`). Annotated validation batches provide qualitative examples of correct detections, missed objects, and category substitutions (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/annotated_batches/val_batch0_pred.jpg`; `kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/annotated_batches/val_batch0_labels.jpg`).

## Limitations

No structured test-set metrics were found, so test performance remains {REAL_REQUIRED}. No Jetson Nano FPS/latency benchmark artifacts were found, so edge-deployment performance remains {REAL_REQUIRED}. No mass-estimation metrics were found, so these YOLO segmentation results must not be presented as full mass-estimation performance. No ONNX or TensorRT export artifact was found, and the resolved configuration records export as disabled (`{rel(ctx, ctx.resolved_config_path)}`).

## Transition to Mass Estimation

These results support the object segmentation component of the pipeline. The next evaluation step should measure how segmentation quality affects depth-assisted volume estimation and mass prediction, using separate mass-estimation and edge-inference benchmark artifacts.
"""
    write_text(ctx.output_root / "09_thesis_ready_results_section.md", content)


def write_prism_notes(ctx: Context) -> None:
    content = f"""# Prism AI Results Integration Notes

## Safe Sources

- Dataset counts and sources: `{rel(ctx, ctx.dataset_manifest_path)}`
- Training configuration: `{rel(ctx, ctx.args_yaml_path)}` and `{rel(ctx, ctx.resolved_config_path)}`
- Validation metrics: `{rel(ctx, ctx.training_summary_path)}` and `{rel(ctx, ctx.results_csv_path)}`
- Checkpoint identity: `{rel(ctx, ctx.training_summary_path)}`, `{rel(ctx, ctx.latest_checkpoint_path)}`
- Visual evidence: `kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/` and `kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/annotated_batches/`

## Values Safe to Use

- Total dataset size and split sizes from `dataset_manifest.json`.
- Overall validation box/mask precision, recall, mAP@50, and mAP@50:95 from `training_summary.json` and `results.csv`.
- Best checkpoint path `models/best.pt`.
- Optuna tuning metadata, only as tuning metadata, from `optuna_best.json`.

## Keep as Placeholders

- Test-set metrics.
- Jetson Nano FPS, latency, power, and memory.
- End-to-end mass-estimation accuracy.
- ONNX/TensorRT deployment results.
- Structured per-class precision, recall, and mAP@50:95.

## Writing Guidance

- Cite artifact paths beside every numeric claim.
- Distinguish validation segmentation metrics from full system performance.
- Discuss weak classes professionally as observed validation limitations.
- Do not overclaim deployment readiness; no export or Jetson benchmark artifact was found.
- Mark image-derived class AP@0.5 and confusion-matrix interpretations as needing confirmation if the thesis requires strict numeric tables.
"""
    write_text(ctx.output_root / "10_prism_ai_results_integration_notes.md", content)


def write_missing_questions(ctx: Context) -> None:
    questions = [
        "Is this installed artifact set the final thesis training run or an intermediate run?",
        "Should `kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/best.pt` be treated as the thesis best model?",
        "Was the run trained on the intended merged TACO + AquaTrash + RealWaste dataset?",
        "Are the reported metrics validation metrics only, or is there a separate test evaluation artifact?",
        "Are there test-set metrics, COCO JSON outputs, or `reports/evaluation.json` files not included in `kaggle-output`?",
        "Are there Jetson Nano FPS/latency/power benchmark artifacts?",
        "Are there mass-estimation evaluation artifacts, such as MAE, RMSE, or R2?",
        "Should the thesis include image-derived class AP@0.5 values from the PR curves, or wait for a structured class metrics export?",
        "Which figures should be included in the final thesis: results curve, Mask PR curve, confusion matrix, labels distribution, or validation examples?",
        "Which weak classes should be discussed in the limitations section?",
        "Should ONNX or TensorRT export be performed later for deployment evidence?",
    ]
    content = f"""# Missing Values and Questions for Karim

## Prioritized Missing or Ambiguous Items

{md_table(missing_rows(ctx), ["item", "status", "source_or_expected_artifact", "notes"])}

## Questions

{bullet(questions)}
"""
    write_text(ctx.output_root / "11_missing_values_and_questions_for_karim.md", content)


def write_figure_indexes(ctx: Context) -> None:
    figure_rows = curve_rows(ctx)
    write_text(
        ctx.output_root / "figures_index" / "figure_index.md",
        f"""# Figure Index

Artifact root: `{ctx.artifact_root.relative_to(Path.cwd()).as_posix()}`

{md_table(figure_rows, ["artifact_path", "artifact_type", "likely_purpose", "thesis_use", "contains_numeric_results", "notes"])}
""",
    )
    selected = [
        {
            "rank": 1,
            "figure": "results.png",
            "artifact_path": "kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.png",
            "why": "Best overview of convergence, losses, and overall validation metrics.",
        },
        {
            "rank": 2,
            "figure": "MaskPR_curve.png",
            "artifact_path": "kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png",
            "why": "Most relevant class-level segmentation evidence; includes image-derived AP@0.5 by class.",
        },
        {
            "rank": 3,
            "figure": "confusion_matrix_normalized.png",
            "artifact_path": "kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/confusion_matrix_normalized.png",
            "why": "Useful for discussing class confusion and background-related misses.",
        },
        {
            "rank": 4,
            "figure": "labels.jpg",
            "artifact_path": "kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/diagnostics/labels.jpg",
            "why": "Supports dataset imbalance and object geometry discussion.",
        },
        {
            "rank": 5,
            "figure": "val_batch0_pred.jpg with val_batch0_labels.jpg",
            "artifact_path": "kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/annotated_batches/val_batch0_pred.jpg; kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/annotated_batches/val_batch0_labels.jpg",
            "why": "Provides qualitative prediction examples and visible failure modes.",
        },
    ]
    write_text(
        ctx.output_root / "figures_index" / "selected_figures_for_thesis.md",
        f"""# Selected Figures for Thesis

Recommended figures are ranked by thesis value and traceability.

{md_table(selected, ["rank", "figure", "artifact_path", "why"])}

Do not treat visual figure legends as a replacement for structured test metrics. For class AP@0.5 values read from PR curve legends, use `needs confirmation` if the final thesis requires fully machine-readable per-class metrics.
""",
    )


def traceability_rows(ctx: Context) -> list[dict[str, Any]]:
    return [
        trace("YOLO segmentation model was trained", rel(ctx, ctx.training_summary_path), f"task={ctx.args_yaml.get('task', MISSING)}, model={ctx.args_yaml.get('model', MISSING)}", "supported", "Training summary and args.yaml are present."),
        trace("Best checkpoint was produced", "kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/best.pt", "best.pt exists", "supported", "Use for thesis model identity."),
        trace("Last checkpoint was produced", "kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/last.pt", "last.pt exists", "supported", "Use for resume/debug traceability."),
        trace("Training curves were generated", "kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.png", "results.png exists", "supported", "Use as convergence figure."),
        trace("Evaluation metrics were produced", rel(ctx, ctx.results_csv_path), f"final epoch={final_row(ctx).get('epoch', MISSING)}", "supported", "Validation metrics only unless confirmed otherwise."),
        trace("Dataset manifest was generated", rel(ctx, ctx.dataset_manifest_path), f"{ctx.dataset_manifest.get('total_images', MISSING)} images; {ctx.dataset_manifest.get('total_instances', MISSING)} instances", "supported", "Contains split/source/class distribution."),
        trace("Export artifacts were produced", REAL_REQUIRED, REAL_REQUIRED, "missing", "No ONNX/TensorRT exports or exports_manifest.json found."),
        trace("Jetson Nano benchmark was produced", REAL_REQUIRED, REAL_REQUIRED, "missing", "No FPS/latency artifacts found."),
        trace("Mass-estimation results were produced", REAL_REQUIRED, REAL_REQUIRED, "missing", "No mass metrics found."),
    ]


def trace(claim: str, artifact: str, value: str, status: str, notes: str) -> dict[str, str]:
    return {
        "Thesis claim": claim,
        "Artifact path": artifact,
        "Extracted value": value,
        "Status": status,
        "Notes": notes,
    }


def generated_section_for(file_name: str, ctx: Context) -> str:
    if file_name.startswith("05_"):
        return f"""## Training Results Artifact Analysis

The installed run artifacts under `{ctx.artifact_root.relative_to(Path.cwd()).as_posix()}` show that the governed YOLO segmentation pipeline produced dataset lineage, Optuna tuning output, checkpointed training metrics, curves, and best/last model checkpoints. Key source files are `{rel(ctx, ctx.dataset_manifest_path)}`, `{rel(ctx, ctx.training_summary_path)}`, `{rel(ctx, ctx.results_csv_path)}`, and `{rel(ctx, ctx.latest_checkpoint_path)}`.

Use the generated extension at `docs/thesis_evidence_pack/results_analysis/` for detailed traceability. The installed artifacts do not include structured test metrics, deployment exports, Jetson benchmarks, or mass-estimation metrics.
"""
    if file_name.startswith("10_"):
        return f"""## Training Results Artifact Analysis

The real artifacts support several honest limitations: no structured test-set metrics were found, no ONNX/TensorRT export files were found, no Jetson Nano latency/FPS artifacts were found, and no mass-estimation performance artifacts were found. Class-level visual evidence suggests weaker segmentation behavior for plastic_bag, mixed_waste, and rigid_plastic, but these class-level values are image-derived from PR curves and need confirmation if used as formal numeric results.
"""
    if file_name.startswith("11_"):
        return f"""## Training Results Artifact Analysis

The project contribution is now supported by real training artifacts: a merged dataset manifest, a tuned YOLO segmentation training run, periodic checkpoints, best/last model artifacts, training curves, confusion matrices, and qualitative validation examples. The main traceable result is mask mAP@50:95 = {metric_value(ctx, 'mask_map50_95')} from `{rel(ctx, ctx.training_summary_path)}`. This is a segmentation validation result, not a full mass-estimation or Jetson deployment result.
"""
    if file_name.startswith("12_"):
        return f"""## Extracted YOLO Segmentation Training Results

Artifact root: `{ctx.artifact_root.relative_to(Path.cwd()).as_posix()}`

| Value | Extracted result | Source |
| --- | --- | --- |
| Dataset size | {ctx.dataset_manifest.get('total_images', MISSING)} images, {ctx.dataset_manifest.get('total_instances', MISSING)} instances | `{rel(ctx, ctx.dataset_manifest_path)}` |
| Train/val/test images | {nested(ctx.dataset_manifest, 'splits', 'train', 'images')}/{nested(ctx.dataset_manifest, 'splits', 'val', 'images')}/{nested(ctx.dataset_manifest, 'splits', 'test', 'images')} | `{rel(ctx, ctx.dataset_manifest_path)}` |
| Task | {ctx.args_yaml.get('task', MISSING)} | `{rel(ctx, ctx.args_yaml_path)}` |
| Model checkpoint | {ctx.args_yaml.get('model', MISSING)} | `{rel(ctx, ctx.args_yaml_path)}` |
| Completed epochs | {ctx.latest_checkpoint.get('completed_epochs', MISSING)} | `{rel(ctx, ctx.latest_checkpoint_path)}` |
| Best-model mask mAP@50 | {metric_value(ctx, 'mask_map50')} | `{rel(ctx, ctx.training_summary_path)}` |
| Best-model mask mAP@50:95 | {metric_value(ctx, 'mask_map50_95')} | `{rel(ctx, ctx.training_summary_path)}` |
| Best checkpoint | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/best.pt` | `{rel(ctx, ctx.training_summary_path)}` |

Values still missing: test metrics, class-level structured precision/recall and mAP@50:95, ONNX/TensorRT exports, Jetson Nano benchmarks, and mass-estimation metrics.
"""
    if file_name.startswith("13_"):
        return f"""## Training Results Artifact Analysis

Prism AI may use `docs/thesis_evidence_pack/results_analysis/09_thesis_ready_results_section.md` as a draft results section. Every numeric claim must keep its artifact-path citation. Prism AI must distinguish validation YOLO segmentation metrics from end-to-end mass-estimation and Jetson deployment performance. Missing values must remain placeholders.
"""
    if file_name.startswith("15_"):
        rows = traceability_rows(ctx)
        return f"""## Training Results Artifact Analysis

{md_table(rows, ["Thesis claim", "Artifact path", "Extracted value", "Status", "Notes"])}
"""
    if file_name == "OPEN_QUESTIONS_FOR_KARIM.md":
        return f"""## Training Results Artifact Analysis

{bullet([
            'Is this epoch-54 installed artifact set the final thesis training run?',
            'Should models/best.pt be considered the thesis best model?',
            'Are there separate test-set metrics or an evaluation.json artifact?',
            'Are there Jetson Nano FPS/latency benchmark artifacts?',
            'Are there mass-estimation evaluation artifacts?',
            'Should image-derived class AP@0.5 values from PR curves be included?',
            'Which figures should be included in the final thesis?',
        ])}
"""
    return "## Training Results Artifact Analysis\n\nSee `docs/thesis_evidence_pack/results_analysis/`.\n"


def update_evidence_pack_files(ctx: Context) -> None:
    file_names = [
        "05_training_pipeline_and_mlops.md",
        "10_challenges_failures_and_solutions.md",
        "11_project_contributions.md",
        "12_results_placeholders_and_replacement_guide.md",
        "13_prism_ai_thesis_generation_instructions.md",
        "15_code_and_artifact_traceability.md",
        "OPEN_QUESTIONS_FOR_KARIM.md",
    ]
    begin = "<!-- BEGIN TRAINING_RESULTS_ARTIFACT_ANALYSIS -->"
    end = "<!-- END TRAINING_RESULTS_ARTIFACT_ANALYSIS -->"
    for file_name in file_names:
        path = ctx.evidence_root / file_name
        if path.exists():
            current = path.read_text(encoding="utf-8")
        else:
            title = file_name.replace(".md", "").replace("_", " ").title()
            current = f"# {title}\n"
        generated = f"{begin}\n{generated_section_for(file_name, ctx).rstrip()}\n{end}\n"
        if begin in current and end in current:
            pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
            updated = pattern.sub(generated.rstrip(), current)
        else:
            updated = current.rstrip() + "\n\n" + generated
        write_text(path, updated)


def generate(ctx: Context) -> None:
    (ctx.output_root / "tables").mkdir(parents=True, exist_ok=True)
    (ctx.output_root / "figures_index").mkdir(parents=True, exist_ok=True)

    inventory = build_inventory_rows(ctx)
    curves = curve_rows(ctx)
    training_rows = training_run_summary_rows(ctx)
    distribution = dataset_distribution_rows(ctx)
    overall = overall_metrics_rows(ctx)
    classes = class_level_metrics_rows(ctx)
    checkpoints = checkpoint_rows(ctx)
    missing_metrics = missing_rows(ctx)

    write_csv(ctx.output_root / "tables" / "training_run_summary.csv", training_rows, ["item", "value", "source_file", "notes"])
    write_csv(ctx.output_root / "tables" / "dataset_distribution.csv", distribution, ["scope", "split", "source", "class_name", "images", "instances", "source_file", "notes"])
    write_csv(ctx.output_root / "tables" / "overall_metrics.csv", overall, ["metric", "value", "source_file", "epoch_or_step", "notes"])
    write_csv(ctx.output_root / "tables" / "class_level_metrics.csv", classes, ["class_name", "precision", "recall", "mAP50", "mAP50_95", "mask_mAP50", "mask_mAP50_95", "source_file", "notes"])
    write_csv(ctx.output_root / "tables" / "curve_artifacts_index.csv", curves, ["artifact_path", "artifact_type", "likely_purpose", "thesis_use", "contains_numeric_results", "notes"])
    write_csv(ctx.output_root / "tables" / "checkpoint_artifacts.csv", checkpoints, ["artifact_path", "type", "purpose", "should_use_for_thesis", "should_use_for_deployment", "notes"])
    write_csv(ctx.output_root / "tables" / "missing_metrics.csv", missing_metrics, ["item", "status", "source_or_expected_artifact", "notes"])

    write_readme(ctx)
    write_artifact_inventory(ctx, inventory)
    write_training_run_summary(ctx, training_rows)
    write_dataset_analysis(ctx)
    write_metrics_analysis(ctx, overall)
    write_validation_metrics(ctx)
    write_curves_analysis(ctx)
    write_class_analysis(ctx)
    write_error_analysis(ctx)
    write_model_selection(ctx, checkpoints)
    write_thesis_section(ctx)
    write_prism_notes(ctx)
    write_missing_questions(ctx)
    write_figure_indexes(ctx)
    update_evidence_pack_files(ctx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze installed YOLO training artifacts for thesis evidence.")
    parser.add_argument("--artifact-root", default="kaggle-output", help="Root containing installed training artifacts.")
    parser.add_argument("--evidence-root", default="docs/thesis_evidence_pack", help="Evidence-pack root to update.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    evidence_root = Path(args.evidence_root)
    if not artifact_root.exists():
        raise SystemExit(f"Artifact root not found: {artifact_root}")
    ctx = discover_context(artifact_root, evidence_root)
    generate(ctx)
    print(f"Analyzed artifact root: {artifact_root.as_posix()}")
    print(f"Wrote evidence root: {evidence_root.as_posix()}")
    print(f"Canonical run root: {ctx.run_root.relative_to(Path.cwd()).as_posix()}")
    print(f"Results CSV: {rel(ctx, ctx.results_csv_path)}")
    print(f"Training summary: {rel(ctx, ctx.training_summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
