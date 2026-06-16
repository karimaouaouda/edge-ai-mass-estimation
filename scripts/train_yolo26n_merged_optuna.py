"""Train YOLO26n-seg on the merged TACO + AquaTrash dataset with MLflow and Optuna."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import mlflow
import optuna
import yaml
from ultralytics import YOLO


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_tracking_uri(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if "://" in stripped:
        return stripped
    return Path(stripped).expanduser().resolve().as_uri()


def setup_mlflow(cfg: dict[str, Any]) -> None:
    mlflow_cfg_path = Path(cfg.get("mlflow_config", "configs/mlflow.yaml"))
    mlflow_cfg = load_yaml(mlflow_cfg_path) if mlflow_cfg_path.exists() else {}

    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or cfg.get("tracking_uri")
        or mlflow_cfg.get("tracking_uri")
        or "./mlruns"
    )
    registry_uri = (
        os.getenv("MLFLOW_REGISTRY_URI")
        or cfg.get("registry_uri")
        or mlflow_cfg.get("registry_uri")
        or tracking_uri
    )
    experiment_name = (
        os.getenv("MLFLOW_EXPERIMENT_NAME")
        or cfg.get("experiment_name")
        or mlflow_cfg.get("experiment_name")
        or "waste-seg-yolo26n-merged"
    )

    mlflow.set_tracking_uri(normalize_tracking_uri(str(tracking_uri)))
    mlflow.set_registry_uri(normalize_tracking_uri(str(registry_uri)))
    mlflow.set_experiment(str(experiment_name))


def metric_value(metrics: Any, preferred: str) -> float:
    results = getattr(metrics, "results_dict", None)
    if not isinstance(results, dict):
        if isinstance(metrics, dict):
            results = metrics
        else:
            return 0.0

    aliases = {
        "mask_map50_95": [
            "metrics/mAP50-95(M)",
            "metrics/mAP50-95(Mask)",
            "metrics/mAP50-95(B)",
        ],
        "mask_map50": [
            "metrics/mAP50(M)",
            "metrics/mAP50(Mask)",
            "metrics/mAP50(B)",
        ],
        "box_map50_95": ["metrics/mAP50-95(B)", "metrics/mAP50-95(M)"],
        "box_map50": ["metrics/mAP50(B)", "metrics/mAP50(M)"],
    }
    for key in aliases.get(preferred, [preferred]):
        if key in results:
            return float(results[key])
    for key, value in results.items():
        if preferred.lower() in str(key).lower():
            return float(value)
    return 0.0


def train_args(cfg: dict[str, Any], run_name: str, epochs: int, overrides: dict[str, Any]) -> dict[str, Any]:
    args = {
        "data": cfg["data"],
        "epochs": epochs,
        "batch": cfg.get("batch", 16),
        "imgsz": cfg.get("imgsz", 640),
        "patience": cfg.get("patience", 20),
        "device": cfg.get("device", 0),
        "project": cfg.get("project", "runs/train"),
        "name": run_name,
        "augment": cfg.get("augment", True),
        "mosaic": cfg.get("mosaic", 1.0),
        "mixup": cfg.get("mixup", 0.1),
        "copy_paste": cfg.get("copy_paste", 0.3),
        "optimizer": cfg.get("optimizer", "AdamW"),
        "lr0": cfg.get("lr0", 0.001),
        "lrf": cfg.get("lrf", 0.01),
        "weight_decay": cfg.get("weight_decay", 0.0005),
    }
    args.update(overrides)
    return args


def log_config_artifacts(config_path: Path, cfg: dict[str, Any]) -> None:
    mlflow.log_artifact(str(config_path))
    dataset_yaml = Path(cfg["data"])
    if dataset_yaml.exists():
        mlflow.log_artifact(str(dataset_yaml))
        merge_summary = dataset_yaml.parent / "merge_summary.json"
        if merge_summary.exists():
            mlflow.log_artifact(str(merge_summary))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/models/yolo26n_merged_training.yaml"))
    parser.add_argument("--skip-optuna", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    setup_mlflow(cfg)

    model_name = cfg.get("model", "yolo26n-seg.pt")
    base_run_name = cfg.get("name", "yolo26n-seg-taco-aquatrash")
    optuna_trials = int(cfg.get("optuna_trials", 10))
    tune_epochs = int(cfg.get("tune_epochs", 12))
    final_epochs = int(cfg.get("epochs", 50))
    optuna_metric = str(cfg.get("optuna_metric", "mask_map50_95"))
    optuna_direction = str(cfg.get("optuna_direction", "maximize"))

    best_params: dict[str, Any] = {}

    with mlflow.start_run(run_name=base_run_name) as parent_run:
        mlflow.log_params(
            {
                "model": model_name,
                "data": cfg["data"],
                "epochs": final_epochs,
                "batch": cfg.get("batch", 16),
                "imgsz": cfg.get("imgsz", 640),
                "optuna_trials": 0 if args.skip_optuna else optuna_trials,
                "tune_epochs": tune_epochs,
                "optuna_metric": optuna_metric,
            }
        )
        log_config_artifacts(args.config, cfg)

        if not args.skip_optuna and optuna_trials > 0:

            def objective(trial: optuna.Trial) -> float:
                trial_params = {
                    "lr0": trial.suggest_float("lr0", 1e-4, 5e-2, log=True),
                    "lrf": trial.suggest_float("lrf", 0.01, 0.5, log=True),
                    "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
                    "mosaic": trial.suggest_float("mosaic", 0.0, 1.0),
                    "mixup": trial.suggest_float("mixup", 0.0, 0.3),
                    "copy_paste": trial.suggest_float("copy_paste", 0.0, 0.5),
                }
                trial_name = f"{base_run_name}-trial-{trial.number:03d}"
                with mlflow.start_run(run_name=trial_name, nested=True):
                    mlflow.log_params(trial_params)
                    model = YOLO(model_name)
                    model.train(**train_args(cfg, run_name=trial_name, epochs=tune_epochs, overrides=trial_params))
                    metrics = model.val(data=cfg["data"], split="val", imgsz=cfg.get("imgsz", 640))
                    score = metric_value(metrics, optuna_metric)
                    mlflow.log_metric(optuna_metric, score)
                    return score

            study = optuna.create_study(direction=optuna_direction)
            study.optimize(objective, n_trials=optuna_trials)
            best_params = dict(study.best_params)
            mlflow.log_params({f"best_{key}": value for key, value in best_params.items()})
            mlflow.log_metric(f"best_{optuna_metric}", float(study.best_value))
            best_params_path = Path(cfg.get("project", "runs/train")) / base_run_name / "optuna_best_params.json"
            best_params_path.parent.mkdir(parents=True, exist_ok=True)
            best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(best_params_path))

        final_model = YOLO(model_name)
        final_model.train(**train_args(cfg, run_name=base_run_name, epochs=final_epochs, overrides=best_params))

        val_metrics = final_model.val(data=cfg["data"], split="val", imgsz=cfg.get("imgsz", 640))
        test_metrics = final_model.val(data=cfg["data"], split="test", imgsz=cfg.get("imgsz", 640))
        mlflow.log_metric("val_" + optuna_metric, metric_value(val_metrics, optuna_metric))
        mlflow.log_metric("test_" + optuna_metric, metric_value(test_metrics, optuna_metric))

        run_dir = Path(cfg.get("project", "runs/train")) / base_run_name
        weights_dir = run_dir / "weights"
        best_weights = weights_dir / "best.pt"
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")
            export_dir = Path("artifacts/exports/yolo-seg")
            export_dir.mkdir(parents=True, exist_ok=True)
            copied_weights = export_dir / f"{base_run_name}_best.pt"
            shutil.copy2(best_weights, copied_weights)
            mlflow.log_artifact(str(copied_weights), artifact_path="exports")

        summary = {
            "parent_run_id": parent_run.info.run_id,
            "model": model_name,
            "dataset": cfg["data"],
            "run_dir": str(run_dir),
            "best_params": best_params,
            "val_metric": metric_value(val_metrics, optuna_metric),
            "test_metric": metric_value(test_metrics, optuna_metric),
        }
        summary_path = run_dir / "training_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(summary_path))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
