"""Build YOLO dataset YAML, train with Optuna, and evaluate the best model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_tracking_uri(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if "://" in stripped:
        return stripped
    return Path(stripped).expanduser().resolve().as_uri()


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    normalized_uri = normalize_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or tracking_uri)
    mlflow.set_tracking_uri(normalized_uri)
    mlflow.set_registry_uri(normalize_tracking_uri(os.getenv("MLFLOW_REGISTRY_URI") or tracking_uri))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME") or experiment_name)


def load_categories(categories_json: Path) -> list[dict[str, Any]]:
    categories = load_json(categories_json).get("categories", [])
    if not categories:
        raise ValueError(f"No categories found in '{categories_json}'.")
    return sorted(categories, key=lambda item: int(item["id"]))


def read_label_ids(label_paths: list[Path]) -> set[int]:
    label_ids: set[int] = set()
    for label_path in label_paths:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            label_ids.add(int(float(parts[0])))
    return label_ids


def infer_label_id_mode(label_ids: set[int], category_ids: set[int], zero_based_ids: set[int]) -> str:
    if not label_ids:
        return "zero_based"
    if label_ids <= zero_based_ids and 0 in label_ids:
        return "zero_based"
    if label_ids <= category_ids:
        return "category_ids"
    if label_ids <= zero_based_ids:
        return "zero_based"
    raise ValueError(f"Label IDs {sorted(label_ids)} do not match category IDs or zero-based YOLO IDs.")


def normalize_label_files(yolo_ready_dir: Path, categories: list[dict[str, Any]], mode: str = "auto") -> dict[str, Any]:
    label_paths = [
        path
        for split in SPLITS
        for path in (yolo_ready_dir / "labels" / split).rglob("*.txt")
    ]
    original_id_to_yolo_id = {
        int(category["id"]): index
        for index, category in enumerate(categories)
    }
    category_ids = set(original_id_to_yolo_id)
    zero_based_ids = set(range(len(categories)))
    label_ids = read_label_ids(label_paths)

    if mode == "auto":
        mode = infer_label_id_mode(label_ids, category_ids, zero_based_ids)
    if mode not in {"category_ids", "zero_based"}:
        raise ValueError("label ID mode must be 'auto', 'category_ids', or 'zero_based'.")

    rewritten = 0
    for label_path in label_paths:
        changed = False
        output_lines: list[str] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue

            raw_class_id = int(float(parts[0]))
            if mode == "category_ids":
                if raw_class_id not in original_id_to_yolo_id:
                    raise ValueError(f"Unknown category ID {raw_class_id} in '{label_path}'.")
                class_id = original_id_to_yolo_id[raw_class_id]
            else:
                class_id = raw_class_id

            if class_id not in zero_based_ids:
                raise ValueError(f"YOLO class ID {class_id} in '{label_path}' is outside 0..{len(categories) - 1}.")

            new_line = " ".join([str(class_id), *parts[1:]])
            output_lines.append(new_line)
            changed = changed or new_line != line.strip()

        if changed:
            label_path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
            rewritten += 1

    return {
        "label_id_mode": mode,
        "labels_seen": len(label_paths),
        "rewritten_label_files": rewritten,
        "original_label_ids": sorted(label_ids),
    }


def split_counts(yolo_ready_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        image_dir = yolo_ready_dir / "images" / split
        label_dir = yolo_ready_dir / "labels" / split
        counts[split] = {
            "images": sum(1 for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            "labels": sum(1 for path in label_dir.rglob("*.txt") if path.is_file()),
        }
    return counts


def build_dataset_yaml(
    yolo_ready_dir: Path,
    categories_json: Path,
    output_yaml: Path | None = None,
    label_id_mode: str = "auto",
) -> tuple[Path, dict[str, Any]]:
    yolo_ready_dir = yolo_ready_dir.resolve()
    categories = load_categories(categories_json)
    names = [str(category["name"]) for category in categories]

    for split in SPLITS:
        if not (yolo_ready_dir / "images" / split).exists():
            raise FileNotFoundError(f"Missing images/{split} in '{yolo_ready_dir}'.")
        if not (yolo_ready_dir / "labels" / split).exists():
            raise FileNotFoundError(f"Missing labels/{split} in '{yolo_ready_dir}'.")

    label_summary = normalize_label_files(yolo_ready_dir, categories, mode=label_id_mode)
    output_yaml = (output_yaml or (yolo_ready_dir / "dataset.yaml")).resolve()

    yaml_lines = [
        f"path: {yolo_ready_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    output_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    summary = {
        "dataset_yaml": str(output_yaml),
        "yolo_ready_dir": str(yolo_ready_dir),
        "categories_json": str(categories_json.resolve()),
        "classes": names,
        "splits": split_counts(yolo_ready_dir),
        **label_summary,
    }
    summary_path = yolo_ready_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_yaml, summary


def metric_results(metrics: Any) -> dict[str, float]:
    results = getattr(metrics, "results_dict", None)
    if not isinstance(results, dict):
        return {}
    clean: dict[str, float] = {}
    for key, value in results.items():
        try:
            clean[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return clean


def metric_value(metrics: Any, preferred: str) -> float:
    results = metric_results(metrics)
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
            return results[key]
    for key, value in results.items():
        if preferred.lower() in key.lower():
            return value
    return 0.0


def safe_metric_name(name: str) -> str:
    return name.replace("/", "_").replace("(", "").replace(")", "").replace(" ", "_")


def log_metric_dict(prefix: str, metrics: Any) -> None:
    for key, value in metric_results(metrics).items():
        mlflow.log_metric(f"{prefix}_{safe_metric_name(key)}", value)


def train_args(args: argparse.Namespace, dataset_yaml: Path, run_name: str, epochs: int, overrides: dict[str, Any]) -> dict[str, Any]:
    train_kwargs = {
        "data": str(dataset_yaml),
        "epochs": epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "patience": args.patience,
        "device": args.device,
        "project": str(args.project),
        "name": run_name,
        "exist_ok": True,
        "augment": args.augment,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
    }
    train_kwargs.update(overrides)
    return train_kwargs


def train_with_optuna(args: argparse.Namespace, dataset_yaml: Path, dataset_summary: dict[str, Any]) -> dict[str, Any]:
    global mlflow, optuna, YOLO
    import mlflow
    import optuna
    from ultralytics import YOLO

    setup_mlflow(args.mlflow_tracking_uri, args.experiment_name)

    best_params: dict[str, Any] = {}
    with mlflow.start_run(run_name=args.run_name) as parent_run:
        mlflow.log_params(
            {
                "model": args.model,
                "dataset_yaml": str(dataset_yaml),
                "epochs": args.epochs,
                "batch": args.batch,
                "imgsz": args.imgsz,
                "optuna_trials": 0 if args.skip_optuna else args.optuna_trials,
                "tune_epochs": args.tune_epochs,
                "optuna_metric": args.optuna_metric,
            }
        )
        mlflow.log_artifact(str(dataset_yaml))
        summary_path = Path(dataset_summary["yolo_ready_dir"]) / "dataset_summary.json"
        if summary_path.exists():
            mlflow.log_artifact(str(summary_path))

        if not args.skip_optuna and args.optuna_trials > 0:

            def objective(trial: optuna.Trial) -> float:
                trial_params = {
                    "lr0": trial.suggest_float("lr0", 1e-4, 5e-2, log=True),
                    "lrf": trial.suggest_float("lrf", 0.01, 0.5, log=True),
                    "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
                    "mosaic": trial.suggest_float("mosaic", 0.0, 1.0),
                    "mixup": trial.suggest_float("mixup", 0.0, 0.3),
                    "copy_paste": trial.suggest_float("copy_paste", 0.0, 0.5),
                }
                trial_name = f"{args.run_name}-trial-{trial.number:03d}"
                with mlflow.start_run(run_name=trial_name, nested=True):
                    mlflow.log_params(trial_params)
                    model = YOLO(args.model)
                    model.train(**train_args(args, dataset_yaml, trial_name, args.tune_epochs, trial_params))
                    val_metrics = model.val(data=str(dataset_yaml), split="val", imgsz=args.imgsz)
                    score = metric_value(val_metrics, args.optuna_metric)
                    mlflow.log_metric(args.optuna_metric, score)
                    log_metric_dict("trial_val", val_metrics)
                    return score

            study = optuna.create_study(direction=args.optuna_direction)
            study.optimize(objective, n_trials=args.optuna_trials)
            best_params = dict(study.best_params)
            best_params_path = args.project / args.run_name / "optuna_best_params.json"
            best_params_path.parent.mkdir(parents=True, exist_ok=True)
            best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
            mlflow.log_params({f"best_{key}": value for key, value in best_params.items()})
            mlflow.log_metric(f"best_{args.optuna_metric}", float(study.best_value))
            mlflow.log_artifact(str(best_params_path))

        final_model = YOLO(args.model)
        train_result = final_model.train(**train_args(args, dataset_yaml, args.run_name, args.epochs, best_params))
        run_dir = Path(getattr(train_result, "save_dir", None) or (args.project / args.run_name))
        best_weights = run_dir / "weights" / "best.pt"
        if not best_weights.exists():
            best_weights = run_dir / "weights" / "last.pt"
        if not best_weights.exists():
            raise FileNotFoundError(f"Could not find trained weights under '{run_dir / 'weights'}'.")

        best_model = YOLO(str(best_weights))
        val_metrics = best_model.val(data=str(dataset_yaml), split="val", imgsz=args.imgsz)
        test_metrics = best_model.val(data=str(dataset_yaml), split="test", imgsz=args.imgsz)
        mlflow.log_metric("val_" + args.optuna_metric, metric_value(val_metrics, args.optuna_metric))
        mlflow.log_metric("test_" + args.optuna_metric, metric_value(test_metrics, args.optuna_metric))
        log_metric_dict("val", val_metrics)
        log_metric_dict("test", test_metrics)

        artifacts_dir = args.artifacts_dir
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        copied_weights = artifacts_dir / f"{args.run_name}_best.pt"
        shutil.copy2(best_weights, copied_weights)
        mlflow.log_artifact(str(best_weights), artifact_path="weights")
        mlflow.log_artifact(str(copied_weights), artifact_path="exports")

        summary = {
            "parent_run_id": parent_run.info.run_id,
            "model": args.model,
            "dataset_yaml": str(dataset_yaml),
            "run_dir": str(run_dir),
            "best_weights": str(best_weights),
            "exported_weights": str(copied_weights),
            "best_params": best_params,
            "val_metrics": metric_results(val_metrics),
            "test_metrics": metric_results(test_metrics),
            "val_metric": metric_value(val_metrics, args.optuna_metric),
            "test_metric": metric_value(test_metrics, args.optuna_metric),
        }
        summary_path = run_dir / "training_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(summary_path))
        print(json.dumps(summary, indent=2))
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-ready-dir", type=Path, default=Path("data/yolo_ready"))
    parser.add_argument("--categories-json", type=Path, default=Path("data/raw/aquatrash/annotations.json"))
    parser.add_argument("--dataset-yaml", type=Path, default=None)
    parser.add_argument("--label-id-mode", choices=["auto", "category_ids", "zero_based"], default="auto")
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--tune-epochs", type=int, default=12)
    parser.add_argument("--optuna-trials", type=int, default=10)
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument("--optuna-direction", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--optuna-metric", default="mask_map50_95")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.1)
    parser.add_argument("--copy-paste", type=float, default=0.3)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--run-name", default="yolo26n-seg-yolo-ready")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/exports/yolo-ready"))
    parser.add_argument("--mlflow-tracking-uri", default="./mlruns")
    parser.add_argument("--experiment-name", default="waste-seg-yolo-ready")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_yaml, dataset_summary = build_dataset_yaml(
        yolo_ready_dir=args.yolo_ready_dir,
        categories_json=args.categories_json,
        output_yaml=args.dataset_yaml,
        label_id_mode=args.label_id_mode,
    )
    train_with_optuna(args, dataset_yaml, dataset_summary)


if __name__ == "__main__":
    main()
