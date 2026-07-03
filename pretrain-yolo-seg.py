"""Cloud GPU entrypoint for YOLO26m segmentation pretraining.

This script is the non-notebook counterpart of ``pretrain-yolo-seg.ipynb``.
It is intentionally focused on the first training stage:

    TACO + AquaTrash -> YOLO-ready dataset -> YOLO26m-seg pretraining

The expected cloud-server layout is:

    <repo>/
      data/input/
        aquatrash/
          Images/
          annotations.json
        taco/                  # created/resumed automatically by this script
          annotations.json
          ...

All mutable outputs are written under ``/working`` by default so the repository
checkout stays clean and the output directory can be archived after a long run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "training" / "yolo_segmentation_pretrain.yaml"
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "input"
DEFAULT_WORK_ROOT = Path("/working")
DEFAULT_RUN_STAGES = "preprocess,train,evaluate"
SKIP_OPTUNA = True


REQUIRED_PACKAGES = {
    # module name -> pip requirement.  ZenML is deliberately absent because this
    # cloud entrypoint forces the native pipeline for simpler long-running jobs.
    "yaml": "pyyaml>=6.0",
    "PIL": "pillow>=10.0",
    "ultralytics": "ultralytics>=8.4",
    "mlflow": "mlflow>=3.0",
    "onnx": "onnx>=1.16",
    "onnxruntime": "onnxruntime>=1.18",
    "matplotlib": "matplotlib>=3.7",
    "seaborn": "seaborn>=0.12",
    "pandas": "pandas>=2.0",
}


def main() -> None:
    args = parse_args()
    configure_python_path()
    install_missing_packages(skip_install=args.no_install_deps)

    # Imports happen after dependency installation and path setup so the script
    # works on a fresh cloud box with only the cloned repository present.
    from edge_ai_mass.training import TACO_ANNOTATIONS_URL, TrainingPipeline, download_taco_dataset

    input_root = args.input_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()

    print_banner(
        {
            "repo_root": str(REPO_ROOT),
            "config": str(config_path),
            "input_root": str(input_root),
            "work_root": str(work_root),
            "run_stages": args.stages,
            "skip_optuna": args.skip_optuna,
        }
    )

    work_root.mkdir(parents=True, exist_ok=True)
    input_root.mkdir(parents=True, exist_ok=True)
    configure_cuda(args.cuda_visible_devices)
    bind_aquatrash(input_root)
    install_taco(input_root, TACO_ANNOTATIONS_URL)

    overrides = build_overrides(
        input_root=input_root,
        work_root=work_root,
        args=args,
    )
    print("Pipeline overrides:")
    for override in overrides:
        print(f"  - {override}")

    pipeline = TrainingPipeline.from_config(config_path, overrides=overrides)
    plan = pipeline.plan(args.stages, skip_optuna=args.skip_optuna)
    print("Execution plan:")
    print(json.dumps(plan, indent=2, default=str)[:30000])
    fail_if_not_ready(plan)

    results = pipeline.run(args.stages, skip_optuna=args.skip_optuna, force=args.force)
    print("Pipeline results:")
    print(json.dumps(results, indent=2, default=str)[:30000])
    print_output_locations(pipeline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain YOLO26m-seg on TACO + AquaTrash using CUDA."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.getenv("TRAIN_CONFIG", DEFAULT_CONFIG)),
        help="Pretrain YAML config. Defaults to configs/training/yolo_segmentation_pretrain.yaml.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(os.getenv("INPUT_ROOT", DEFAULT_INPUT_ROOT)),
        help="Raw input root containing aquatrash/ and the auto-created taco/ directory.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(os.getenv("WORK_ROOT", DEFAULT_WORK_ROOT)),
        help="Writable output root for processed data, artifacts, MLflow, and checkpoints.",
    )
    parser.add_argument(
        "--stages",
        default=os.getenv("RUN_STAGES", DEFAULT_RUN_STAGES),
        help="Pipeline stages to run. Default: preprocess,train,evaluate.",
    )
    parser.add_argument(
        "--skip-optuna",
        action=argparse.BooleanOptionalAction,
        default=parse_bool(os.getenv("SKIP_OPTUNA"), default=SKIP_OPTUNA),
        help="Skip the Optuna tune stage. Defaults to true for the pretrain run.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.getenv("CUDA_VISIBLE_DEVICES", "0"),
        help="CUDA_VISIBLE_DEVICES value to set before importing/using torch.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=optional_int(os.getenv("TRAINING_EPOCHS")),
        help="Override training.epochs. Leave unset to use the YAML value.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=optional_int(os.getenv("TRAINING_BATCH")),
        help="Override training.batch. Leave unset to use the YAML value.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=optional_int(os.getenv("TRAINING_IMGSZ")),
        help="Override training.imgsz and evaluation.imgsz. Leave unset to use YAML values.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=optional_int(os.getenv("TRAINING_WORKERS")),
        help="Override training.workers. Leave unset to use the YAML value.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=optional_int(os.getenv("TRAIN_CHECKPOINT_INTERVAL")),
        help="Save managed checkpoints every N epochs.",
    )
    parser.add_argument(
        "--resume-mode",
        default=os.getenv("TRAIN_RESUME_MODE"),
        choices=(None, "auto", "never", "required"),
        help="Override training.checkpointing.resume.mode.",
    )
    parser.add_argument(
        "--resume-epoch",
        type=int,
        default=optional_int(os.getenv("TRAINING_RESUME_EPOCH")),
        help="Resume from a managed checkpoint epoch, e.g. 55.",
    )
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=optional_int(os.getenv("TRAIN_ADDITIONAL_EPOCHS")),
        help="Train N more epochs from the resolved checkpoint.",
    )
    parser.add_argument(
        "--run-name",
        default=os.getenv("TRAINING_RUN_NAME", "waste-seg-yolo-pretrain"),
        help="Run/artifact name under /working/artifacts/training/yolo.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional TrainingConfig override. Repeatable.",
    )
    parser.add_argument(
        "--no-install-deps",
        action="store_true",
        help="Do not pip-install missing packages before running.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=parse_bool(os.getenv("TRAIN_FORCE"), default=True),
        help="Allow reruns under the same run name. Defaults to true for resumable cloud runs.",
    )
    return parser.parse_args()


def configure_python_path() -> None:
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def install_missing_packages(*, skip_install: bool) -> None:
    missing = [
        requirement
        for module, requirement in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        print("All required Python packages are already importable.")
        return
    if skip_install:
        raise RuntimeError(
            "Missing Python packages and --no-install-deps was set: "
            + ", ".join(missing)
        )
    print("Installing missing Python packages:", missing)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *missing]
    )


def configure_cuda(cuda_visible_devices: str) -> None:
    # Set allocator and visible devices before torch is imported by the pipeline.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    import torch

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("CUDA available =", torch.cuda.is_available())
    print("CUDA device count =", torch.cuda.device_count())
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "CUDA is required for this cloud pretraining script, but no CUDA GPU "
            "is visible. Check the cloud instance GPU/runtime and CUDA_VISIBLE_DEVICES."
        )
    for index in range(torch.cuda.device_count()):
        print(f"CUDA device [{index}]: {torch.cuda.get_device_name(index)}")
    torch.cuda.empty_cache()


def bind_aquatrash(input_root: Path) -> None:
    aquatrash_root = Path(
        os.getenv("AQUATRASH_ROOT", input_root / "aquatrash")
    ).expanduser().resolve()
    annotations = Path(
        os.getenv("AQUATRASH_ANNOTATIONS", aquatrash_root / "annotations.json")
    ).expanduser().resolve()
    images = Path(
        os.getenv("AQUATRASH_IMAGES", aquatrash_root / "Images")
    ).expanduser().resolve()

    if not annotations.is_file():
        raise FileNotFoundError(
            f"AquaTrash annotations not found: {annotations}. Expected "
            f"{input_root / 'aquatrash' / 'annotations.json'} or set AQUATRASH_ANNOTATIONS."
        )
    if not images.is_dir():
        raise FileNotFoundError(
            f"AquaTrash Images directory not found: {images}. Expected "
            f"{input_root / 'aquatrash' / 'Images'} or set AQUATRASH_IMAGES."
        )

    os.environ["AQUATRASH_ANNOTATIONS"] = str(annotations)
    os.environ["AQUATRASH_IMAGES"] = str(images)
    print({"aquatrash_annotations": str(annotations), "aquatrash_images": str(images)})


def install_taco(input_root: Path, default_annotations_url: str) -> None:
    from edge_ai_mass.training import download_taco_dataset

    taco_root = Path(os.getenv("TACO_ROOT", input_root / "taco")).expanduser().resolve()
    summary = download_taco_dataset(
        taco_root,
        annotations_url=os.getenv("TACO_ANNOTATIONS_URL", default_annotations_url),
        max_workers=int(os.getenv("TACO_DOWNLOAD_WORKERS", "8")),
        timeout_seconds=float(os.getenv("TACO_DOWNLOAD_TIMEOUT", "30")),
        retries=int(os.getenv("TACO_DOWNLOAD_RETRIES", "3")),
        max_failures=int(os.getenv("TACO_MAX_DOWNLOAD_FAILURES", "0")),
    )
    os.environ["TACO_ANNOTATIONS"] = str(taco_root / "annotations.json")
    os.environ["TACO_IMAGES"] = str(taco_root)
    print("TACO ready:", json.dumps(summary, indent=2)[:4000])


def build_overrides(
    *,
    input_root: Path,
    work_root: Path,
    args: argparse.Namespace,
) -> list[str]:
    processed_dir = work_root / "data" / "processed" / "waste_seg_yolo_pretrain"
    artifacts_dir = work_root / "artifacts" / "training" / "yolo"
    mlflow_db = work_root / "artifacts" / "mlflow" / "mlflow.db"
    optuna_db = work_root / "artifacts" / "optuna" / "yolo_pretrain.db"

    overrides = [
        f"project.root={work_root.as_posix()}",
        "orchestration.zenml.enabled=false",
        f"data.output_dir={processed_dir.as_posix()}",
        "data.materialize=copy",
        f"training.run_name={args.run_name}",
        f"training.artifacts_dir={artifacts_dir.as_posix()}",
        "training.device=0",
        "training.amp=true",
        "evaluation.device=0",
        "export.options.common.device=0",
        "tuning.enabled=false",
        f"tuning.storage=sqlite:///{optuna_db.as_posix()}",
        f"tracking.uri=sqlite:///{mlflow_db.as_posix()}",
        f"tracking.registry_uri=sqlite:///{mlflow_db.as_posix()}",
        "tracking.registry.enabled=false",
        "publication.enabled=false",
    ]

    optional_overrides = {
        "training.epochs": args.epochs,
        "training.batch": args.batch,
        "training.imgsz": args.imgsz,
        "evaluation.imgsz": args.imgsz,
        "training.workers": args.workers,
        "training.checkpointing.interval_epochs": args.checkpoint_interval,
        "training.checkpointing.resume.mode": args.resume_mode,
        "training.checkpointing.resume.selected_epoch": args.resume_epoch,
        "training.checkpointing.resume.additional_epochs": args.additional_epochs,
    }
    for key, value in optional_overrides.items():
        if value is not None:
            overrides.append(f"{key}={value}")

    overrides.extend(args.extra_override)
    return overrides


def fail_if_not_ready(plan: dict[str, Any]) -> None:
    if plan.get("ready"):
        return
    issues = "\n".join(f"  - {item}" for item in plan.get("blocking_issues", []))
    raise RuntimeError(f"Training plan is not ready:\n{issues}")


def print_output_locations(pipeline: Any) -> None:
    state_path = pipeline.config.artifacts_dir / "pipeline_state.json"
    print("Output locations:")
    print(f"  dataset manifest: {pipeline.config.dataset_dir / 'dataset_manifest.json'}")
    print(f"  artifacts:        {pipeline.config.artifacts_dir}")
    print(f"  pipeline state:   {state_path}")


def print_banner(payload: dict[str, Any]) -> None:
    print("=" * 80)
    print("YOLO26m segmentation pretraining")
    print(json.dumps(payload, indent=2, default=str))
    print("=" * 80)


def parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


if __name__ == "__main__":
    main()
