"""Configuration loading and validation for training pipelines."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from edge_ai_mass.utils.config import load_config


EXECUTION_STAGES = ("preprocess", "tune", "train", "evaluate", "export", "register")
FULL_PIPELINE_ALIASES = {"all", "detection", "yolo", "yolo-segmentation"}


class TrainingConfigError(ValueError):
    """Raised when a training config is incomplete or internally inconsistent."""


class TrainingConfig:
    """Validated config wrapper with predictable path and override semantics."""

    def __init__(self, payload: dict[str, Any], source_path: str | Path):
        self.payload = payload
        self.source_path = Path(source_path).expanduser().resolve()
        root_value = payload.get("project", {}).get("root", ".")
        self.project_root = Path(root_value).expanduser().resolve()
        self._validate()

    @classmethod
    def load(
        cls,
        path: str | Path,
        overrides: list[str] | None = None,
    ) -> "TrainingConfig":
        payload = load_config(path)
        if not isinstance(payload, dict):
            raise TrainingConfigError(f"Training config must contain a YAML mapping: {path}")
        payload = copy.deepcopy(payload)
        for expression in overrides or []:
            _apply_override(payload, expression)
        return cls(payload, path)

    def _validate(self) -> None:
        required_sections = ("project", "data", "model", "training", "tracking")
        missing = [
            name for name in required_sections if not isinstance(self.payload.get(name), dict)
        ]
        if missing:
            raise TrainingConfigError(f"Missing configuration sections: {', '.join(missing)}")

        project = self.payload["project"]
        if project.get("pipeline") != "yolo":
            raise TrainingConfigError("project.pipeline must be 'yolo' for the implemented trainer")
        if project.get("model_stage", "detection") != "detection":
            raise TrainingConfigError("Only the detection model stage is implemented for now")

        data = self.payload["data"]
        sources = data.get("sources")
        if not isinstance(sources, list) or not sources:
            raise TrainingConfigError("data.sources must be a non-empty list")
        names: set[str] = set()
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                raise TrainingConfigError(f"data.sources[{index}] must be a mapping")
            name = str(source.get("name", "")).strip().lower()
            if not name:
                raise TrainingConfigError(f"data.sources[{index}].name is required")
            if name in names:
                raise TrainingConfigError(f"Duplicate dataset source name: {name}")
            names.add(name)
            for key in ("annotations", "images"):
                if not source.get(key):
                    raise TrainingConfigError(f"data source '{name}' requires '{key}'")

        split = data.get("split", {})
        defaults = (("train", 0.7), ("val", 0.15), ("test", 0.15))
        ratios = [float(split.get(name, default)) for name, default in defaults]
        if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
            raise TrainingConfigError(
                "data.split train/val/test values must be non-negative and sum to 1"
            )

        classes = data.get("classes")
        if not isinstance(classes, list) or not classes or len(set(classes)) != len(classes):
            raise TrainingConfigError("data.classes must be a non-empty list of unique labels")

        model = self.payload["model"]
        if not model.get("checkpoint"):
            raise TrainingConfigError(
                "model.checkpoint is required (local path or Ultralytics model name)"
            )
        if model.get("task", "segment") not in {"segment", "detect"}:
            raise TrainingConfigError("model.task must be 'segment' or 'detect'")

        tuning = self.payload.get("tuning", {})
        if tuning.get("enabled", True) and not isinstance(tuning.get("search_space", {}), dict):
            raise TrainingConfigError("tuning.search_space must be a mapping")
        if tuning.get("enabled", True) and tuning.get("enforce_complete_space", True):
            training = self.payload["training"]
            optimizable = {
                "batch",
                "imgsz",
                "patience",
                "optimizer",
                *training.get("hyperparameters", {}).keys(),
                *training.get("augmentation", {}).keys(),
                *training.get("extra_args", {}).keys(),
            }
            missing_from_space = sorted(optimizable - set(tuning.get("search_space", {})))
            if missing_from_space:
                raise TrainingConfigError(
                    "Every configured model-quality parameter must be represented in "
                    f"tuning.search_space; missing: {missing_from_space}. Set "
                    "tuning.enforce_complete_space=false only for an intentional fixed ablation."
                )

        export = self.payload.get("export", {})
        if export.get("enabled", False):
            formats = export.get("formats", [])
            if not isinstance(formats, list) or not formats:
                raise TrainingConfigError(
                    "export.formats must select at least one format when export.enabled is true"
                )
            supported = {
                "torchscript",
                "onnx",
                "openvino",
                "engine",
                "coreml",
                "saved_model",
                "pb",
                "tflite",
                "edgetpu",
                "tfjs",
                "paddle",
                "mnn",
                "ncnn",
                "imx",
                "rknn",
                "executorch",
                "axelera",
            }
            invalid = sorted(set(formats) - supported)
            if invalid:
                raise TrainingConfigError(f"Unsupported export format(s): {invalid}")
            if len(set(formats)) != len(formats):
                raise TrainingConfigError("export.formats must not contain duplicates")

    def path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    @property
    def dataset_dir(self) -> Path:
        return self.path(self.payload["data"].get("output_dir", "data/processed/waste_seg_yolo"))

    @property
    def run_name(self) -> str:
        return str(self.payload["training"].get("run_name", "waste-seg-yolo"))

    @property
    def artifacts_dir(self) -> Path:
        base = self.payload["training"].get("artifacts_dir", "artifacts/training/yolo")
        return self.path(base) / self.run_name

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def data_digest(self) -> str:
        data = copy.deepcopy(self.payload["data"])
        data.pop("output_dir", None)
        data.pop("materialize", None)
        data.pop("visualization", None)
        for source in data.get("sources", []):
            source.pop("annotations", None)
            source.pop("images", None)
        payload = {
            "data": data,
            "task": self.payload["model"].get("task", "segment"),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def optimization_digest(self) -> str:
        tuning = self.payload.get("tuning", {})
        payload = {
            "data_config_digest": self.data_digest,
            "model": self.payload["model"],
            "training": self.payload["training"],
            "tuning": {
                key: value
                for key, value in tuning.items()
                if key
                not in {
                    "storage",
                    "trials",
                    "timeout_seconds",
                    "show_progress_bar",
                    "study_name",
                    "version_by_config",
                }
            },
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def resolved_copy(self) -> dict[str, Any]:
        """Return a serializable copy containing resolved operational paths."""
        result = copy.deepcopy(self.payload)
        result.setdefault("runtime", {})
        result["runtime"].update(
            {
                "config_path": str(self.source_path),
                "project_root": str(self.project_root),
                "dataset_dir": str(self.dataset_dir),
                "artifacts_dir": str(self.artifacts_dir),
                "config_digest": self.digest,
                "data_config_digest": self.data_digest,
                "optimization_digest": self.optimization_digest,
            }
        )
        return result


def normalize_stages(stage: str, *, skip_optuna: bool = False) -> list[str]:
    value = stage.strip().lower()
    if value in FULL_PIPELINE_ALIASES:
        stages = list(EXECUTION_STAGES)
    else:
        stages = [item.strip().lower() for item in value.split(",") if item.strip()]
        invalid = [item for item in stages if item not in EXECUTION_STAGES]
        if invalid:
            valid = ", ".join((*EXECUTION_STAGES, *sorted(FULL_PIPELINE_ALIASES)))
            raise TrainingConfigError(f"Unknown training stage(s) {invalid}. Valid values: {valid}")
    if skip_optuna:
        stages = [item for item in stages if item != "tune"]
    return stages


def _apply_override(payload: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise TrainingConfigError(f"Override must use dotted.path=value syntax: {expression}")
    dotted_key, raw_value = expression.split("=", 1)
    keys = [key for key in dotted_key.strip().split(".") if key]
    if not keys:
        raise TrainingConfigError(f"Override has no key: {expression}")
    value = yaml.safe_load(raw_value)
    cursor: dict[str, Any] = payload
    for key in keys[:-1]:
        existing = cursor.get(key)
        if existing is None:
            existing = {}
            cursor[key] = existing
        if not isinstance(existing, dict):
            raise TrainingConfigError(f"Cannot descend into non-mapping override key: {key}")
        cursor = existing
    cursor[keys[-1]] = value
