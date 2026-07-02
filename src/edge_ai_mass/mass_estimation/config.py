"""Configuration loading and validation for mass-estimation MLOps stages."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from edge_ai_mass.utils.config import load_config


MASS_EXECUTION_STAGES = (
    "preprocess",
    "features",
    "split",
    "train",
    "evaluate",
    "register",
)
FULL_PIPELINE_ALIASES = {"all", "mass", "mass-estimation", "mass_estimation"}


class MassEstimationConfigError(ValueError):
    """Raised when a mass-estimation config is incomplete or unsafe."""


class MassEstimationConfig:
    """Validated config wrapper with the same path and override semantics as YOLO."""

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
    ) -> "MassEstimationConfig":
        payload = load_config(path)
        if not isinstance(payload, dict):
            raise MassEstimationConfigError(f"Config must contain a YAML mapping: {path}")
        payload = copy.deepcopy(payload)
        for expression in overrides or []:
            _apply_override(payload, expression)
        return cls(payload, path)

    def _validate(self) -> None:
        required_sections = (
            "project",
            "data",
            "features",
            "split",
            "model",
            "training",
            "evaluation",
            "tracking",
        )
        missing = [
            name for name in required_sections if not isinstance(self.payload.get(name), dict)
        ]
        if missing:
            raise MassEstimationConfigError(
                f"Missing configuration sections: {', '.join(missing)}"
            )

        project = self.payload["project"]
        if project.get("pipeline") != "mass_estimation":
            raise MassEstimationConfigError("project.pipeline must be 'mass_estimation'")
        if project.get("model_stage", "mass") != "mass":
            raise MassEstimationConfigError("project.model_stage must be 'mass'")

        data = self.payload["data"]
        if not data.get("measurements"):
            raise MassEstimationConfigError("data.measurements is required")
        columns = data.get("columns", {})
        if not isinstance(columns, dict):
            raise MassEstimationConfigError("data.columns must be a mapping")
        for key in ("sample_id", "class_name", "real_mass_g"):
            if not columns.get(key, key):
                raise MassEstimationConfigError(f"data.columns.{key} cannot be empty")

        split = self.payload["split"]
        ratios = [float(split.get(name, 0.0)) for name in ("train", "val", "test")]
        if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
            raise MassEstimationConfigError(
                "split train/val/test values must be non-negative and sum to 1"
            )
        if float(split.get("train", 0.0)) <= 0:
            raise MassEstimationConfigError("split.train must be greater than zero")

        model = self.payload["model"]
        candidates = model.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise MassEstimationConfigError("model.candidates must be a non-empty list")
        names: set[str] = set()
        supported = {
            "physics_only",
            "ridge",
            "huber",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "hist_gradient_boosting",
            "xgboost",
            "lightgbm",
            "mlp",
        }
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise MassEstimationConfigError(f"model.candidates[{index}] must be a mapping")
            name = str(candidate.get("name", "")).strip()
            kind = str(candidate.get("type", "")).strip()
            if not name:
                raise MassEstimationConfigError(f"model.candidates[{index}].name is required")
            if name in names:
                raise MassEstimationConfigError(f"Duplicate model candidate name: {name}")
            names.add(name)
            if kind not in supported:
                raise MassEstimationConfigError(
                    f"Unsupported model candidate type '{kind}'. Supported: {sorted(supported)}"
                )

        features = self.payload["features"]
        for key in ("numeric_columns", "categorical_columns", "exclude_columns"):
            value = features.get(key, [])
            if value is not None and not isinstance(value, list):
                raise MassEstimationConfigError(f"features.{key} must be a list")

        tracking = self.payload["tracking"]
        if not isinstance(tracking.get("registry", {}), dict):
            raise MassEstimationConfigError("tracking.registry must be a mapping")

        training = self.payload["training"]
        mode = str(training.get("mode", "compare_candidates"))
        allowed_training_modes = {
            "compare",
            "compare_candidates",
            "best_of_candidates",
            "best",
            "single",
            "one_model",
        }
        if mode not in allowed_training_modes:
            raise MassEstimationConfigError(
                "training.mode must be compare_candidates or single"
            )

        zenml = self.payload.get("orchestration", {}).get("zenml", {})
        if zenml and not isinstance(zenml, dict):
            raise MassEstimationConfigError("orchestration.zenml must be a mapping")

    def path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    @property
    def run_name(self) -> str:
        return str(self.payload["training"].get("run_name", "mass-residual-regression"))

    @property
    def processed_dir(self) -> Path:
        return self.path(
            self.payload["data"].get("output_dir", "data/processed/mass_estimation")
        )

    @property
    def artifacts_dir(self) -> Path:
        base = self.payload["training"].get(
            "artifacts_dir", "artifacts/mass_estimation"
        )
        return self.path(base) / self.run_name

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def data_digest(self) -> str:
        payload = {
            "data": self.payload["data"],
            "features": self.payload["features"],
            "split": self.payload["split"],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def resolved_copy(self) -> dict[str, Any]:
        result = copy.deepcopy(self.payload)
        result.setdefault("runtime", {})
        result["runtime"].update(
            {
                "config_path": str(self.source_path),
                "project_root": str(self.project_root),
                "processed_dir": str(self.processed_dir),
                "artifacts_dir": str(self.artifacts_dir),
                "config_digest": self.digest,
                "data_config_digest": self.data_digest,
            }
        )
        return result


def normalize_mass_stages(stage: str) -> list[str]:
    value = stage.strip().lower()
    if value in FULL_PIPELINE_ALIASES:
        return list(MASS_EXECUTION_STAGES)
    stages = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in stages if item not in MASS_EXECUTION_STAGES]
    if invalid:
        valid = ", ".join((*MASS_EXECUTION_STAGES, *sorted(FULL_PIPELINE_ALIASES)))
        raise MassEstimationConfigError(
            f"Unknown mass-estimation stage(s) {invalid}. Valid values: {valid}"
        )
    return stages


def _apply_override(payload: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise MassEstimationConfigError(
            f"Override must use dotted.path=value syntax: {expression}"
        )
    dotted_key, raw_value = expression.split("=", 1)
    keys = [key for key in dotted_key.strip().split(".") if key]
    if not keys:
        raise MassEstimationConfigError(f"Override has no key: {expression}")
    value = yaml.safe_load(raw_value)
    cursor: dict[str, Any] = payload
    for key in keys[:-1]:
        existing = cursor.get(key)
        if existing is None:
            existing = {}
            cursor[key] = existing
        if not isinstance(existing, dict):
            raise MassEstimationConfigError(f"Cannot descend into non-mapping key: {key}")
        cursor = existing
    cursor[keys[-1]] = value
