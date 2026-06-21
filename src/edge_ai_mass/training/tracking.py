"""Lazy MLflow integration used by training and model registration."""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from edge_ai_mass.training.config import TrainingConfig
from edge_ai_mass.training.state import PipelineState, git_metadata


def require_mlflow():
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            'MLflow is required for this stage. Install with: pip install -e ".[mlops]"'
        ) from exc
    return mlflow


def normalize_uri(value: str, project_root: Path) -> str:
    stripped = value.strip()
    if stripped.startswith("sqlite:///"):
        database = Path(stripped.removeprefix("sqlite:///")).expanduser()
        if not database.is_absolute():
            database = project_root / database
        database.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{database.resolve().as_posix()}"
    if "://" in stripped or stripped.startswith("databricks"):
        return stripped
    path = Path(stripped).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve().as_uri()


class MLflowSession:
    def __init__(self, config: TrainingConfig, state: PipelineState):
        self.config = config
        self.state = state
        self.mlflow = require_mlflow()
        tracking = config.payload["tracking"]
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or str(tracking.get("uri", "./mlruns"))
        registry_uri = os.getenv("MLFLOW_REGISTRY_URI") or str(
            tracking.get("registry_uri", tracking_uri)
        )
        self.mlflow.set_tracking_uri(normalize_uri(tracking_uri, config.project_root))
        self.mlflow.set_registry_uri(normalize_uri(registry_uri, config.project_root))
        experiment = os.getenv("MLFLOW_EXPERIMENT_NAME") or str(
            tracking.get("experiment", "edge-ai-mass-yolo")
        )
        self.mlflow.set_experiment(experiment)

    @contextlib.contextmanager
    def run(self) -> Iterator[Any]:
        run_id = self.state.data.get("mlflow_run_id")
        kwargs = {"run_id": run_id} if run_id else {"run_name": self.config.run_name}
        with self.mlflow.start_run(**kwargs) as run:
            if not run_id:
                self.state.update("tracking", mlflow_run_id=run.info.run_id)
                self._log_initial_context()
            yield run

    def _log_initial_context(self) -> None:
        mlflow = self.mlflow
        project = self.config.payload["project"]
        tags = {
            "project": str(project.get("name", "edge-ai-mass-estimation")),
            "model_stage": str(project.get("model_stage", "detection")),
            "pipeline": "yolo",
            "config_digest": self.config.digest,
            "optimization_digest": self.config.optimization_digest,
            **{
                str(key): str(value)
                for key, value in self.config.payload["tracking"].get("tags", {}).items()
            },
        }
        mlflow.set_tags(tags)
        mlflow.log_dict(self.config.resolved_copy(), "configuration/resolved_config.json")
        mlflow.log_dict(git_metadata(self.config.project_root), "lineage/git.json")
        mlflow.log_param("checkpoint", str(self.config.payload["model"]["checkpoint"]))
        mlflow.log_param("task", str(self.config.payload["model"]["task"]))
        mlflow.log_param("config_digest", self.config.digest)

    def log_dataset(self, manifest: dict[str, Any]) -> None:
        self.mlflow.log_dict(manifest, "lineage/dataset_manifest.json")
        self.mlflow.log_param("dataset_fingerprint", manifest["dataset_fingerprint"])
        self.mlflow.log_param("dataset_images", manifest["total_images"])
        self.mlflow.log_param("dataset_instances", manifest["total_instances"])

    def log_metrics(self, metrics: dict[str, float], *, prefix: str = "") -> None:
        safe = {
            f"{prefix}{_safe_metric_name(key)}": float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        if safe:
            self.mlflow.log_metrics(safe)


def _safe_metric_name(value: str) -> str:
    return (
        value.replace("metrics/", "")
        .replace("(", "_")
        .replace(")", "")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def flatten_scalars(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(flatten_scalars(item, name))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[name] = item
        else:
            result[name] = json.dumps(item, sort_keys=True, default=str)
    return result
