"""Optional MLflow integration for the mass-estimation pipeline."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

from edge_ai_mass.mass_estimation.config import MassEstimationConfig
from edge_ai_mass.training.state import PipelineState, git_metadata
from edge_ai_mass.training.tracking import flatten_scalars, normalize_uri


class MassMLflowSession:
    """Lazy MLflow session that can be disabled for local tests and dry runs."""

    def __init__(self, config: MassEstimationConfig, state: PipelineState):
        self.config = config
        self.state = state
        self.enabled = bool(config.payload["tracking"].get("enabled", True))
        self.mlflow = _require_mlflow() if self.enabled else None
        if not self.enabled:
            return
        tracking = config.payload["tracking"]
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or str(
            tracking.get("uri", "./mlruns")
        )
        registry_uri = os.getenv("MLFLOW_REGISTRY_URI") or str(
            tracking.get("registry_uri", tracking_uri)
        )
        self.mlflow.set_tracking_uri(normalize_uri(tracking_uri, config.project_root))
        self.mlflow.set_registry_uri(normalize_uri(registry_uri, config.project_root))
        experiment = os.getenv("MLFLOW_EXPERIMENT_NAME") or str(
            tracking.get("experiment", "edge-ai-mass-residual-mass")
        )
        self.mlflow.set_experiment(experiment)

    @contextlib.contextmanager
    def run(self) -> Iterator[Any | None]:
        if not self.enabled:
            yield None
            return
        run_id = self.state.data.get("mlflow_run_id")
        kwargs = {"run_id": run_id} if run_id else {"run_name": self.config.run_name}
        with self.mlflow.start_run(**kwargs) as run:
            if not run_id:
                self.state.update("tracking", mlflow_run_id=run.info.run_id)
                self._log_initial_context()
            yield run

    def _log_initial_context(self) -> None:
        project = self.config.payload["project"]
        tags = {
            "project": str(project.get("name", "edge-ai-mass-estimation")),
            "model_stage": "mass",
            "pipeline": "mass_estimation",
            "config_digest": self.config.digest,
            "data_config_digest": self.config.data_digest,
            **{
                str(key): str(value)
                for key, value in self.config.payload["tracking"].get("tags", {}).items()
            },
        }
        self.mlflow.set_tags(tags)
        self.mlflow.log_dict(self.config.resolved_copy(), "configuration/resolved_config.json")
        self.mlflow.log_dict(git_metadata(self.config.project_root), "lineage/git.json")
        self.mlflow.log_param("config_digest", self.config.digest)
        self.mlflow.log_param("data_config_digest", self.config.data_digest)

    def log_metrics(self, metrics: dict[str, Any], *, prefix: str = "") -> None:
        if not self.enabled:
            return
        safe = {
            f"{prefix}{key}".replace(".", "_").replace("/", "_"): float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        if safe:
            self.mlflow.log_metrics(safe)

    def log_params(self, params: dict[str, Any], *, prefix: str = "") -> None:
        if not self.enabled:
            return
        flattened = flatten_scalars(params)
        self.mlflow.log_params({f"{prefix}{key}": value for key, value in flattened.items()})


def _require_mlflow():
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            'MLflow is required when tracking.enabled=true. Install with: pip install -e ".[mlops]"'
        ) from exc
    return mlflow
