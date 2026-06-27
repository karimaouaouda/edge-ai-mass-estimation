"""ZenML orchestration layer over the repository's existing training stages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from zenml import pipeline, step


@step(name="initialize_training_run", enable_cache=False)
def initialize_training_run(
    config_path: str,
    overrides: list[str],
    force: bool,
    selected_stages: list[str],
) -> str:
    """Prepare durable state and clear results from the previous ZenML run."""
    from edge_ai_mass.training.pipeline import TrainingPipeline

    training = TrainingPipeline.from_config(config_path, overrides=overrides)
    training._prepare_state(force=force)
    results_dir = training.config.artifacts_dir / "zenml" / "stage_results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    context = {
        "config": str(training.config.source_path),
        "config_digest": training.config.digest,
        "artifacts_dir": str(training.config.artifacts_dir),
        "stages": selected_stages,
    }
    _write_json(results_dir.parent / "run_context.json", context)
    return "initialized"


def _execute_stage(
    *,
    stage_name: str,
    enabled: bool,
    config_path: str,
    overrides: list[str],
    force: bool,
    upstream: str,
) -> str:
    """Execute or explicitly skip one ordered native stage."""
    del upstream
    if not enabled:
        print(f"ZenML stage '{stage_name}' skipped by the selected execution plan.")
        return f"{stage_name}:skipped"

    from edge_ai_mass.training.pipeline import TrainingPipeline

    training = TrainingPipeline.from_config(config_path, overrides=overrides)
    native_result = training.run_native(stage_name, force=force)
    result = native_result.get(stage_name, native_result)
    results_dir = training.config.artifacts_dir / "zenml" / "stage_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(results_dir / f"{stage_name}.json", result)
    return f"{stage_name}:completed"


@step(name="preprocess_stage", enable_cache=False)
def preprocess_stage(
    enabled: bool,
    config_path: str,
    overrides: list[str],
    force: bool,
    upstream: str,
) -> str:
    return _execute_stage(
        stage_name="preprocess",
        enabled=enabled,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=upstream,
    )


@step(name="tune_stage", enable_cache=False)
def tune_stage(
    enabled: bool,
    config_path: str,
    overrides: list[str],
    force: bool,
    upstream: str,
) -> str:
    return _execute_stage(
        stage_name="tune",
        enabled=enabled,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=upstream,
    )


@step(name="train_stage", enable_cache=False)
def train_stage(
    enabled: bool,
    config_path: str,
    overrides: list[str],
    force: bool,
    upstream: str,
) -> str:
    return _execute_stage(
        stage_name="train",
        enabled=enabled,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=upstream,
    )


@step(name="evaluate_stage", enable_cache=False)
def evaluate_stage(
    enabled: bool,
    config_path: str,
    overrides: list[str],
    force: bool,
    upstream: str,
) -> str:
    print(f"evaluate_stage : ", enabled)
    return _execute_stage(
        stage_name="evaluate",
        enabled=enabled,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=upstream,
    )


@step(name="export_stage", enable_cache=False)
def export_stage(
    enabled: bool,
    config_path: str,
    overrides: list[str],
    force: bool,
    upstream: str,
) -> str:
    return _execute_stage(
        stage_name="export",
        enabled=enabled,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=upstream,
    )


@step(name="register_stage", enable_cache=False)
def register_stage(
    enabled: bool,
    config_path: str,
    overrides: list[str],
    force: bool,
    upstream: str,
) -> str:
    return _execute_stage(
        stage_name="register",
        enabled=enabled,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=upstream,
    )


@step(name="publish_stage", enable_cache=False)
def publish_stage(
    enabled: bool,
    config_path: str,
    overrides: list[str],
    force: bool,
    upstream: str,
) -> str:
    return _execute_stage(
        stage_name="publish",
        enabled=enabled,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=upstream,
    )


@pipeline(
    name="edge_ai_mass_yolo_training",
    enable_cache=False,
    enable_step_logs=True,
    tags=["edge-ai-mass", "yolo", "training"],
)
def governed_training_pipeline(
    config_path: str,
    overrides: list[str],
    force: bool,
    selected_stages: list[str],
) -> None:
    """Ordered training DAG; unselected stages remain visible as explicit skips."""
    token = initialize_training_run(
        config_path=config_path,
        overrides=overrides,
        force=force,
        selected_stages=selected_stages,
    )
    token = preprocess_stage(
        enabled="preprocess" in selected_stages,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=token,
    )
    token = tune_stage(
        enabled="tune" in selected_stages,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=token,
    )
    token = train_stage(
        enabled="train" in selected_stages,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=token,
    )
    token = evaluate_stage(
        enabled="evaluate" in selected_stages,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=token,
    )
    token = export_stage(
        enabled="export" in selected_stages,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=token,
    )
    token = register_stage(
        enabled="register" in selected_stages,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=token,
    )
    publish_stage(
        enabled="publish" in selected_stages,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=token,
    )


def run_zenml_pipeline(
    *,
    config_path: str | Path,
    overrides: list[str],
    stages: list[str],
    force: bool,
    pipeline_name: str,
) -> dict[str, Any]:
    """Run the ordered ZenML DAG and preserve the existing return contract."""
    run = governed_training_pipeline(
        config_path=str(Path(config_path).resolve()),
        overrides=list(overrides),
        force=force,
        selected_stages=list(stages),
    )

    from edge_ai_mass.training.pipeline import TrainingPipeline

    training = TrainingPipeline.from_config(config_path, overrides=overrides)
    results_dir = training.config.artifacts_dir / "zenml" / "stage_results"
    results = {
        stage_name: json.loads(
            (results_dir / f"{stage_name}.json").read_text(encoding="utf-8")
        )
        for stage_name in stages
    }
    run_id = getattr(run, "id", None)
    metadata = {
        "enabled": True,
        "pipeline": pipeline_name,
        "registered_pipeline": "edge_ai_mass_yolo_training",
        "run_id": str(run_id) if run_id else None,
        "selected_stages": stages,
    }
    training.state.patch(zenml=metadata)
    return {"zenml": metadata, **results}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = ["governed_training_pipeline", "run_zenml_pipeline"]
