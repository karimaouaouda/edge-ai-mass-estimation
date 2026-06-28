"""ZenML orchestration for mass-estimation stages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from zenml import pipeline, step


@step(name="initialize_mass_estimation_run", enable_cache=False)
def initialize_mass_estimation_run(
    config_path: str,
    overrides: list[str],
    force: bool,
    selected_stages: list[str],
) -> str:
    from edge_ai_mass.mass_estimation.pipeline import MassEstimationPipeline

    mass_pipeline = MassEstimationPipeline.from_config(config_path, overrides=overrides)
    mass_pipeline._prepare_state(force=force)
    results_dir = mass_pipeline.config.artifacts_dir / "zenml" / "stage_results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    context = {
        "config": str(mass_pipeline.config.source_path),
        "config_digest": mass_pipeline.config.digest,
        "artifacts_dir": str(mass_pipeline.config.artifacts_dir),
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
    del upstream
    if not enabled:
        print(f"ZenML mass-estimation stage '{stage_name}' skipped.")
        return f"{stage_name}:skipped"
    from edge_ai_mass.mass_estimation.pipeline import MassEstimationPipeline

    mass_pipeline = MassEstimationPipeline.from_config(config_path, overrides=overrides)
    native_result = mass_pipeline.run_native(stage_name, force=force)
    result = native_result.get(stage_name, native_result)
    results_dir = mass_pipeline.config.artifacts_dir / "zenml" / "stage_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(results_dir / f"{stage_name}.json", result)
    return f"{stage_name}:completed"


@step(name="mass_preprocess_stage", enable_cache=False)
def mass_preprocess_stage(
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


@step(name="mass_features_stage", enable_cache=False)
def mass_features_stage(
    enabled: bool,
    config_path: str,
    overrides: list[str],
    force: bool,
    upstream: str,
) -> str:
    return _execute_stage(
        stage_name="features",
        enabled=enabled,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=upstream,
    )


@step(name="mass_split_stage", enable_cache=False)
def mass_split_stage(
    enabled: bool,
    config_path: str,
    overrides: list[str],
    force: bool,
    upstream: str,
) -> str:
    return _execute_stage(
        stage_name="split",
        enabled=enabled,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=upstream,
    )


@step(name="mass_train_stage", enable_cache=False)
def mass_train_stage(
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


@step(name="mass_evaluate_stage", enable_cache=False)
def mass_evaluate_stage(
    enabled: bool,
    config_path: str,
    overrides: list[str],
    force: bool,
    upstream: str,
) -> str:
    return _execute_stage(
        stage_name="evaluate",
        enabled=enabled,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=upstream,
    )


@step(name="mass_register_stage", enable_cache=False)
def mass_register_stage(
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


@pipeline(
    name="edge_ai_mass_mass_estimation",
    enable_cache=False,
    enable_step_logs=True,
    tags=["edge-ai-mass", "mass-estimation", "training"],
)
def governed_mass_estimation_pipeline(
    config_path: str,
    overrides: list[str],
    force: bool,
    selected_stages: list[str],
) -> None:
    token = initialize_mass_estimation_run(
        config_path=config_path,
        overrides=overrides,
        force=force,
        selected_stages=selected_stages,
    )
    token = mass_preprocess_stage(
        enabled="preprocess" in selected_stages,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=token,
    )
    token = mass_features_stage(
        enabled="features" in selected_stages,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=token,
    )
    token = mass_split_stage(
        enabled="split" in selected_stages,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=token,
    )
    token = mass_train_stage(
        enabled="train" in selected_stages,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=token,
    )
    token = mass_evaluate_stage(
        enabled="evaluate" in selected_stages,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=token,
    )
    mass_register_stage(
        enabled="register" in selected_stages,
        config_path=config_path,
        overrides=overrides,
        force=force,
        upstream=token,
    )


def run_mass_zenml_pipeline(
    *,
    config_path: str | Path,
    overrides: list[str],
    stages: list[str],
    force: bool,
    pipeline_name: str,
) -> dict[str, Any]:
    run = governed_mass_estimation_pipeline(
        config_path=str(Path(config_path).resolve()),
        overrides=list(overrides),
        force=force,
        selected_stages=list(stages),
    )

    from edge_ai_mass.mass_estimation.pipeline import MassEstimationPipeline

    mass_pipeline = MassEstimationPipeline.from_config(config_path, overrides=overrides)
    results_dir = mass_pipeline.config.artifacts_dir / "zenml" / "stage_results"
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
        "registered_pipeline": "edge_ai_mass_mass_estimation",
        "run_id": str(run_id) if run_id else None,
        "selected_stages": stages,
    }
    mass_pipeline.state.patch(zenml=metadata)
    return {"zenml": metadata, **results}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = ["governed_mass_estimation_pipeline", "run_mass_zenml_pipeline"]
