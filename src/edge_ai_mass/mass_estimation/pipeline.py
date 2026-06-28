"""Stage orchestrator for the governed mass-estimation workflow."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from edge_ai_mass.mass_estimation.config import (
    MassEstimationConfig,
    normalize_mass_stages,
)
from edge_ai_mass.mass_estimation.data import preprocess_mass_dataset, split_feature_dataset
from edge_ai_mass.mass_estimation.evaluation import MassModelEvaluator
from edge_ai_mass.mass_estimation.features import build_feature_dataset
from edge_ai_mass.mass_estimation.training import MassModelTrainer
from edge_ai_mass.training.state import PipelineState


class MassEstimationPipeline:
    """Professional, resumable pipeline for hybrid residual mass estimation."""

    def __init__(self, config: MassEstimationConfig):
        self.config = config
        self.overrides: list[str] = []
        self.state = PipelineState(config.artifacts_dir / "pipeline_state.json")

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        *,
        overrides: list[str] | None = None,
    ) -> "MassEstimationPipeline":
        instance = cls(MassEstimationConfig.load(path, overrides=overrides))
        instance.overrides = list(overrides or [])
        return instance

    def plan(self, stage: str = "all") -> dict[str, Any]:
        stages = normalize_mass_stages(stage)
        packages = {
            "pandas": bool(importlib.util.find_spec("pandas")),
            "sklearn": bool(importlib.util.find_spec("sklearn")),
            "joblib": bool(importlib.util.find_spec("joblib")),
            "mlflow": bool(importlib.util.find_spec("mlflow")),
            "zenml": bool(importlib.util.find_spec("zenml")),
            "matplotlib": bool(importlib.util.find_spec("matplotlib")),
        }
        blocking_issues = []
        measurements = self.config.path(self.config.payload["data"]["measurements"])
        if "preprocess" in stages and not measurements.is_file():
            blocking_issues.append(f"missing mass measurement file: {measurements}")

        required_packages = {"pandas", "sklearn", "joblib"}
        if self._tracking_enabled():
            required_packages.add("mlflow")
        if self._zenml_enabled():
            required_packages.add("zenml")
        blocking_issues.extend(
            f"missing Python package: {name}"
            for name in sorted(required_packages)
            if not packages[name]
        )
        return {
            "config": str(self.config.source_path),
            "config_digest": self.config.digest,
            "data_config_digest": self.config.data_digest,
            "pipeline": self.config.payload["project"]["pipeline"],
            "model_stage": self.config.payload["project"].get("model_stage", "mass"),
            "execution_stages": stages,
            "measurements": str(measurements),
            "processed_dir": str(self.config.processed_dir),
            "artifacts_dir": str(self.config.artifacts_dir),
            "packages": packages,
            "ready": not blocking_issues,
            "blocking_issues": blocking_issues,
        }

    def run(self, stage: str = "all", *, force: bool = False) -> dict[str, Any]:
        stages = normalize_mass_stages(stage)
        if self._zenml_enabled():
            from edge_ai_mass.mass_estimation.zenml_pipeline import run_mass_zenml_pipeline

            settings = self.config.payload.get("orchestration", {}).get("zenml", {})
            return run_mass_zenml_pipeline(
                config_path=self.config.source_path,
                overrides=self.overrides,
                stages=stages,
                force=force,
                pipeline_name=str(
                    settings.get("pipeline_name", "edge_ai_mass_mass_estimation")
                ),
            )
        return self.run_native(",".join(stages), force=force)

    def run_native(self, stage: str, *, force: bool = False) -> dict[str, Any]:
        stages = normalize_mass_stages(stage)
        self._prepare_state(force=force)
        trainer = MassModelTrainer(self.config, self.state)
        evaluator = MassModelEvaluator(self.config, self.state)
        results: dict[str, Any] = {}
        for execution_stage in stages:
            if execution_stage == "preprocess":
                result = preprocess_mass_dataset(self.config)
                self.state.update(
                    "preprocess",
                    config_digest=self.config.digest,
                    data_config_digest=self.config.data_digest,
                    preprocessing=result,
                    preprocessed_dataset=str(
                        self.config.processed_dir / "preprocessed_objects.csv"
                    ),
                )
            elif execution_stage == "features":
                result = build_feature_dataset(self.config)
                self.state.update(
                    "features",
                    feature_dataset=result["features"],
                    feature_schema=result["schema"],
                    feature_summary=result["summary"],
                    feature_dataset_fingerprint=result["dataset_fingerprint"],
                )
            elif execution_stage == "split":
                result = split_feature_dataset(self.config)
                self.state.update("split", split_summary=result)
            elif execution_stage == "train":
                result = trainer.train()
            elif execution_stage == "evaluate":
                result = evaluator.evaluate()
            elif execution_stage == "register":
                result = trainer.register(force=force)
            else:  # pragma: no cover - normalize_mass_stages guards this
                raise RuntimeError(f"Unhandled mass-estimation stage: {execution_stage}")
            results[execution_stage] = result
        return results

    def _zenml_enabled(self) -> bool:
        return bool(
            self.config.payload.get("orchestration", {})
            .get("zenml", {})
            .get("enabled", True)
        )

    def _tracking_enabled(self) -> bool:
        return bool(self.config.payload["tracking"].get("enabled", True))

    def _prepare_state(self, *, force: bool) -> None:
        previous = self.state.data.get("config_digest")
        completed = set(self.state.data.get("completed_stages", []))
        model_work_exists = completed.intersection({"train", "evaluate", "register"})
        if previous and previous != self.config.digest and model_work_exists:
            if not force:
                raise RuntimeError(
                    "This mass-estimation run contains artifacts from a different config. "
                    "Choose a new training.run_name or rerun with --force."
                )
            self.state.reset(config_digest=self.config.digest)
        elif not previous:
            self.state.data["config_digest"] = self.config.digest
            self.state.save()
