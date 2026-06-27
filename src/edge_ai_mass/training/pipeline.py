"""Stage orchestrator shared by the CLI and notebook interface."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from edge_ai_mass.training.checkpoints import resolve_model_source, resume_target_epochs
from edge_ai_mass.training.config import TrainingConfig, normalize_stages
from edge_ai_mass.training.preprocessing import build_yolo_dataset, inspect_sources
from edge_ai_mass.training.publication import TrainingOutputPublisher
from edge_ai_mass.training.state import PipelineState
from edge_ai_mass.training.visualization import create_dataset_visualizations
from edge_ai_mass.training.yolo import YOLOTrainer


class TrainingPipeline:
    """Professional, resumable YOLO training pipeline.

    The public API intentionally mirrors the CLI so notebooks remain thin and
    portable: load one YAML config, then run one or more named stages.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.overrides: list[str] = []
        self.state = PipelineState(config.artifacts_dir / "pipeline_state.json")

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        *,
        overrides: list[str] | None = None,
    ) -> "TrainingPipeline":
        instance = cls(TrainingConfig.load(path, overrides=overrides))
        instance.overrides = list(overrides or [])
        return instance

    def plan(self, stage: str = "all", *, skip_optuna: bool = False) -> dict[str, Any]:
        stages = normalize_stages(stage, skip_optuna=skip_optuna)
        sources = inspect_sources(self.config)
        packages = {
            "ultralytics": bool(importlib.util.find_spec("ultralytics")),
            "mlflow": bool(importlib.util.find_spec("mlflow")),
            "optuna": bool(importlib.util.find_spec("optuna")),
            "dvc": bool(importlib.util.find_spec("dvc")),
            "onnx": bool(importlib.util.find_spec("onnx")),
            "tensorrt": bool(importlib.util.find_spec("tensorrt")),
            "zenml": bool(importlib.util.find_spec("zenml")),
            "kaggle": bool(importlib.util.find_spec("kaggle")),
        }
        blocking_issues = []
        if "preprocess" in stages:
            for source in sources:
                if source["enabled"] and source["required"]:
                    if not source["annotations_exist"] or not source["images_exist"]:
                        blocking_issues.append(f"missing required dataset source: {source['name']}")
        required_packages = set()
        if set(stages).intersection({"tune", "train", "evaluate", "export", "register"}):
            required_packages.update({"ultralytics", "mlflow"})
        if "tune" in stages:
            required_packages.add("optuna")
        if "export" in stages and self.config.payload.get("export", {}).get("enabled", False):
            export_formats = set(self.config.payload["export"].get("formats", []))
            if "onnx" in export_formats:
                required_packages.add("onnx")
            if "engine" in export_formats:
                required_packages.add("tensorrt")
        if self._zenml_enabled():
            required_packages.add("zenml")
        publication = self.config.payload.get("publication", {})
        if "publish" in stages and publication and publication.get("enabled", True):
            required_packages.add("kaggle")
            if publication.get("require_checkpoint", True) and "train" not in stages:
                has_reusable_model = any(
                    path.is_file()
                    for path in (
                        self.config.artifacts_dir / "checkpoints" / "latest.json",
                        self.config.artifacts_dir / "models" / "best.pt",
                        self.config.artifacts_dir / "models" / "last.pt",
                    )
                )
                if not has_reusable_model:
                    blocking_issues.append(
                        "publish requires a reusable checkpoint/model; run train first"
                    )
        blocking_issues.extend(
            f"missing Python package: {name}"
            for name in sorted(required_packages)
            if not packages[name]
        )
        try:
            model_source = resolve_model_source(
                self.config,
                artifacts_dir=self.config.artifacts_dir,
                prefer_resume="train" in stages,
            )
            model_source_payload = {
                "path": model_source.path,
                "kind": model_source.kind,
                "resume": model_source.resume,
                "completed_epochs": model_source.completed_epochs,
                "target_epochs": resume_target_epochs(self.config, model_source),
                "manifest": model_source.manifest,
            }
        except (FileNotFoundError, RuntimeError) as exc:
            blocking_issues.append(str(exc))
            model_source_payload = None
        return {
            "config": str(self.config.source_path),
            "config_digest": self.config.digest,
            "pipeline": self.config.payload["project"]["pipeline"],
            "model_stage": self.config.payload["project"].get("model_stage", "detection"),
            "execution_stages": stages,
            "checkpoint": str(self.config.payload["model"]["checkpoint"]),
            "model_source": model_source_payload,
            "dataset_dir": str(self.config.dataset_dir),
            "artifacts_dir": str(self.config.artifacts_dir),
            "sources": sources,
            "packages": packages,
            "ready": not blocking_issues,
            "blocking_issues": blocking_issues,
        }

    def run(
        self,
        stage: str = "all",
        *,
        skip_optuna: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        stages = normalize_stages(stage, skip_optuna=skip_optuna)
        if self._zenml_enabled():
            from edge_ai_mass.training.zenml_pipeline import run_zenml_pipeline

            settings = self.config.payload.get("orchestration", {}).get("zenml", {})
            return run_zenml_pipeline(
                config_path=self.config.source_path,
                overrides=self.overrides,
                stages=stages,
                force=force,
                pipeline_name=str(
                    settings.get("pipeline_name", "edge_ai_mass_yolo_training")
                ),
            )
        return self.run_native(",".join(stages), force=force)

    def run_native(
        self,
        stage: str,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Execute native stages without entering ZenML recursively."""
        stages = normalize_stages(stage)
        self._prepare_state(force=force)
        trainer = YOLOTrainer(self.config, self.state)
        
        
        print(f"stages to be executed : ", stages)
        results: dict[str, Any] = {}
        for execution_stage in stages:
            if execution_stage == "preprocess":
                manifest = build_yolo_dataset(self.config)
                visualizations = create_dataset_visualizations(self.config, force=force)
                self.state.update(
                    "preprocess",
                    config_digest=self.config.digest,
                    data_config_digest=self.config.data_digest,
                    dataset_manifest=str(self.config.dataset_dir / "dataset_manifest.json"),
                    dataset_fingerprint=manifest["dataset_fingerprint"],
                    dataset_visualizations=visualizations,
                )
                results[execution_stage] = {
                    "dataset": manifest,
                    "visualizations": visualizations,
                }
            elif execution_stage == "tune":
                results[execution_stage] = trainer.tune()
            elif execution_stage == "train":
                results[execution_stage] = trainer.train()
            elif execution_stage == "evaluate":
                results[execution_stage] = trainer.evaluate()
            elif execution_stage == "export":
                results[execution_stage] = trainer.export()
            elif execution_stage == "register":
                results[execution_stage] = trainer.register(force=force)
            elif execution_stage == "publish":
                results[execution_stage] = TrainingOutputPublisher(
                    self.config,
                    self.state,
                ).publish()
        return results

    def _zenml_enabled(self) -> bool:
        return bool(
            self.config.payload.get("orchestration", {})
            .get("zenml", {})
            .get("enabled", True)
        )

    def _prepare_state(self, *, force: bool) -> None:
        previous = self.state.data.get("config_digest")
        completed = set(self.state.data.get("completed_stages", []))
        model_work_exists = completed.intersection(
            {"tune", "train", "evaluate", "export", "register"}
        )
        if previous and previous != self.config.digest and model_work_exists:
            if not force:
                raise RuntimeError(
                    "This run name already contains artifacts from a different config. "
                    "Choose a new training.run_name or rerun with --force."
                )
            self.state.reset(config_digest=self.config.digest)
        elif not previous:
            self.state.data["config_digest"] = self.config.digest
            self.state.save()
