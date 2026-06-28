"""Ultralytics YOLO trainer with Optuna optimization and MLflow governance."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from edge_ai_mass.training.checkpoints import (
    CheckpointStore,
    resolve_model_source,
    resume_target_epochs,
)
from edge_ai_mass.training.config import TrainingConfig
from edge_ai_mass.training.state import PipelineState
from edge_ai_mass.training.tracking import MLflowSession, flatten_scalars
from edge_ai_mass.training.visualization import IMAGE_SUFFIXES, create_dataset_visualizations

logger = logging.getLogger(__name__)


class YOLOTrainer:
    def __init__(self, config: TrainingConfig, state: PipelineState):
        self.config = config
        self.state = state
        self.artifacts_dir = config.artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def tune(self) -> dict[str, Any]:
        tuning = self.config.payload.get("tuning", {})
        if not tuning.get("enabled", True) or int(tuning.get("trials", 0)) <= 0:
            result = {"enabled": False, "best_params": {}}
            self._write_json("optimization/optuna_best.json", result)
            self.state.update("tune", best_params={}, optuna={"enabled": False})
            return result

        optuna = _require_optuna()
        YOLO = _require_yolo()
        tracker = MLflowSession(self.config, self.state)
        storage = _optuna_storage(
            self.config, str(tuning.get("storage", "artifacts/optuna/yolo.db"))
        )
        sampler = optuna.samplers.TPESampler(
            seed=int(tuning.get("seed", 42)),
            multivariate=bool(tuning.get("multivariate", True)),
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(tuning.get("pruner_startup_trials", 3)),
            n_warmup_steps=int(tuning.get("pruner_warmup_epochs", 3)),
        )
        study_name = str(tuning.get("study_name", f"{self.config.run_name}-optuna"))
        if tuning.get("version_by_config", True):
            dataset_version = self._dataset_manifest()["dataset_fingerprint"][:10]
            study_name = (
                f"{study_name}-{dataset_version}-{self.config.optimization_digest[:10]}"
            )
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=str(tuning.get("direction", "maximize")),
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
        metric_name = str(tuning.get("metric", "mask_map50_95"))
        search_space = tuning.get("search_space", {})
        checkpoint = resolve_model_source(
            self.config,
            artifacts_dir=self.artifacts_dir,
            prefer_resume=False,
        ).path

        with tracker.run():
            tracker.log_dataset(self._dataset_manifest())
            self._log_dataset_visualizations(tracker)

            def objective(trial: Any) -> float:
                parameters = {
                    name: _suggest(trial, name, specification)
                    for name, specification in search_space.items()
                }
                trial_name = f"{self.config.run_name}-trial-{trial.number:04d}"
                with tracker.mlflow.start_run(run_name=trial_name, nested=True):
                    tracker.mlflow.log_params(flatten_scalars(parameters))
                    model = YOLO(checkpoint, task=self.config.payload["model"]["task"])

                    def report_epoch(trainer: Any) -> None:
                        score = metric_value(getattr(trainer, "metrics", {}), metric_name)
                        if score is None:
                            return
                        trial.report(score, step=int(getattr(trainer, "epoch", 0)))
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                    model.add_callback("on_fit_epoch_end", report_epoch)
                    try:
                        trial_args = self._train_args(
                            parameters, tuning=True, run_name=trial_name
                        )
                        model.train(**trial_args)
                        metrics = model.val(
                            data=str(self.config.dataset_dir / "dataset.yaml"),
                            split=str(tuning.get("split", "val")),
                            imgsz=int(trial_args["imgsz"]),
                            batch=trial_args["batch"],
                            device=self.config.payload["training"].get("device", 0),
                            plots=False,
                        )
                        normalized = normalize_metrics(metrics)
                        tracker.log_metrics(normalized, prefix="objective_")
                        score = metric_value(metrics, metric_name)
                        if score is None:
                            raise RuntimeError(
                                f"Optuna metric '{metric_name}' was not found. "
                                f"Available: {list(normalized)}"
                            )
                        return score
                    finally:
                        del model
                        _release_accelerator_memory()

            study.optimize(
                objective,
                n_trials=int(tuning.get("trials", 20)),
                timeout=tuning.get("timeout_seconds"),
                gc_after_trial=True,
                show_progress_bar=bool(tuning.get("show_progress_bar", True)),
                catch=(RuntimeError, ValueError),
            )
            optimization_plots = self._write_optuna_visualizations(study)
            result = {
                "enabled": True,
                "study_name": study.study_name,
                "storage": storage,
                "direction": study.direction.name.lower(),
                "metric": metric_name,
                "best_value": float(study.best_value),
                "best_params": dict(study.best_params),
                "completed_trials": len(study.trials),
                "trials_by_state": {
                    state.name.lower(): sum(trial.state == state for trial in study.trials)
                    for state in optuna.trial.TrialState
                },
                "visualizations": optimization_plots,
            }
            path = self._write_json("optimization/optuna_best.json", result)
            tracker.mlflow.log_artifact(str(path), artifact_path="optimization")
            plots_dir = self.artifacts_dir / "optimization" / "visualizations"
            if plots_dir.is_dir():
                tracker.mlflow.log_artifacts(
                    str(plots_dir), artifact_path="optimization/visualizations"
                )
            tracker.mlflow.log_params(
                {f"optuna.best.{key}": value for key, value in result["best_params"].items()}
            )
            tracker.mlflow.log_metric(f"optuna_best_{metric_name}", result["best_value"])

        self.state.update("tune", best_params=result["best_params"], optuna=result)
        return result

    def train(self) -> dict[str, Any]:
        YOLO = _require_yolo()
        tracker = MLflowSession(self.config, self.state)
        best_params = dict(self.state.data.get("best_params", {}))
        model_source = resolve_model_source(
            self.config,
            artifacts_dir=self.artifacts_dir,
            prefer_resume=True,
        )
        model = YOLO(model_source.path, task=self.config.payload["model"]["task"])
        dataset_manifest = self._dataset_manifest()
        checkpoint_store = CheckpointStore(
            self.config,
            self.state,
            dataset_fingerprint=dataset_manifest["dataset_fingerprint"],
            model_source=model_source,
        )
        if checkpoint_store.enabled:
            model.add_callback("on_model_save", checkpoint_store.callback)
        target_epochs = resume_target_epochs(self.config, model_source)
        if model_source.resume:
            # Ultralytics restores train_args from the checkpoint and normally
            # ignores a new epochs value. Updating it at this lifecycle point
            # preserves optimizer state while allowing controlled chunking.
            def configure_resume_target(trainer: Any) -> None:
                trainer.epochs = target_epochs
                trainer.args.epochs = target_epochs

            model.add_callback("on_pretrain_routine_start", configure_resume_target)
        with tracker.run() as run:
            tracker.log_dataset(dataset_manifest)
            self._log_dataset_visualizations(tracker)
            train_args = self._train_args(
                best_params,
                tuning=False,
                run_name=self.config.run_name,
            )
            if model_source.resume:
                train_args["resume"] = True
            tracker.mlflow.log_params(flatten_scalars(train_args))
            tracker.mlflow.log_dict(
                {
                    "kind": model_source.kind,
                    "path": model_source.path,
                    "resume": model_source.resume,
                    "completed_epochs": model_source.completed_epochs,
                    "manifest": model_source.manifest,
                    "checkpoint_interval_epochs": checkpoint_store.interval,
                    "target_epochs": target_epochs,
                },
                "training/model_source.json",
            )
            result = model.train(**train_args)
            run_dir = Path(
                getattr(result, "save_dir", getattr(model.trainer, "save_dir", ""))
            ).resolve()
            early_stopping = self._early_stopping_report(
                model.trainer,
                train_args=train_args,
                target_epochs=target_epochs,
            )
            source_best = run_dir / "weights" / "best.pt"
            source_last = run_dir / "weights" / "last.pt"
            if not source_best.is_file():
                raise RuntimeError(
                    f"Ultralytics did not produce expected checkpoint: {source_best}"
                )
            models_dir = self.artifacts_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            best_path = models_dir / "best.pt"
            last_path = models_dir / "last.pt"
            shutil.copy2(source_best, best_path)
            if source_last.is_file():
                shutil.copy2(source_last, last_path)
            training_metrics = normalize_metrics(getattr(model.trainer, "metrics", {}))
            tracker.log_metrics(training_metrics, prefix="train_")
            tracker.mlflow.log_dict(early_stopping, "training/early_stopping.json")
            tracker.mlflow.log_artifact(str(best_path), artifact_path="weights")
            if last_path.is_file():
                tracker.mlflow.log_artifact(str(last_path), artifact_path="weights")
            training_artifacts = self._collect_training_artifacts(run_dir)
            training_artifacts_dir = self.artifacts_dir / "training"
            if training_artifacts_dir.is_dir():
                tracker.mlflow.log_artifacts(
                    str(training_artifacts_dir), artifact_path="training"
                )
            checkpoints_dir = self.artifacts_dir / "checkpoints"
            if checkpoints_dir.is_dir():
                tracker.mlflow.log_artifacts(
                    str(checkpoints_dir), artifact_path="checkpoints"
                )
            latest_checkpoint = _read_optional_json(checkpoints_dir / "latest.json")
            summary = {
                "mlflow_run_id": run.info.run_id,
                "checkpoint": str(self.config.payload["model"]["checkpoint"]),
                "model_source": {
                    "path": model_source.path,
                    "kind": model_source.kind,
                    "resume": model_source.resume,
                    "completed_epochs": model_source.completed_epochs,
                    "manifest": model_source.manifest,
                },
                "resumed": model_source.resume,
                "target_epochs": target_epochs,
                "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
                "best_params": best_params,
                "best_weights": str(best_path),
                "last_weights": str(last_path) if last_path.is_file() else None,
                "early_stopping": early_stopping,
                "latest_checkpoint": latest_checkpoint,
                "checkpoints_dir": str(checkpoints_dir),
                "ultralytics_run_dir": str(run_dir),
                "metrics": training_metrics,
                "artifacts": training_artifacts,
            }
            summary_path = self._write_json("reports/training_summary.json", summary)
            tracker.mlflow.log_artifact(str(summary_path), artifact_path="reports")
        self.state.update(
            "train",
            best_weights=str(best_path),
            last_weights=summary["last_weights"],
            training=summary,
        )
        del model
        _release_accelerator_memory()
        return summary

    def evaluate(self) -> dict[str, Any]:
        self.state.require("best_weights")
        YOLO = _require_yolo()
        evaluation = self.config.payload.get("evaluation", {})
        model = YOLO(
            str(self.state.data["best_weights"]),
            task=self.config.payload["model"]["task"],
        )
        tracker = MLflowSession(self.config, self.state)
        reports: dict[str, Any] = {}
        artifact_reports: dict[str, Any] = {}
        evaluation_root = self.artifacts_dir / "evaluation"
        gc.collect()
        torch.cuda.empty_cache()        
        batch = 1
        print(f"train on batch {batch if batch is not None else 'auto'} with workers {evaluation.get('workers', 0)}")
        with tracker.run(), torch.no_grad():
            model.eval()
            
            for split in evaluation.get("splits", ["val", "test"]):
                split_dir = evaluation_root / str(split)
                if split_dir.exists():
                    shutil.rmtree(split_dir)
                    
                metrics = model.val(
                    data=str(self.config.dataset_dir / "dataset.yaml"),
                    split=str(split),
                    imgsz=int(
                        evaluation.get(
                            "imgsz", self.config.payload["training"].get("imgsz", 640)
                        )
                    ),
                    batch=batch,
                    device="0",
                    conf=float(evaluation.get("conf", 0.001)),
                    iou=float(evaluation.get("iou", 0.7)),
                    plots=bool(evaluation.get("plots", False)),
                    save_json=bool(evaluation.get("save_json", True)),
                    project=str(evaluation_root),
                    name=str(split),
                    exist_ok=True,
                    workers=int(evaluation.get("workers", 0)),
                    max_det=int(evaluation.get("max_det", 3)),
                )
                reports[str(split)] = normalize_metrics(metrics)
                tracker.log_metrics(reports[str(split)], prefix=f"{split}_")
                actual_split_dir = Path(getattr(metrics, "save_dir", split_dir))
                annotated = self._write_annotated_evaluation_samples(
                    model,
                    split=str(split),
                    split_dir=actual_split_dir,
                    evaluation=evaluation,
                )
                curves = _move_selected(
                    actual_split_dir,
                    actual_split_dir / "curves",
                    patterns={
                        "*curve*.png",
                        "confusion_matrix*.png",
                        "results*.png",
                        "results*.csv",
                    },
                )
                artifact_reports[str(split)] = {
                    "output_dir": str(actual_split_dir),
                    "curves": curves,
                    "annotated_samples": annotated,
                }
                if actual_split_dir.is_dir():
                    tracker.mlflow.log_artifacts(
                        str(actual_split_dir), artifact_path=f"evaluation/{split}"
                    )

            self._enforce_metric_gates(reports, evaluation.get("quality_gates", {}))
            report = {
                "weights": str(self.state.data["best_weights"]),
                "dataset_fingerprint": self._dataset_manifest()["dataset_fingerprint"],
                "splits": reports,
                "artifacts": artifact_reports,
                "quality_gates": evaluation.get("quality_gates", {}),
                "passed": True,
            }
            report_path = self._write_json("reports/evaluation.json", report)
            tracker.mlflow.log_artifact(str(report_path), artifact_path="reports")
        self.state.update("evaluate", evaluation=report, evaluation_report=str(report_path))
        del model
        _release_accelerator_memory()
        return report

    def export(self) -> dict[str, Any]:
        """Export only the selected best checkpoint to one or more deployment formats."""
        export_cfg = self.config.payload.get("export", {})
        if not export_cfg.get("enabled", False):
            result = {"enabled": False, "reason": "export.enabled is false", "formats": []}
            self.state.update("export", exports=result)
            return result

        self.state.require("best_weights")
        YOLO = _require_yolo()
        tracker = MLflowSession(self.config, self.state)
        best_weights = Path(self.state.data["best_weights"]).resolve()
        formats = [str(value) for value in export_cfg["formats"]]
        output_root = self.artifacts_dir / "exports"
        output_root.mkdir(parents=True, exist_ok=True)
        common_options = dict(export_cfg.get("options", {}).get("common", {}))
        format_options = export_cfg.get("options", {}).get("formats", {})
        continue_on_error = bool(export_cfg.get("continue_on_error", False))
        entries: list[dict[str, Any]] = []

        with tracker.run():
            for export_format in formats:
                destination_dir = output_root / export_format
                if destination_dir.exists():
                    shutil.rmtree(destination_dir)
                destination_dir.mkdir(parents=True, exist_ok=True)
                options = {
                    **common_options,
                    **dict(format_options.get(export_format, {})),
                }
                if options.get("int8") and "data" not in options:
                    options["data"] = str(self.config.dataset_dir / "dataset.yaml")
                model = YOLO(str(best_weights), task=self.config.payload["model"]["task"])
                try:
                    exported_value = model.export(format=export_format, **options)
                    exported_paths = _normalize_export_paths(exported_value)
                    organized_paths = [
                        _organize_export(path, destination_dir, best_weights)
                        for path in exported_paths
                    ]
                    entry = {
                        "format": export_format,
                        "status": "complete",
                        "options": options,
                        "artifacts": [
                            {
                                "path": str(path),
                                "sha256": _path_digest(path),
                                "size_bytes": _path_size(path),
                            }
                            for path in organized_paths
                        ],
                    }
                    tracker.mlflow.log_artifacts(
                        str(destination_dir), artifact_path=f"exports/{export_format}"
                    )
                except Exception as exc:
                    entry = {
                        "format": export_format,
                        "status": "failed",
                        "options": options,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    entries.append(entry)
                    del model
                    _release_accelerator_memory()
                    if continue_on_error:
                        continue
                    raise RuntimeError(
                        f"Failed to export best model as '{export_format}': {exc}"
                    ) from exc
                entries.append(entry)
                del model
                _release_accelerator_memory()

            result = {
                "enabled": True,
                "source_best_weights": str(best_weights),
                "source_sha256": _path_digest(best_weights),
                "formats": entries,
                "successful_formats": [
                    entry["format"] for entry in entries if entry["status"] == "complete"
                ],
                "failed_formats": [
                    entry["format"] for entry in entries if entry["status"] == "failed"
                ],
                "output_dir": str(output_root),
            }
            manifest_path = self._write_json("exports/exports_manifest.json", result)
            tracker.mlflow.log_artifact(str(manifest_path), artifact_path="exports")
        self.state.update("export", exports=result, exports_manifest=str(manifest_path))
        return result

    def register(self, *, force: bool = False) -> dict[str, Any]:
        registry = self.config.payload["tracking"].get("registry", {})
        if not registry.get("enabled", True):
            result = {"enabled": False, "reason": "tracking.registry.enabled is false"}
            self.state.update("register", registration=result)
            return result
        if self.state.data.get("registration", {}).get("version") and not force:
            return self.state.data["registration"]
        self.state.require("best_weights", "evaluation")
        tracker = MLflowSession(self.config, self.state)
        mlflow = tracker.mlflow
        model_name = str(registry.get("name", "edge-ai-mass-yolo-segmentation"))
        alias = str(registry.get("alias", "candidate"))
        artifact_path = str(registry.get("artifact_path", "model"))

        with tracker.run() as run:
            python_model = _build_pyfunc_model(mlflow)
            log_model_args = {
                "python_model": python_model,
                "artifacts": {"weights": str(Path(self.state.data["best_weights"]).resolve())},
                "pip_requirements": [
                    "mlflow>=2.14",
                    "ultralytics>=8.0",
                    "pandas>=2.0",
                    "Pillow>=10.0",
                ],
            }
            if int(str(mlflow.__version__).split(".", 1)[0]) >= 3:
                log_model_args["name"] = artifact_path
            else:
                log_model_args["artifact_path"] = artifact_path
            model_info = mlflow.pyfunc.log_model(**log_model_args)
            model_uri = model_info.model_uri
            registered = mlflow.register_model(model_uri=model_uri, name=model_name)
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(model_name, alias, registered.version)
            client.set_model_version_tag(
                model_name,
                registered.version,
                "dataset_fingerprint",
                self._dataset_manifest()["dataset_fingerprint"],
            )
            client.set_model_version_tag(
                model_name, registered.version, "config_digest", self.config.digest
            )
            result = {
                "enabled": True,
                "name": model_name,
                "version": str(registered.version),
                "alias": alias,
                "model_uri": model_uri,
                "run_id": run.info.run_id,
            }
            result_path = self._write_json("reports/registration.json", result)
            mlflow.log_artifact(str(result_path), artifact_path="reports")
        self.state.update("register", registration=result)
        return result

    def _dataset_manifest(self) -> dict[str, Any]:
        path = self.config.dataset_dir / "dataset_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset manifest not found: {path}. Run --stage preprocess before model stages."
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("data_config_digest") != self.config.data_digest:
            raise RuntimeError(
                "Dataset manifest does not match the current data config. "
                "Run --stage preprocess before model stages."
            )
        return manifest

    def _train_args(
        self,
        overrides: dict[str, Any],
        *,
        tuning: bool,
        run_name: str,
    ) -> dict[str, Any]:
        training = self.config.payload["training"]
        tuning_cfg = self.config.payload.get("tuning", {})
        args: dict[str, Any] = {
            "data": str(self.config.dataset_dir / "dataset.yaml"),
            "epochs": int(
                tuning_cfg.get("epochs_per_trial", 15)
                if tuning
                else training.get("epochs", 100)
            ),
            "batch": training.get("batch", 16),
            "imgsz": int(training.get("imgsz", 640)),
            "patience": self._early_stopping_patience(),
            "device": training.get("device", 0),
            "workers": int(training.get("workers", 8)),
            "cache": training.get("cache", False),
            "amp": bool(training.get("amp", True)),
            "seed": int(training.get("seed", 42)),
            "deterministic": bool(training.get("deterministic", True)),
            "optimizer": training.get("optimizer", "AdamW"),
            "project": str(self.artifacts_dir / ("tuning_runs" if tuning else "training_runs")),
            "name": run_name,
            "exist_ok": bool(tuning),
            "plots": not tuning,
            "verbose": bool(training.get("verbose", True)),
        }
        checkpointing = training.get("checkpointing", {})
        if not tuning:
            args["save"] = True
            # The pipeline stores richer snapshots in artifacts/checkpoints via
            # on_model_save. Disable duplicate Ultralytics epoch*.pt files.
            args["save_period"] = -1
        args.update(training.get("hyperparameters", {}))
        args.update(training.get("augmentation", {}))
        args.update(training.get("extra_args", {}))
        args.update(overrides)
        return args

    def _early_stopping_patience(self) -> int:
        """Resolve pipeline early stopping to Ultralytics' patience argument.

        Ultralytics monitors validation fitness internally and stops when that
        value has not improved for ``patience`` epochs. Passing ``patience=0``
        disables the stopper, so the pipeline keeps this native behavior while
        exposing a clearer ``training.early_stopping`` config block.
        """
        training = self.config.payload["training"]
        early_stopping = training.get("early_stopping", {})
        legacy_patience = int(training.get("patience", 20))
        enabled = bool(early_stopping.get("enabled", legacy_patience != 0))
        if not enabled:
            return 0
        return int(early_stopping.get("patience", legacy_patience))

    def _early_stopping_report(
        self,
        trainer: Any,
        *,
        train_args: dict[str, Any],
        target_epochs: int,
    ) -> dict[str, Any]:
        """Create a durable summary of early-stopping state for artifacts."""
        patience = int(train_args.get("patience", 0))
        completed_epochs = int(getattr(trainer, "epoch", -1)) + 1
        stopper = getattr(trainer, "stopper", None)
        best_epoch = getattr(stopper, "best_epoch", None)
        best_fitness = getattr(stopper, "best_fitness", None)
        stopped_early = bool(
            patience > 0
            and completed_epochs > 0
            and completed_epochs < int(target_epochs)
            and getattr(trainer, "stop", False)
        )
        return {
            "enabled": patience > 0,
            "patience": patience,
            "monitor": "validation_fitness",
            "mode": "max",
            "completed_epochs": completed_epochs,
            "target_epochs": int(target_epochs),
            "stopped_early": stopped_early,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_fitness": float(best_fitness) if best_fitness is not None else None,
            "epochs_without_improvement": (
                completed_epochs - int(best_epoch)
                if best_epoch is not None and completed_epochs >= 0
                else None
            ),
        }

    def _log_dataset_visualizations(self, tracker: MLflowSession) -> dict[str, Any]:
        report = create_dataset_visualizations(self.config)
        output_dir = self.artifacts_dir / "dataset_visualizations"
        if report.get("enabled") and output_dir.is_dir():
            tracker.mlflow.log_artifacts(
                str(output_dir), artifact_path="dataset_visualizations"
            )
        return report

    def _collect_training_artifacts(self, run_dir: Path) -> dict[str, Any]:
        output_dir = self.artifacts_dir / "training"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        curves_dir = output_dir / "curves"
        diagnostics_dir = output_dir / "diagnostics"
        batches_dir = output_dir / "annotated_batches"
        diagnostic_names = {"labels.jpg", "labels_correlogram.jpg"}
        curves = _copy_selected(
            run_dir,
            curves_dir,
            names={"results.csv", "results.png"},
            patterns={"*curve*.png", "confusion_matrix*.png"},
        )
        diagnostics = _copy_selected(run_dir, diagnostics_dir, names=diagnostic_names)
        batches = _copy_selected(
            run_dir,
            batches_dir,
            patterns={"train_batch*.jpg", "val_batch*.jpg"},
        )
        return {
            "curves": curves,
            "diagnostics": diagnostics,
            "annotated_batches": batches,
        }

    def _write_annotated_evaluation_samples(
        self,
        model: Any,
        *,
        split: str,
        split_dir: Path,
        evaluation: dict[str, Any],
    ) -> list[str]:
        sample_count = int(evaluation.get("annotated_samples", 12))
        if sample_count <= 0:
            return []
        images_root = self.config.dataset_dir / "images" / split
        image_paths = sorted(
            path for path in images_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        rng = random.Random(
            f"{evaluation.get('sample_seed', 42)}:{split}:evaluation"
        )
        selected = rng.sample(image_paths, min(sample_count, len(image_paths)))
        output_dir = split_dir / "annotated_samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for index, image_path in enumerate(selected):
            prediction = model.predict(
                source=str(image_path),
                imgsz=int(
                    evaluation.get(
                        "imgsz", self.config.payload["training"].get("imgsz", 640)
                    )
                ),
                conf=float(evaluation.get("prediction_conf", 0.25)),
                iou=float(evaluation.get("iou", 0.7)),
                device=evaluation.get(
                    "device", self.config.payload["training"].get("device", 0)
                ),
                verbose=False,
            )[0]
            plotted_bgr = prediction.plot()
            relative = image_path.relative_to(images_root)
            source_name = "_".join(relative.parts[:-1]) or "root"
            output_path = output_dir / f"{index:03d}_{source_name}_{image_path.stem}.jpg"
            Image.fromarray(plotted_bgr[..., ::-1].copy()).save(output_path, quality=92)
            outputs.append(str(output_path))
        return outputs

    def _write_optuna_visualizations(self, study: Any) -> list[str]:
        output_dir = self.artifacts_dir / "optimization" / "visualizations"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_parallel_coordinate,
                plot_param_importances,
                plot_slice,
            )

            plotters = {
                "optimization_history": plot_optimization_history,
                "parallel_coordinate": plot_parallel_coordinate,
                "parameter_importances": plot_param_importances,
                "parameter_slices": plot_slice,
            }
            for name, plotter in plotters.items():
                try:
                    path = output_dir / f"{name}.html"
                    plotter(study).write_html(str(path), include_plotlyjs="cdn")
                    outputs.append(str(path))
                except Exception as exc:
                    logger.warning(
                        "Could not render Optuna visualization %s: %s", name, exc
                    )
        except ImportError:
            logger.warning("Plotly is unavailable; Optuna HTML visualizations were skipped")
        return outputs

    def _enforce_metric_gates(
        self,
        reports: dict[str, dict[str, float]],
        gates: dict[str, Any],
    ) -> None:
        failures = []
        for qualified_name, minimum in gates.items():
            split, _, metric_name = str(qualified_name).partition(".")
            actual = reports.get(split, {}).get(metric_name)
            if actual is None or actual < float(minimum):
                failures.append(f"{qualified_name}={actual} < {minimum}")
        if failures:
            raise RuntimeError("Model quality gates failed: " + "; ".join(failures))

    def _write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.artifacts_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def _copy_selected(
    source_root: Path,
    destination_root: Path,
    *,
    names: set[str] | None = None,
    patterns: set[str] | None = None,
) -> list[str]:
    selected: set[Path] = set()
    for name in names or set():
        selected.update(path for path in source_root.rglob(name) if path.is_file())
    for pattern in patterns or set():
        selected.update(path for path in source_root.rglob(pattern) if path.is_file())
    outputs = []
    for source in sorted(selected):
        destination = destination_root / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        outputs.append(str(destination))
    return outputs


def _move_selected(
    source_root: Path,
    destination_root: Path,
    *,
    patterns: set[str],
) -> list[str]:
    selected = {
        path
        for pattern in patterns
        for path in source_root.glob(pattern)
        if path.is_file()
    }
    outputs = []
    destination_root.mkdir(parents=True, exist_ok=True)
    for source in sorted(selected):
        destination = destination_root / source.name
        if destination.exists():
            destination.unlink()
        shutil.move(str(source), str(destination))
        outputs.append(str(destination))
    return outputs


def _normalize_export_paths(value: Any) -> list[Path]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    paths = [Path(item).expanduser().resolve() for item in values if item]
    if not paths or any(not path.exists() for path in paths):
        raise RuntimeError(f"Ultralytics returned invalid export path(s): {value!r}")
    return paths


def _organize_export(source: Path, destination_dir: Path, best_weights: Path) -> Path:
    source = source.resolve()
    destination = destination_dir / source.name
    if source == destination.resolve():
        return source
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    # Ultralytics normally writes beside best.pt. Move those generated files so
    # deployment formats exist only under exports/<format>/.
    if source.parent == best_weights.parent and source != best_weights:
        shutil.move(str(source), str(destination))
    elif source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    return destination.resolve()


def _path_digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(_path_digest(file_path).encode())
    return digest.hexdigest()


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def normalize_metrics(metrics: Any) -> dict[str, float]:
    source = getattr(metrics, "results_dict", metrics)
    if not isinstance(source, dict):
        return {}
    result = {str(key): float(value) for key, value in source.items() if _is_number(value)}
    aliases = {
        "mask_map50_95": ("metrics/mAP50-95(M)", "metrics/mAP50-95(Mask)"),
        "mask_map50": ("metrics/mAP50(M)", "metrics/mAP50(Mask)"),
        "box_map50_95": ("metrics/mAP50-95(B)",),
        "box_map50": ("metrics/mAP50(B)",),
        "fitness": ("fitness",),
    }
    for alias, candidates in aliases.items():
        for candidate in candidates:
            if candidate in result:
                result[alias] = result[candidate]
                break
    return result


def metric_value(metrics: Any, preferred: str) -> float | None:
    return normalize_metrics(metrics).get(preferred)


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _suggest(trial: Any, name: str, specification: dict[str, Any]) -> Any:
    kind = str(specification.get("type", "float"))
    if kind == "float":
        return trial.suggest_float(
            name,
            float(specification["low"]),
            float(specification["high"]),
            log=bool(specification.get("log", False)),
            step=specification.get("step"),
        )
    if kind == "int":
        return trial.suggest_int(
            name,
            int(specification["low"]),
            int(specification["high"]),
            step=int(specification.get("step", 1)),
            log=bool(specification.get("log", False)),
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, specification["choices"])
    raise ValueError(f"Unsupported Optuna parameter type '{kind}' for '{name}'")


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _optuna_storage(config: TrainingConfig, value: str) -> str:
    if "://" in value:
        return value
    path = config.path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def _require_optuna():
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'Optuna is required for tuning. Install with: pip install -e ".[mlops]"'
        ) from exc
    return optuna


def _require_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Ultralytics is required for YOLO training. Install the project dependencies."
        ) from exc
    return YOLO


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _build_pyfunc_model(mlflow: Any) -> Any:
    class YOLOPyFuncModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context: Any) -> None:
            from ultralytics import YOLO

            self.model = YOLO(context.artifacts["weights"])

        def predict(self, context, model_input, params=None):
            import pandas as pd

            if hasattr(model_input, "columns") and "source" in model_input.columns:
                sources = model_input["source"].astype(str).tolist()
            elif isinstance(model_input, (list, tuple)):
                sources = [str(value) for value in model_input]
            else:
                raise ValueError(
                    "model_input must be a DataFrame with a 'source' column or a list of paths"
                )
            options = {"verbose": False, **(params or {})}
            predictions = self.model.predict(source=sources, **options)
            rows = []
            for result in predictions:
                boxes = []
                if result.boxes is not None:
                    xyxy = result.boxes.xyxy.cpu().tolist()
                    classes = result.boxes.cls.cpu().tolist()
                    confidences = result.boxes.conf.cpu().tolist()
                    boxes = [
                        {"xyxy": box, "class_id": int(class_id), "confidence": float(confidence)}
                        for box, class_id, confidence in zip(xyxy, classes, confidences)
                    ]
                polygons = result.masks.xy if result.masks is not None else []
                rows.append(
                    {
                        "source": str(result.path),
                        "predictions": json.dumps(
                            {"boxes": boxes, "polygons": [polygon.tolist() for polygon in polygons]}
                        ),
                    }
                )
            return pd.DataFrame(rows)

    return YOLOPyFuncModel()
