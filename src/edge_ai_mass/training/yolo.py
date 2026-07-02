"""Ultralytics YOLO trainer with Optuna optimization and MLflow governance."""

from __future__ import annotations

import copy
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
from edge_ai_mass.training.devices import (
    device_to_log_value,
    is_cpu_device,
    is_directml_device,
    resolve_device_for_ultralytics,
    resolve_export_device,
)
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
                            device=trial_args["device"],
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
            rollback_to_selected=True,
        )
        if model_source.rollback:
            self.state.patch(
                latest_checkpoint=model_source.path,
                latest_checkpoint_manifest=model_source.manifest,
                checkpoint_history=model_source.rollback.get("checkpoint_history", []),
                checkpoint_rollback=model_source.rollback,
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
                    "resolved_device": device_to_log_value(train_args.get("device")),
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
    
    @torch.inference_mode()
    def evaluate(self) -> dict[str, Any]:
        print("evaluation start ==========> <===================")
        self.state.require("best_weights")
        YOLO = _require_yolo()
        base_evaluation = copy.deepcopy(self.config.payload.get("evaluation", {}))
        source_best_weights = Path(self.state.data["best_weights"]).resolve()
        requested_device = base_evaluation.get("device")
        if requested_device is None:
            requested_device = self.config.payload["training"].get("device", 0)
        current_device = resolve_device_for_ultralytics(
            requested_device,
            purpose="YOLO evaluation",
        )
        current_batch = self._effective_evaluation_batch(base_evaluation)
        current_plots = bool(base_evaluation.get("plots", False))
        fallback_used = False
        fallback_reason: str | None = None
        attempt_number = 0
        last_error: Exception | None = None
        max_attempts = 64

        while attempt_number < max_attempts:
            attempt_number += 1
            evaluation = self._evaluation_for_device(
                base_evaluation,
                device=current_device,
                batch=current_batch,
                plots=current_plots,
                cuda_oom_fallback=bool(
                    fallback_reason
                    and fallback_reason.endswith("_out_of_memory_cpu")
                ),
            )
            precision = self._evaluation_precision_config(evaluation)
            print(
                "[evaluation] Attempt "
                f"{attempt_number} starting on device="
                f"{evaluation.get('device')} precision={precision['precision']} "
                f"batch={evaluation.get('batch')} plots={evaluation.get('plots')} "
                f"strategy={fallback_reason or 'configured'}"
            )
            try:
                report, report_path = self._evaluate_once(
                    YOLO,
                    evaluation=evaluation,
                    precision=precision,
                    source_best_weights=source_best_weights,
                    requested_device=requested_device,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason,
                )
            except Exception as exc:
                last_error = exc
                oom_kind = _accelerator_out_of_memory_kind(exc)
                if (
                    _is_cuda_runtime_import_error(exc)
                    and self._evaluation_pre_export_enabled(base_evaluation)
                ):
                    print(
                        "[evaluation] CUDA runtime import error detected while using "
                        "the evaluation pre-export backend. This usually means an "
                        "ONNXRuntime/TensorRT GPU package expects CUDA libraries that "
                        "are not installed in this runtime."
                    )
                    print(f"[evaluation] Original error: {type(exc).__name__}: {exc}")
                    print(
                        "[evaluation] Disabling evaluation pre-export and retrying "
                        "with the original PyTorch weights."
                    )
                    logger.warning(
                        "Evaluation pre-export backend is incompatible with the CUDA "
                        "runtime; retrying with PyTorch weights. Error: %s",
                        exc,
                    )
                    base_evaluation = self._evaluation_without_pre_export(
                        base_evaluation,
                        reason="cuda_runtime_import_error",
                    )
                    self._remove_partial_evaluation_pre_exports()
                    _release_accelerator_memory()
                    fallback_used = True
                    fallback_reason = "cuda_runtime_import_disable_pre_export"
                    continue
                if oom_kind and not _is_cpu_device(current_device):
                    print(
                        f"[evaluation] {oom_kind.upper()} out-of-memory detected during "
                        "evaluation. Applying the next memory-saving retry step."
                    )
                    print(f"[evaluation] Original error: {type(exc).__name__}: {exc}")
                    logger.warning(
                        "%s OOM during evaluation attempt %s; retrying with a "
                        "lower-memory strategy. Error: %s",
                        oom_kind,
                        attempt_number,
                        exc,
                    )
                    self._remove_partial_evaluation_pre_exports()
                    _release_accelerator_memory()

                    if current_batch > 1:
                        next_batch = _next_smaller_evaluation_batch(current_batch)
                        print(
                            "[evaluation] Retrying on the accelerator with a smaller batch: "
                            f"{current_batch} -> {next_batch}."
                        )
                        current_batch = next_batch
                        fallback_used = True
                        fallback_reason = (
                            f"{oom_kind}_out_of_memory_reduce_batch_to_{next_batch}"
                        )
                        continue

                    if current_plots:
                        print(
                            "[evaluation] Batch is already 1. Disabling validation "
                            "plots and retrying on the accelerator."
                        )
                        current_plots = False
                        fallback_used = True
                        fallback_reason = f"{oom_kind}_out_of_memory_disable_plots"
                        continue

                    print(
                        "[evaluation] Accelerator still ran out of memory at batch=1 with "
                        "plots disabled. Retrying the full evaluation on CPU."
                    )
                    current_device = "cpu"
                    current_batch = 1
                    current_plots = False
                    fallback_used = True
                    fallback_reason = f"{oom_kind}_out_of_memory_cpu"
                    continue
                _release_accelerator_memory()
                raise

            self.state.update(
                "evaluate",
                evaluation=report,
                evaluation_report=str(report_path),
            )
            _release_accelerator_memory()
            if fallback_reason and fallback_reason.endswith("_out_of_memory_cpu"):
                print("[evaluation] CPU fallback evaluation completed successfully.")
            elif fallback_used:
                print(
                    "[evaluation] Evaluation completed successfully after accelerator "
                    f"memory mitigation: {fallback_reason}."
                )
            else:
                print("[evaluation] Evaluation completed successfully.")
            return report

        if last_error is not None:
            raise last_error
        raise RuntimeError(
            f"Evaluation exceeded the retry limit ({max_attempts}) without completing."
        )

    def _evaluate_once(
        self,
        YOLO: Any,
        *,
        evaluation: dict[str, Any],
        precision: dict[str, Any],
        source_best_weights: Path,
        requested_device: Any,
        fallback_used: bool,
        fallback_reason: str | None,
    ) -> tuple[dict[str, Any], Path]:
        tracker = MLflowSession(self.config, self.state)
        reports: dict[str, Any] = {}
        artifact_reports: dict[str, Any] = {}
        evaluation_root = self.artifacts_dir / "evaluation"
        model: Any | None = None
        _release_accelerator_memory()

        with tracker.run():
            pre_export_cfg = evaluation.get("pre_export") or {}
            print(
                "[evaluation] Pre-export enabled="
                f"{bool(pre_export_cfg.get('enabled', False))} "
                f"format={pre_export_cfg.get('format')} device={evaluation.get('device')}"
            )
            pre_export = self._prepare_evaluation_pre_export(
                YOLO,
                source_best_weights=source_best_weights,
                evaluation=evaluation,
                tracker=tracker,
            )
            evaluation_weights = Path(
                pre_export.get("evaluation_weights") or source_best_weights
            ).resolve()
            print(f"[evaluation] Loading evaluation weights: {evaluation_weights}")
            try:
                model = YOLO(
                    str(evaluation_weights), task=self.config.payload["model"]["task"]
                )
                precision_report = self._prepare_evaluation_precision(
                    model,
                    precision=precision,
                )
                precision_report["eval_mode"] = _set_evaluation_mode(model)
                tracker.mlflow.log_dict(precision_report, "evaluation/precision.json")
                tracker.mlflow.log_dict(pre_export, "evaluation/pre_export.json")

                execution_report = {
                    "requested_device": device_to_log_value(requested_device),
                    "device": device_to_log_value(evaluation.get("device")),
                    "batch": self._effective_evaluation_batch(evaluation),
                    "plots": bool(evaluation.get("plots", False)),
                    "fallback_used": fallback_used,
                    "fallback_reason": fallback_reason,
                    "precision": precision["precision"],
                }
                tracker.mlflow.log_dict(execution_report, "evaluation/execution.json")

                for split in evaluation.get("splits", ["val", "test"]):
                    split_dir = evaluation_root / str(split)
                    if split_dir.exists():
                        shutil.rmtree(split_dir)

                    validation_args = self._evaluation_validation_args(
                        evaluation,
                        split=str(split),
                        precision=precision,
                        output_root=evaluation_root,
                    )
                    print(
                        "[evaluation] Running split="
                        f"{split!r} on device={validation_args.get('device')} "
                        f"batch={validation_args.get('batch')} "
                        f"half={validation_args.get('half')}"
                    )
                    metrics = model.val(**validation_args)
                    reports[str(split)] = normalize_metrics(metrics)
                    tracker.log_metrics(reports[str(split)], prefix=f"{split}_")
                    actual_split_dir = Path(getattr(metrics, "save_dir", split_dir))
                    print(
                        "[evaluation] Writing annotated samples for split="
                        f"{split!r} into {actual_split_dir}"
                    )
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
                    "weights": str(source_best_weights),
                    "evaluation_weights": str(evaluation_weights),
                    "precision": precision_report,
                    "pre_export": pre_export,
                    "execution": execution_report,
                    "dataset_fingerprint": self._dataset_manifest()[
                        "dataset_fingerprint"
                    ],
                    "splits": reports,
                    "artifacts": artifact_reports,
                    "quality_gates": evaluation.get("quality_gates", {}),
                    "passed": True,
                }
                report_path = self._write_json("reports/evaluation.json", report)
                tracker.mlflow.log_artifact(str(report_path), artifact_path="reports")
                return report, report_path
            finally:
                if model is not None:
                    del model
                _release_accelerator_memory()

    def _evaluation_for_device(
        self,
        evaluation: dict[str, Any],
        *,
        device: Any,
        batch: int | None = None,
        plots: bool | None = None,
        cuda_oom_fallback: bool = False,
    ) -> dict[str, Any]:
        """Return an evaluation config pinned to one device.

        CPU fallback intentionally disables FP16 model casting and TensorRT engine
        pre-export because both can keep evaluation tied to CUDA memory.
        """
        result = copy.deepcopy(evaluation)
        result["device"] = device
        if batch is not None:
            result["batch"] = int(batch)
        if plots is not None:
            result["plots"] = bool(plots)

        pre_export = result.get("pre_export")
        if isinstance(pre_export, dict):
            pre_export = dict(pre_export)
            options = dict(pre_export.get("options") or {})
            options["device"] = device
            if batch is not None:
                options["batch"] = int(batch)
            pre_export["options"] = options
            result["pre_export"] = pre_export

        if is_directml_device(device):
            # DirectML is a PyTorch execution backend. Keep validation on the
            # DirectML torch device, but avoid ONNX/TensorRT pre-export because
            # those backends do not consume DirectML devices in this pipeline.
            result["precision"] = "fp32"
            result["half"] = False
            result["cast_model_to_half"] = False
            if isinstance(result.get("pre_export"), dict):
                pre_export = result["pre_export"]
                options = dict(pre_export.get("options") or {})
                options["device"] = "cpu"
                options["half"] = False
                if batch is not None:
                    options["batch"] = int(batch)
                pre_export["options"] = options
                pre_export["enabled"] = False
                pre_export["use_for_evaluation"] = False
                pre_export["disabled_reason"] = (
                    "directml_evaluation_uses_pytorch_backend"
                )
            return result

        if not _is_cpu_device(device):
            return result

        result["precision"] = "fp32"
        result["half"] = False
        result["cast_model_to_half"] = False
        result["batch"] = 1 if batch is None else int(batch)
        result["plots"] = False if plots is None else bool(plots)
        if isinstance(result.get("pre_export"), dict):
            pre_export = result["pre_export"]
            options = dict(pre_export.get("options") or {})
            options["device"] = "cpu"
            options["half"] = False
            options["batch"] = int(result["batch"])
            pre_export["options"] = options
            if str(pre_export.get("format", "")).lower() == "engine":
                pre_export["enabled"] = False
                pre_export["use_for_evaluation"] = False
                pre_export["disabled_reason"] = (
                    "cpu_fallback_skips_tensorrt_engine_pre_export"
                    if cuda_oom_fallback
                    else "cpu_device_skips_tensorrt_engine_pre_export"
                )
        return result

    def _effective_evaluation_batch(self, evaluation: dict[str, Any]) -> int:
        """Resolve evaluation batch to a positive integer for retry decisions."""
        value = evaluation.get("batch")
        if value is None:
            value = self.config.payload["training"].get("batch", 16)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            logger.warning(
                "Evaluation batch=%r is not an integer; using batch=1 for safe retry.",
                value,
            )
            return 1

    def _remove_partial_evaluation_pre_exports(self) -> None:
        """Remove partial optimized backends before retrying after CUDA OOM."""
        pre_export_root = self.artifacts_dir / "evaluation" / "pre_export"
        if pre_export_root.exists():
            shutil.rmtree(pre_export_root)
            print(
                "[evaluation] Removed partial evaluation pre-export artifacts: "
                f"{pre_export_root}"
            )

    def _evaluation_pre_export_enabled(self, evaluation: dict[str, Any]) -> bool:
        """Return true when evaluation is currently trying an optimized backend."""
        pre_export = evaluation.get("pre_export") or {}
        return bool(pre_export.get("enabled", False))

    def _evaluation_without_pre_export(
        self,
        evaluation: dict[str, Any],
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Disable evaluation pre-export while preserving the rest of the config."""
        result = copy.deepcopy(evaluation)
        pre_export = dict(result.get("pre_export") or {})
        pre_export["enabled"] = False
        pre_export["use_for_evaluation"] = False
        pre_export["disabled_reason"] = reason
        result["pre_export"] = pre_export
        return result

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
                requested_export_device = options.get("device")
                if requested_export_device is not None:
                    options["device"] = resolve_export_device(
                        requested_export_device,
                        purpose=f"YOLO export ({export_format})",
                    )
                if options.get("int8") and "data" not in options:
                    options["data"] = str(self.config.dataset_dir / "dataset.yaml")
                model = YOLO(str(best_weights), task=self.config.payload["model"]["task"])
                try:
                    print(f"exporting : {export_format} with options: {options}")
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
                        "requested_device": device_to_log_value(requested_export_device),
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
                        "requested_device": device_to_log_value(requested_export_device),
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
        raw_device = args.get("device", training.get("device", 0))
        args["device"] = resolve_device_for_ultralytics(
            raw_device,
            purpose="YOLO training",
        )
        if is_directml_device(raw_device):
            # Ultralytics AMP is CUDA-oriented. DirectML execution is more
            # reliable in FP32 unless a future torch-directml release provides
            # full AMP parity.
            if args.get("amp", False):
                print("[device] DirectML selected for training; disabling AMP.")
            args["amp"] = False
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

    def _evaluation_precision_config(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        """Resolve evaluation precision settings with backward-compatible aliases."""
        requested_precision = evaluation.get("precision")
        if requested_precision is None:
            precision = "fp16" if bool(evaluation.get("half", False)) else "fp32"
        else:
            precision = str(requested_precision).lower().strip()
        half = precision == "fp16"
        return {
            "precision": precision,
            "half": half,
            "cast_model_to_half": bool(evaluation.get("cast_model_to_half", half)),
        }

    def _evaluation_validation_args(
        self,
        evaluation: dict[str, Any],
        *,
        split: str,
        precision: dict[str, Any],
        output_root: Path,
    ) -> dict[str, Any]:
        """Build Ultralytics validation args from config without hidden constants."""
        args: dict[str, Any] = {
            "data": str(self.config.dataset_dir / "dataset.yaml"),
            "split": split,
            "imgsz": int(
                evaluation.get(
                    "imgsz", self.config.payload["training"].get("imgsz", 640)
                )
            ),
            "batch": evaluation.get(
                "batch", self.config.payload["training"].get("batch", 16)
            ),
            "device": evaluation.get(
                "device", self.config.payload["training"].get("device", 0)
            ),
            "conf": float(evaluation.get("conf", 0.001)),
            "iou": float(evaluation.get("iou", 0.7)),
            "plots": bool(evaluation.get("plots", False)),
            "save_json": bool(evaluation.get("save_json", True)),
            "project": str(output_root),
            "name": split,
            "exist_ok": True,
            "workers": int(evaluation.get("workers", 0)),
            "half": bool(precision["half"]),
        }
        if evaluation.get("max_det") is not None:
            args["max_det"] = int(evaluation["max_det"])
        return args

    def _prepare_evaluation_precision(
        self,
        model: Any,
        *,
        precision: dict[str, Any],
    ) -> dict[str, Any]:
        """Optionally cast YOLO's torch module to FP16 and report memory impact."""
        report = {
            **precision,
            "parameter_footprint_before": None,
            "parameter_footprint_after": None,
            "cast_status": "not_requested",
        }
        module = getattr(model, "model", None)
        if not precision["half"]:
            report["cast_status"] = "fp32_requested"
            return report
        if not precision["cast_model_to_half"]:
            report["cast_status"] = "half_validation_only"
            return report
        if module is None or not callable(getattr(module, "half", None)):
            report["cast_status"] = "skipped_no_torch_module"
            return report

        report["parameter_footprint_before"] = _parameter_footprint(module)
        module.half()
        report["parameter_footprint_after"] = _parameter_footprint(module)
        report["cast_status"] = "converted_to_fp16"
        before = report["parameter_footprint_before"] or {}
        after = report["parameter_footprint_after"] or {}
        if before.get("bytes") and after.get("bytes"):
            report["parameter_bytes_reduction_fraction"] = 1 - (
                after["bytes"] / before["bytes"]
            )
        logger.info(
            "Evaluation precision=%s half=%s cast_status=%s",
            report["precision"],
            report["half"],
            report["cast_status"],
        )
        return report

    def _prepare_evaluation_pre_export(
        self,
        YOLO: Any,
        *,
        source_best_weights: Path,
        evaluation: dict[str, Any],
        tracker: MLflowSession,
    ) -> dict[str, Any]:
        """Optionally export an optimized backend before evaluation.

        Supported formats are ONNX and TensorRT engine. ONNX is more portable
        for Kaggle/CI memory reduction; TensorRT remains available when the
        runtime has a compatible TensorRT installation.
        """
        pre_export = evaluation.get("pre_export", {})
        if not pre_export or not pre_export.get("enabled", False):
            return {
                "enabled": False,
                "evaluation_weights": None,
                "reason": "evaluation.pre_export.enabled is false",
            }

        export_format = str(pre_export.get("format", "engine")).lower()
        options = self._evaluation_pre_export_options(
            export_format,
            evaluation=evaluation,
            configured_options=dict(pre_export.get("options") or {}),
        )
        options.setdefault(
            "imgsz",
            int(
                evaluation.get(
                    "imgsz", self.config.payload["training"].get("imgsz", 640)
                )
            ),
        )
        options.setdefault("batch", evaluation.get("batch", 1))
        options.setdefault(
            "device",
            evaluation.get("device", self.config.payload["training"].get("device", 0)),
        )
        requested_export_device = options.get("device")
        options["device"] = resolve_export_device(
            requested_export_device,
            purpose=f"evaluation pre-export ({export_format})",
        )
        if export_format == "engine" and options.get("int8") and "data" not in options:
            options["data"] = str(self.config.dataset_dir / "dataset.yaml")

        destination_dir = self.artifacts_dir / "evaluation" / "pre_export" / export_format
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        continue_on_error = bool(pre_export.get("continue_on_error", False))
        engine_model = YOLO(str(source_best_weights), task=self.config.payload["model"]["task"])

        try:
            exported_value = engine_model.export(format=export_format, **options)
            exported_paths = _normalize_export_paths(exported_value)
            organized_paths = [
                _organize_export(path, destination_dir, source_best_weights)
                for path in exported_paths
            ]
            selected = next(
                (path for path in organized_paths if path.suffix == f".{export_format}"),
                organized_paths[0] if organized_paths else None,
            )
            result = {
                "enabled": True,
                "format": export_format,
                "status": "complete",
                "quantization_bits": _quantization_bits(options),
                "options": options,
                "requested_device": device_to_log_value(requested_export_device),
                "use_for_evaluation": bool(pre_export.get("use_for_evaluation", False)),
                "evaluation_weights": (
                    str(selected)
                    if selected is not None and pre_export.get("use_for_evaluation", False)
                    else None
                ),
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
                str(destination_dir),
                artifact_path=f"evaluation/pre_export/{export_format}",
            )
            return result
        except Exception as exc:
            result = {
                "enabled": True,
                "format": export_format,
                "status": "failed",
                "quantization_bits": _quantization_bits(options),
                "options": options,
                "requested_device": device_to_log_value(requested_export_device),
                "use_for_evaluation": False,
                "evaluation_weights": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if _is_cuda_runtime_import_error(exc):
                result["runtime_error_kind"] = "cuda_runtime_import_error"
                result["disabled_reason"] = (
                    "pre_export_backend_requires_missing_cuda_runtime"
                )
                print(
                    "[evaluation] Evaluation pre-export failed because the optimized "
                    "backend requires CUDA runtime libraries that are not available. "
                    "Skipping pre-export and using PyTorch weights."
                )
                logger.warning(
                    "Evaluation pre-export skipped due to CUDA runtime import error: %s",
                    exc,
                )
                return result
            if continue_on_error:
                logger.warning("Evaluation pre-export failed and will be skipped: %s", exc)
                return result
            raise RuntimeError(
                f"Failed to pre-export evaluation model as '{export_format}': {exc}"
            ) from exc
        finally:
            del engine_model
            _release_accelerator_memory()

    def _evaluation_pre_export_options(
        self,
        export_format: str,
        *,
        evaluation: dict[str, Any],
        configured_options: dict[str, Any],
    ) -> dict[str, Any]:
        """Return format-specific export options for evaluation pre-export."""
        options = dict(configured_options)
        precision = str(evaluation.get("precision", "fp32")).lower()
        if export_format == "onnx":
            options.setdefault("opset", 17)
            options.setdefault("dynamic", False)
            options.setdefault("simplify", True)
            options.setdefault("half", precision == "fp16")
            options.setdefault("nms", False)
            options.pop("int8", None)
            options.pop("workspace", None)
            return options
        if export_format == "engine":
            options.setdefault("half", precision == "fp16")
            options.setdefault("int8", False)
            options.setdefault("dynamic", False)
            options.setdefault("workspace", 4)
            options.setdefault("simplify", True)
            options.setdefault("nms", False)
            return options
        raise ValueError(
            f"Unsupported evaluation pre-export format: {export_format}"
        )

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
        precision = self._evaluation_precision_config(evaluation)
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
                half=bool(precision["half"]),
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
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
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


def _parameter_footprint(module: Any) -> dict[str, Any] | None:
    """Return parameter count, byte size, and dtype breakdown for a torch module."""
    parameters_fn = getattr(module, "parameters", None)
    if not callable(parameters_fn):
        return None
    total_parameters = 0
    total_bytes = 0
    dtype_counts: dict[str, int] = {}
    for parameter in parameters_fn():
        count = int(parameter.numel())
        total_parameters += count
        total_bytes += count * int(parameter.element_size())
        dtype_name = str(getattr(parameter, "dtype", "unknown"))
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + count
    return {
        "parameters": total_parameters,
        "bytes": total_bytes,
        "megabytes": round(total_bytes / (1024**2), 4),
        "dtypes": dtype_counts,
    }


def _set_evaluation_mode(model: Any) -> dict[str, Any]:
    """Put a PyTorch-backed YOLO model in eval mode when that backend exists.

    TensorRT/ONNX backends are not always torch modules. Calling ``YOLO.eval()``
    can delegate into ``model.model.eval()`` and fail when ``model.model`` is a
    string/path-like backend. Validation itself still runs through
    ``YOLO.val(...)``, so non-PyTorch backends should simply skip this step.
    """
    module = getattr(model, "model", None)
    eval_fn = getattr(module, "eval", None)
    if callable(eval_fn):
        eval_fn()
        return {
            "status": "torch_module_eval",
            "backend_type": type(module).__name__,
        }
    return {
        "status": "skipped_non_torch_backend",
        "backend_type": type(module).__name__ if module is not None else None,
    }


def _is_cpu_device(device: Any) -> bool:
    """Return true for the explicit Ultralytics CPU device selector."""
    return is_cpu_device(device)


def _next_smaller_evaluation_batch(batch: int) -> int:
    """Halve an evaluation batch conservatively while still reaching batch 1."""
    if batch <= 1:
        return 1
    return max(1, math.ceil(batch / 2))


def _is_cuda_runtime_import_error(exc: BaseException) -> bool:
    """Detect CUDA shared-library import failures from optional GPU backends."""
    seen: set[int] = set()
    current: BaseException | None = exc
    messages: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__

    text = " ".join(messages).lower()
    missing_library = any(
        token in text
        for token in (
            "cannot open shared object file",
            "could not load library",
            "failed to load",
            "dll load failed",
        )
    )
    cuda_library = any(
        token in text
        for token in (
            "libcudart",
            "libcublas",
            "libcufft",
            "libcurand",
            "libcusolver",
            "libcusparse",
            "libcudnn",
            "cudnn",
            "cuda runtime",
            "onnxruntime_pybind11_state",
        )
    )
    return missing_library and cuda_library


def _is_cuda_out_of_memory(exc: BaseException) -> bool:
    """Detect CUDA OOM even when it is wrapped by higher-level libraries."""
    seen: set[int] = set()
    current: BaseException | None = exc
    messages: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        try:
            if isinstance(current, torch.cuda.OutOfMemoryError):
                return True
        except AttributeError:
            pass
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__

    text = " ".join(messages).lower()
    memory_signal = "out of memory" in text or "oom" in text
    cuda_signal = "cuda" in text or "cudnn" in text or "gpu" in text
    return memory_signal and cuda_signal


def _accelerator_out_of_memory_kind(exc: BaseException) -> str | None:
    """Classify accelerator OOM errors for retry policy decisions."""
    if _is_cuda_out_of_memory(exc):
        return "cuda"
    if _is_directml_out_of_memory(exc):
        return "directml"
    return None


def _is_directml_out_of_memory(exc: BaseException) -> bool:
    """Detect DirectML OOM messages wrapped by PyTorch/Ultralytics."""
    seen: set[int] = set()
    current: BaseException | None = exc
    messages: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__

    text = " ".join(messages).lower()
    memory_signal = (
        "out of memory" in text
        or "oom" in text
        or "not enough memory" in text
        or "insufficient memory" in text
    )
    directml_signal = any(
        token in text
        for token in (
            "directml",
            "torch_directml",
            "torch-directml",
            "privateuseone",
            "dml",
            "d3d12",
            "dx12",
        )
    )
    return memory_signal and directml_signal


def _quantization_bits(options: dict[str, Any]) -> int:
    """Describe export precision in bits for artifact metadata."""
    if options.get("int8"):
        return 8
    if options.get("half"):
        return 16
    return 32


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
    try:
        import torch_directml

        empty_cache = getattr(torch_directml, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except Exception:
        # torch-directml is optional; cache release should never mask the real
        # training/evaluation error.
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
