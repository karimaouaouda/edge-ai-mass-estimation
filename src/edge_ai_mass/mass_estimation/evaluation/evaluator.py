"""Evaluation stage for physics baseline and hybrid residual models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from edge_ai_mass.mass_estimation.config import MassEstimationConfig
from edge_ai_mass.mass_estimation.evaluation.metrics import (
    per_class_regression_metrics,
    regression_metrics,
)
from edge_ai_mass.mass_estimation.io import read_table, write_json, write_table
from edge_ai_mass.mass_estimation.models import ResidualMassModel
from edge_ai_mass.mass_estimation.tracking import MassMLflowSession
from edge_ai_mass.training.state import PipelineState


class MassModelEvaluator:
    """Evaluate final mass predictions, residual correction, and baseline."""

    def __init__(self, config: MassEstimationConfig, state: PipelineState):
        self.config = config
        self.state = state
        self.artifacts_dir = config.artifacts_dir

    def evaluate(self) -> dict[str, Any]:
        self.state.require("best_model")
        model = ResidualMassModel.load(self.state.data["best_model"])
        evaluation_cfg = self.config.payload["evaluation"]
        splits = [str(split) for split in evaluation_cfg.get("splits", ["val", "test"])]
        epsilon_g = float(evaluation_cfg.get("small_mass_epsilon_g", 1.0))
        tracker = MassMLflowSession(self.config, self.state)

        split_reports: dict[str, Any] = {}
        prediction_frames = []
        per_class_frames = []
        with tracker.run():
            for split in splits:
                split_df = read_table(_split_path(self.config, split))
                if split_df.empty:
                    raise RuntimeError(f"Cannot evaluate empty split: {split}")
                report, predictions, per_class = self._evaluate_split(
                    model,
                    split_df,
                    split=split,
                    epsilon_g=epsilon_g,
                )
                split_reports[split] = report
                prediction_frames.append(predictions)
                per_class_frames.append(per_class)
                if tracker.enabled:
                    tracker.log_metrics(
                        _flatten_metrics(report["hybrid"]),
                        prefix=f"{split}_hybrid_",
                    )
                    tracker.log_metrics(
                        _flatten_metrics(report["baseline"]),
                        prefix=f"{split}_baseline_",
                    )
                    tracker.mlflow.log_artifacts(
                        str(self.artifacts_dir / "evaluation" / split),
                        artifact_path=f"evaluation/{split}",
                    )

            pd = __import__("pandas")
            all_predictions = pd.concat(prediction_frames, ignore_index=True)
            all_per_class = pd.concat(per_class_frames, ignore_index=True)
            root_eval_dir = self.artifacts_dir / "evaluation"
            predictions_path = write_table(all_predictions, root_eval_dir / "predictions.csv")
            per_class_path = write_table(all_per_class, root_eval_dir / "per_class_metrics.csv")
            error_analysis_path = write_table(
                _error_analysis(all_predictions),
                root_eval_dir / "error_analysis.csv",
            )
            report = {
                "model": self.state.data["best_model"],
                "splits": split_reports,
                "predictions": str(predictions_path),
                "per_class_metrics": str(per_class_path),
                "error_analysis": str(error_analysis_path),
                "small_mass_epsilon_g": epsilon_g,
            }
            metrics_path = write_json(root_eval_dir / "metrics.json", report)
            reports_path = write_json(self.artifacts_dir / "reports" / "evaluation.json", report)
            if tracker.enabled:
                tracker.mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
                tracker.mlflow.log_artifact(str(reports_path), artifact_path="reports")
                tracker.mlflow.log_artifact(str(predictions_path), artifact_path="evaluation")
                tracker.mlflow.log_artifact(str(per_class_path), artifact_path="evaluation")
                tracker.mlflow.log_artifact(
                    str(error_analysis_path), artifact_path="evaluation"
                )

        self.state.update("evaluate", evaluation=report, evaluation_report=str(reports_path))
        return report

    def _evaluate_split(
        self,
        model: ResidualMassModel,
        frame: Any,
        *,
        split: str,
        epsilon_g: float,
    ) -> tuple[dict[str, Any], Any, Any]:
        mass_base_column = model.mass_base_column
        predicted_correction = model.predict_correction(frame)
        predicted_mass = np.asarray(frame[mass_base_column], dtype=float) + predicted_correction
        residual_target = np.asarray(frame["real_mass_g"] - frame[mass_base_column], dtype=float)

        predictions = frame[
            [
                column
                for column in ("sample_id", "object_id", "class_name", "material")
                if column in frame.columns
            ]
        ].copy()
        predictions["split"] = split
        predictions["real_mass_g"] = frame["real_mass_g"].to_numpy(dtype=float)
        predictions["mass_base_g"] = frame[mass_base_column].to_numpy(dtype=float)
        predictions["residual_target_g"] = residual_target
        predictions["predicted_correction_g"] = predicted_correction
        predictions["predicted_mass_g"] = predicted_mass
        predictions["baseline_error_g"] = predictions["mass_base_g"] - predictions["real_mass_g"]
        predictions["hybrid_error_g"] = predictions["predicted_mass_g"] - predictions["real_mass_g"]
        predictions["baseline_abs_error_g"] = np.abs(predictions["baseline_error_g"])
        predictions["hybrid_abs_error_g"] = np.abs(predictions["hybrid_error_g"])
        predictions["abs_error_improvement_g"] = (
            predictions["baseline_abs_error_g"] - predictions["hybrid_abs_error_g"]
        )

        split_dir = self.artifacts_dir / "evaluation" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = write_table(predictions, split_dir / "predictions.csv")

        baseline_metrics = regression_metrics(
            predictions["real_mass_g"],
            predictions["mass_base_g"],
            epsilon_g=epsilon_g,
        )
        hybrid_metrics = regression_metrics(
            predictions["real_mass_g"],
            predictions["predicted_mass_g"],
            epsilon_g=epsilon_g,
        )
        correction_metrics = regression_metrics(
            predictions["residual_target_g"],
            predictions["predicted_correction_g"],
            epsilon_g=epsilon_g,
        )
        per_class_baseline = per_class_regression_metrics(
            predictions,
            true_column="real_mass_g",
            pred_column="mass_base_g",
            epsilon_g=epsilon_g,
        )
        per_class_baseline["model"] = "physics_baseline"
        per_class_hybrid = per_class_regression_metrics(
            predictions,
            true_column="real_mass_g",
            pred_column="predicted_mass_g",
            epsilon_g=epsilon_g,
        )
        per_class_hybrid["model"] = "hybrid_residual"
        pd = __import__("pandas")
        per_class = pd.concat([per_class_baseline, per_class_hybrid], ignore_index=True)
        per_class["split"] = split
        per_class_path = write_table(per_class, split_dir / "per_class_metrics.csv")

        comparison = _baseline_comparison(baseline_metrics, hybrid_metrics)
        plots = _write_plots(predictions, per_class, split_dir / "plots")
        report = {
            "rows": int(len(predictions)),
            "baseline": baseline_metrics,
            "correction": correction_metrics,
            "hybrid": hybrid_metrics,
            "comparison": comparison,
            "predictions": str(predictions_path),
            "per_class_metrics": str(per_class_path),
            "plots": plots,
        }
        write_json(split_dir / "metrics.json", report)
        return report, predictions, per_class


def _split_path(config: MassEstimationConfig, split: str) -> Path:
    for suffix in (".parquet", ".csv"):
        path = config.processed_dir / f"features_{split}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"features_{split}.csv/parquet not found under {config.processed_dir}. "
        "Run stage split first."
    )


def _baseline_comparison(
    baseline: dict[str, float | None],
    hybrid: dict[str, float | None],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in ("mae_g", "rmse_g", "median_absolute_error_g", "mape_percent", "smape_percent"):
        base_value = baseline.get(name)
        hybrid_value = hybrid.get(name)
        if base_value is None or hybrid_value is None:
            result[f"{name}_improvement"] = None
            result[f"{name}_relative_improvement_percent"] = None
            continue
        result[f"{name}_improvement"] = float(base_value - hybrid_value)
        result[f"{name}_relative_improvement_percent"] = (
            float((base_value - hybrid_value) / base_value * 100.0)
            if abs(base_value) > 1e-12
            else None
        )
    return result


def _error_analysis(predictions: Any) -> Any:
    analysis = predictions.copy()
    denominator = np.maximum(np.abs(analysis["real_mass_g"].to_numpy(dtype=float)), 1.0)
    analysis["baseline_abs_percent_error"] = (
        analysis["baseline_abs_error_g"].to_numpy(dtype=float) / denominator * 100.0
    )
    analysis["hybrid_abs_percent_error"] = (
        analysis["hybrid_abs_error_g"].to_numpy(dtype=float) / denominator * 100.0
    )
    analysis["hybrid_improved"] = (
        analysis["hybrid_abs_error_g"] < analysis["baseline_abs_error_g"]
    )
    return analysis.sort_values("hybrid_abs_error_g", ascending=False).reset_index(drop=True)


def _write_plots(predictions: Any, per_class: Any, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"enabled": True, "files": {}}
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        manifest["enabled"] = False
        manifest["reason"] = "matplotlib is not installed"
        write_json(output_dir / "plot_manifest.json", manifest)
        return manifest

    plotters = {
        "predicted_vs_real": lambda: _plot_predicted_vs_real(plt, predictions),
        "residual_error_distribution": lambda: _plot_error_distribution(plt, predictions),
        "per_class_mae": lambda: _plot_per_class_mae(plt, per_class),
        "baseline_vs_hybrid": lambda: _plot_baseline_vs_hybrid(plt, predictions),
        "error_vs_true_mass": lambda: _plot_error_vs_true_mass(plt, predictions),
    }
    for name, plotter in plotters.items():
        fig = plotter()
        path = output_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        manifest["files"][name] = str(path)
    write_json(output_dir / "plot_manifest.json", manifest)
    return manifest


def _plot_predicted_vs_real(plt: Any, predictions: Any) -> Any:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(predictions["real_mass_g"], predictions["mass_base_g"], label="physics base", alpha=0.75)
    ax.scatter(predictions["real_mass_g"], predictions["predicted_mass_g"], label="hybrid", alpha=0.75)
    low = float(min(predictions["real_mass_g"].min(), predictions["predicted_mass_g"].min()))
    high = float(max(predictions["real_mass_g"].max(), predictions["predicted_mass_g"].max()))
    ax.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Measured mass (g)")
    ax.set_ylabel("Predicted mass (g)")
    ax.legend()
    ax.set_title("Predicted vs measured mass")
    return fig


def _plot_error_distribution(plt: Any, predictions: Any) -> Any:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(predictions["hybrid_error_g"], bins=min(20, max(5, len(predictions) // 2)))
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Hybrid error (g)")
    ax.set_ylabel("Objects")
    ax.set_title("Residual error distribution")
    return fig


def _plot_per_class_mae(plt: Any, per_class: Any) -> Any:
    hybrid = per_class[per_class["model"] == "hybrid_residual"].sort_values("class_name")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(hybrid["class_name"].astype(str), hybrid["mae_g"])
    ax.set_ylabel("MAE (g)")
    ax.set_title("Hybrid MAE by class")
    ax.tick_params(axis="x", rotation=30)
    return fig


def _plot_baseline_vs_hybrid(plt: Any, predictions: Any) -> Any:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Physics base", "Hybrid"]
    values = [
        float(predictions["baseline_abs_error_g"].mean()),
        float(predictions["hybrid_abs_error_g"].mean()),
    ]
    ax.bar(labels, values)
    ax.set_ylabel("MAE (g)")
    ax.set_title("Baseline vs hybrid MAE")
    return fig


def _plot_error_vs_true_mass(plt: Any, predictions: Any) -> Any:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(predictions["real_mass_g"], predictions["hybrid_abs_error_g"], alpha=0.75)
    ax.set_xlabel("Measured mass (g)")
    ax.set_ylabel("Hybrid absolute error (g)")
    ax.set_title("Error versus true mass")
    return fig


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
