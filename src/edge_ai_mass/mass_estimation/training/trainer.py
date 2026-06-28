"""Train residual correction regressors and save registry-ready artifacts."""

from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from edge_ai_mass.mass_estimation.config import (
    MassEstimationConfig,
    MassEstimationConfigError,
)
from edge_ai_mass.mass_estimation.evaluation.metrics import regression_metrics
from edge_ai_mass.mass_estimation.io import read_table, write_json
from edge_ai_mass.mass_estimation.models import ResidualMassModel
from edge_ai_mass.mass_estimation.tracking import MassMLflowSession
from edge_ai_mass.training.state import PipelineState


class MassModelTrainer:
    """Train and register tabular residual mass-estimation models."""

    def __init__(self, config: MassEstimationConfig, state: PipelineState):
        self.config = config
        self.state = state
        self.artifacts_dir = config.artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> dict[str, Any]:
        schema = _read_json(self.config.processed_dir / "feature_schema.json")
        train_df = read_table(_split_path(self.config, "train"))
        val_df = read_table(_split_path(self.config, "val"))
        if train_df.empty:
            raise MassEstimationConfigError("Training split is empty")
        if val_df.empty:
            raise MassEstimationConfigError("Validation split is empty")

        feature_columns = list(schema["feature_columns"])
        numeric_columns = list(schema["numeric_columns"])
        categorical_columns = list(schema["categorical_columns"])
        mass_base_column = str(schema.get("mass_base_column", "mass_base_g"))
        epsilon_g = float(self.config.payload["evaluation"].get("small_mass_epsilon_g", 1.0))

        candidates = [
            candidate
            for candidate in self.config.payload["model"]["candidates"]
            if bool(candidate.get("enabled", True))
        ]
        if not candidates:
            raise MassEstimationConfigError("No enabled model candidates were configured")

        tracker = MassMLflowSession(self.config, self.state)
        candidate_results = []
        trained_models: dict[str, ResidualMassModel] = {}
        with tracker.run():
            tracker.log_params(self.config.payload["model"], prefix="model.")
            for candidate in candidates:
                model = self._train_candidate(
                    candidate,
                    schema=schema,
                    train_df=train_df,
                    mass_base_column=mass_base_column,
                )
                candidate_dir = self.artifacts_dir / "training" / candidate["name"]
                model_path = model.save(candidate_dir / "model.joblib")
                result = {
                    "name": model.name,
                    "type": model.model_type,
                    "model_path": str(model_path),
                    "train": _evaluate_frame(
                        model,
                        train_df,
                        mass_base_column=mass_base_column,
                        epsilon_g=epsilon_g,
                    ),
                    "val": _evaluate_frame(
                        model,
                        val_df,
                        mass_base_column=mass_base_column,
                        epsilon_g=epsilon_g,
                    ),
                    "params": copy.deepcopy(candidate.get("params", {})),
                }
                write_json(candidate_dir / "training_summary.json", result)
                if tracker.enabled:
                    tracker.mlflow.log_artifacts(
                        str(candidate_dir), artifact_path=f"training/{candidate['name']}"
                    )
                    tracker.log_metrics(
                        _flatten_candidate_metrics(result["val"]),
                        prefix=f"val_{candidate['name']}_",
                    )
                candidate_results.append(result)
                trained_models[model.name] = model

            selection = self.config.payload["training"].get("selection", {})
            best_result = _select_best_candidate(candidate_results, selection)
            best_model = trained_models[best_result["name"]]
            models_dir = self.artifacts_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = best_model.save(models_dir / "best_model.joblib")
            metadata = {
                "selected_model": best_result["name"],
                "selected_type": best_result["type"],
                "selection": selection,
                "feature_schema": schema,
                "config_digest": self.config.digest,
                "data_config_digest": self.config.data_digest,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "hybrid_formula": "mass_pred_g = mass_base_g + model(features)",
                "target_formula": "residual_g = real_mass_g - mass_base_g",
            }
            write_json(models_dir / "best_model_metadata.json", metadata)
            write_json(
                self.artifacts_dir / "reports" / "model_config_resolved.json",
                self.config.resolved_copy(),
            )
            summary = {
                "best_model": str(best_model_path),
                "best_model_metadata": str(models_dir / "best_model_metadata.json"),
                "selected_candidate": best_result,
                "candidates": candidate_results,
                "feature_schema": str(self.config.processed_dir / "feature_schema.json"),
                "model_config_resolved": str(
                    self.artifacts_dir / "reports" / "model_config_resolved.json"
                ),
            }
            summary_path = write_json(
                self.artifacts_dir / "reports" / "training_summary.json", summary
            )
            if tracker.enabled:
                tracker.mlflow.log_artifact(str(best_model_path), artifact_path="models")
                tracker.mlflow.log_artifact(
                    str(models_dir / "best_model_metadata.json"), artifact_path="models"
                )
                tracker.mlflow.log_artifact(str(summary_path), artifact_path="reports")

        self.state.update(
            "train",
            best_model=str(best_model_path),
            training=summary,
            selected_mass_model=best_result["name"],
        )
        return summary

    def register(self, *, force: bool = False) -> dict[str, Any]:
        registry = self.config.payload["tracking"].get("registry", {})
        if not registry.get("enabled", False):
            result = {"enabled": False, "reason": "tracking.registry.enabled is false"}
            self.state.update("register", registration=result)
            return result
        self.state.require("best_model")

        tracker = MassMLflowSession(self.config, self.state)
        if not tracker.enabled:
            raise MassEstimationConfigError(
                "Model registry requires tracking.enabled=true because it uses MLflow"
            )
        model_path = Path(self.state.data["best_model"])
        model_name = str(registry.get("name", "edge-ai-mass-residual-mass"))
        alias = str(registry.get("alias", "candidate"))
        artifact_path = str(registry.get("artifact_path", "model"))

        with tracker.run() as run:
            mlflow = tracker.mlflow
            pyfunc_model = _build_pyfunc_model(mlflow)
            log_model_args = {
                "python_model": pyfunc_model,
                "artifacts": {"model": str(model_path)},
                "pip_requirements": [
                    "pandas>=2.0",
                    "numpy>=1.24",
                    "scikit-learn>=1.3",
                    "joblib>=1.3",
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
                model_name, registered.version, "config_digest", self.config.digest
            )
            client.set_model_version_tag(
                model_name,
                registered.version,
                "data_config_digest",
                self.config.data_digest,
            )
            result = {
                "enabled": True,
                "name": model_name,
                "version": str(registered.version),
                "alias": alias,
                "model_uri": model_uri,
                "run_id": run.info.run_id,
                "force": force,
            }
            path = write_json(self.artifacts_dir / "reports" / "registration.json", result)
            mlflow.log_artifact(str(path), artifact_path="reports")
        self.state.update("register", registration=result)
        return result

    def _train_candidate(
        self,
        candidate: dict[str, Any],
        *,
        schema: dict[str, Any],
        train_df: Any,
        mass_base_column: str,
    ) -> ResidualMassModel:
        name = str(candidate["name"])
        model_type = str(candidate["type"])
        feature_columns = list(schema["feature_columns"])
        if model_type == "physics_only":
            estimator = None
        else:
            estimator = _build_estimator(
                model_type,
                params=dict(candidate.get("params", {})),
                numeric_columns=list(schema["numeric_columns"]),
                categorical_columns=list(schema["categorical_columns"]),
                seed=int(self.config.payload["training"].get("seed", 42)),
            )
            residual_target = np.asarray(
                train_df["real_mass_g"] - train_df[mass_base_column],
                dtype=float,
            )
            estimator.fit(train_df[feature_columns], residual_target)

        return ResidualMassModel(
            name=name,
            model_type=model_type,
            estimator=estimator,
            feature_columns=feature_columns,
            numeric_columns=list(schema["numeric_columns"]),
            categorical_columns=list(schema["categorical_columns"]),
            mass_base_column=mass_base_column,
            metadata={
                "candidate": copy.deepcopy(candidate),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "target": "real_mass_g - mass_base_g",
            },
        )


def _build_estimator(
    model_type: str,
    *,
    params: dict[str, Any],
    numeric_columns: list[str],
    categorical_columns: list[str],
    seed: int,
) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import HuberRegressor, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _one_hot_encoder(OneHotEncoder)),
        ]
    )
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical, categorical_columns))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if model_type == "ridge":
        estimator = Ridge(**params)
    elif model_type == "huber":
        estimator = HuberRegressor(max_iter=int(params.pop("max_iter", 1000)), **params)
    elif model_type == "random_forest":
        estimator = RandomForestRegressor(random_state=seed, **params)
    elif model_type == "mlp":
        estimator = MLPRegressor(random_state=seed, max_iter=1000, **params)
    else:
        raise MassEstimationConfigError(f"Unsupported learned model type: {model_type}")
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", estimator)])


def _one_hot_encoder(factory: Any) -> Any:
    try:
        return factory(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - scikit-learn <1.2 compatibility
        return factory(handle_unknown="ignore", sparse=False)


def _evaluate_frame(
    model: ResidualMassModel,
    frame: Any,
    *,
    mass_base_column: str,
    epsilon_g: float,
) -> dict[str, Any]:
    correction_true = np.asarray(frame["real_mass_g"] - frame[mass_base_column], dtype=float)
    correction_pred = model.predict_correction(frame)
    mass_pred = np.asarray(frame[mass_base_column], dtype=float) + correction_pred
    return {
        "correction": regression_metrics(
            correction_true, correction_pred, epsilon_g=epsilon_g
        ),
        "baseline": regression_metrics(
            frame["real_mass_g"], frame[mass_base_column], epsilon_g=epsilon_g
        ),
        "hybrid": regression_metrics(frame["real_mass_g"], mass_pred, epsilon_g=epsilon_g),
    }


def _select_best_candidate(
    candidate_results: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any]:
    split = str(selection.get("split", "val"))
    metric = str(selection.get("metric", "hybrid.mae_g"))
    mode = str(selection.get("mode", "minimize"))
    reverse = mode == "maximize"
    if mode not in {"minimize", "maximize"}:
        raise MassEstimationConfigError("training.selection.mode must be minimize or maximize")

    def score(result: dict[str, Any]) -> float:
        cursor: Any = result.get(split, {})
        for part in metric.split("."):
            cursor = cursor.get(part) if isinstance(cursor, dict) else None
        if cursor is None:
            return float("-inf") if reverse else float("inf")
        return float(cursor)

    return sorted(candidate_results, key=score, reverse=reverse)[0]


def _flatten_candidate_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for group, values in metrics.items():
        for name, value in values.items():
            if isinstance(value, (int, float)):
                flattened[f"{group}_{name}"] = float(value)
    return flattened


def _split_path(config: MassEstimationConfig, split: str) -> Path:
    for suffix in (".parquet", ".csv"):
        path = config.processed_dir / f"features_{split}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"features_{split}.csv/parquet not found under {config.processed_dir}. "
        "Run stage split first."
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_pyfunc_model(mlflow: Any) -> Any:
    class ResidualMassPyFuncModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context: Any) -> None:
            from edge_ai_mass.mass_estimation.models import ResidualMassModel

            self.model = ResidualMassModel.load(context.artifacts["model"])

        def predict(self, context: Any, model_input: Any, params: dict[str, Any] | None = None):
            import pandas as pd

            del context, params
            frame = pd.DataFrame(model_input).copy()
            correction = self.model.predict_correction(frame)
            mass = frame[self.model.mass_base_column].to_numpy(dtype=float) + correction
            return pd.DataFrame(
                {
                    "predicted_correction_g": correction,
                    "predicted_mass_g": mass,
                }
            )

    return ResidualMassPyFuncModel()
