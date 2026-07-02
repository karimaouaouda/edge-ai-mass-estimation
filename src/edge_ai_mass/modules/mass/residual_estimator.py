"""Residual tabular mass estimator for the edge inference cascade.

This module loads the governed mass-estimation artifact produced by
`edge_ai_mass.mass_estimation`. It reconstructs the same tabular feature row at
runtime, predicts the learned correction, and adds it to the physics base mass.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from edge_ai_mass.mass_estimation.models.residual import ResidualMassModel
from edge_ai_mass.mass_estimation.runtime_features import build_runtime_mass_feature_row
from edge_ai_mass.modules.base import BaseModule, ModuleResult

logger = logging.getLogger(__name__)


class ResidualMassEstimator(BaseModule):
    """Use the trained residual mass model after detection, depth, and geometry.

    Expected input is `kwargs["features"]`, assembled by the pipeline from the
    first inference stages. The output includes `mass_features`, which is the
    exact tabular row sent to the residual model.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model_path = Path(str(config.get("model_path") or ""))
        self.require_model = bool(config.get("require_model", True))
        self.clamp_nonnegative = bool(config.get("clamp_nonnegative_mass", True))
        self.model: ResidualMassModel | None = None

    def load(self) -> None:
        if not str(self.model_path):
            if self.require_model:
                raise FileNotFoundError("Residual mass model path is not configured")
            self._is_loaded = True
            return
        if not self.model_path.is_file():
            if self.require_model:
                raise FileNotFoundError(
                    f"Residual mass model artifact not found: {self.model_path}"
                )
            logger.warning("Residual mass model artifact not found: %s", self.model_path)
            self._is_loaded = True
            return

        self.model = ResidualMassModel.load(self.model_path)
        self._is_loaded = True

    def _forward(self, image: np.ndarray, **kwargs: Any) -> ModuleResult:
        features = kwargs.get("features")
        if not isinstance(features, dict):
            raise ValueError("ResidualMassEstimator requires kwargs['features'] as a dict")
        if self.model is None:
            if self.require_model:
                raise RuntimeError("Residual mass model is not loaded")
            raise RuntimeError("Residual mass model is disabled because no artifact was loaded")

        built = build_runtime_mass_feature_row(features, self.config)
        mass_features, alignment_warnings = self._align_features_for_model(built.row)
        warnings = [*built.warnings, *alignment_warnings]

        pd = _require_pandas()
        frame = pd.DataFrame([mass_features])
        predicted_correction_g = float(self.model.predict_correction(frame)[0])
        predicted_mass_g_raw = float(self.model.predict_mass(frame)[0])
        predicted_mass_g = predicted_mass_g_raw
        if self.clamp_nonnegative and predicted_mass_g < 0:
            predicted_mass_g = 0.0
            warnings.append("mass_prediction_clamped_nonnegative")

        return ModuleResult(
            data={
                "mass_kg": predicted_mass_g / 1000.0,
                "mass_g": predicted_mass_g,
                "predicted_mass_g": predicted_mass_g,
                "predicted_mass_g_raw": predicted_mass_g_raw,
                "predicted_correction_g": predicted_correction_g,
                "mass_base_g": float(mass_features[self.model.mass_base_column]),
                "volume_m3": float(mass_features["estimated_volume_m3"]),
                "volume_method": str(mass_features.get("volume_source") or "runtime_features"),
                "material": str(mass_features.get("material") or "other"),
                "density_used": float(mass_features.get("effective_density_kg_m3", 0.0)),
                "mass_features": mass_features,
                "warnings": warnings,
            },
            metadata={
                "method": "residual_mass_model",
                "model_name": self.model.name,
                "model_type": self.model.model_type,
                "model_path": str(self.model_path),
                "feature_count": len(self.model.feature_columns),
            },
        )

    def _align_features_for_model(self, row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        assert self.model is not None
        warnings: list[str] = []
        aligned = {str(key): _jsonable(value) for key, value in row.items()}
        categorical = set(self.model.categorical_columns)
        required_columns = [*self.model.feature_columns]
        if self.model.mass_base_column not in required_columns:
            required_columns.append(self.model.mass_base_column)

        missing = [column for column in required_columns if column not in aligned]
        for column in missing:
            aligned[column] = "unknown" if column in categorical else 0.0
        if missing:
            warnings.append(f"mass_features_filled_missing_columns:{','.join(missing)}")

        for column in required_columns:
            if column in categorical:
                value = aligned.get(column)
                aligned[column] = "unknown" if value in (None, "") else str(value)
                continue
            aligned[column] = _finite_float(aligned.get(column), default=0.0)
        return aligned, warnings


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pandas is required for residual mass inference. "
            'Install with: pip install -e ".[mlops]"'
        ) from exc
    return pd


def _finite_float(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
