"""A saved model wrapper for hybrid residual mass estimation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ResidualMassModel:
    """Predict residual correction and add it to the physics base mass.

    The wrapped estimator learns:

        correction_g = real_mass_g - mass_base_g

    At inference time, final mass is reconstructed as:

        predicted_mass_g = mass_base_g + predicted_correction_g
    """

    name: str
    model_type: str
    estimator: Any
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    mass_base_column: str = "mass_base_g"
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_correction(self, frame: Any) -> np.ndarray:
        if self.model_type == "physics_only" or self.estimator is None:
            return np.zeros(len(frame), dtype=float)
        features = frame[self.feature_columns].copy()
        with warnings.catch_warnings():
            if self.model_type == "lightgbm":
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "X does not have valid feature names, but LGBMRegressor "
                        "was fitted with feature names"
                    ),
                    category=UserWarning,
                )
            return np.asarray(self.estimator.predict(features), dtype=float)

    def predict_mass(self, frame: Any) -> np.ndarray:
        base_mass = np.asarray(frame[self.mass_base_column], dtype=float)
        return base_mass + self.predict_correction(frame)

    def save(self, path: str | Path) -> Path:
        joblib = _require_joblib()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "ResidualMassModel":
        joblib = _require_joblib()
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Model artifact is not a ResidualMassModel: {path}")
        return model


def _require_joblib():
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            'joblib is required for mass model artifacts. Install with: pip install -e ".[mlops]"'
        ) from exc
    return joblib
