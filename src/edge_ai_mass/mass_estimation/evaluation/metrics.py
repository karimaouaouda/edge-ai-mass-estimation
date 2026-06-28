"""Metrics for physics baseline and hybrid mass-estimation evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def regression_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    epsilon_g: float = 1.0,
) -> dict[str, float | None]:
    """Return robust scalar metrics in grams with small-mass safeguards."""

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.size == 0:
        return {
            "mae_g": None,
            "rmse_g": None,
            "median_absolute_error_g": None,
            "r2": None,
            "mape_percent": None,
            "smape_percent": None,
        }
    error = true - pred
    abs_error = np.abs(error)
    denominator = np.maximum(np.abs(true), float(epsilon_g))
    smape_denominator = np.maximum((np.abs(true) + np.abs(pred)) / 2.0, float(epsilon_g))
    return {
        "mae_g": float(np.mean(abs_error)),
        "rmse_g": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error_g": float(np.median(abs_error)),
        "r2": _safe_r2(true, pred),
        "mape_percent": float(np.mean(abs_error / denominator) * 100.0),
        "smape_percent": float(np.mean(abs_error / smape_denominator) * 100.0),
    }


def per_class_regression_metrics(
    frame: Any,
    *,
    true_column: str,
    pred_column: str,
    class_column: str = "class_name",
    epsilon_g: float = 1.0,
) -> Any:
    rows = []
    for class_name, subset in frame.groupby(class_column, dropna=False):
        metrics = regression_metrics(
            subset[true_column],
            subset[pred_column],
            epsilon_g=epsilon_g,
        )
        rows.append(
            {
                "class_name": class_name,
                "n": int(len(subset)),
                **metrics,
            }
        )
    pd = __import__("pandas")
    return pd.DataFrame(rows).sort_values("class_name").reset_index(drop=True)


def _safe_r2(true: np.ndarray, pred: np.ndarray) -> float | None:
    if true.size < 2:
        return None
    total = float(np.sum((true - np.mean(true)) ** 2))
    if total <= 0.0:
        return None
    residual = float(np.sum((true - pred) ** 2))
    return 1.0 - residual / total
