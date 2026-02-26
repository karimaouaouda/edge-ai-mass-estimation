"""Evaluation metrics for detection, depth, and mass estimation.

All functions accept numpy arrays and return plain Python floats so they are
easy to serialise / log.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Mass estimation metrics
# ------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ------------------------------------------------------------------
# Depth estimation metrics
# ------------------------------------------------------------------
def depth_abs_rel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Absolute relative error for depth maps."""
    valid = y_true > 0
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid]) / y_true[valid]))


def depth_delta_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 1.25
) -> float:
    """Fraction of pixels where max(pred/gt, gt/pred) < threshold."""
    valid = y_true > 0
    ratio = np.maximum(y_pred[valid] / y_true[valid], y_true[valid] / y_pred[valid])
    return float(np.mean(ratio < threshold))


# ------------------------------------------------------------------
# Detection metrics (thin wrapper — heavy lifting done by COCO / ultralytics)
# ------------------------------------------------------------------
def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / (union + 1e-8))
