"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from edge_ai_mass.evaluation.metrics import iou, mae, mape, r_squared, rmse


def test_rmse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 3.1])
    assert rmse(y_true, y_pred) == pytest.approx(0.1, abs=1e-6)


def test_mape_zero_error():
    y = np.array([1.0, 2.0, 3.0])
    assert mape(y, y) == pytest.approx(0.0, abs=1e-4)


def test_r_squared_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == pytest.approx(1.0, abs=1e-6)


def test_mae_simple():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert mae(y_true, y_pred) == pytest.approx(0.5)


def test_iou_identical():
    box = np.array([0, 0, 10, 10])
    assert iou(box, box) == pytest.approx(1.0, abs=1e-6)


def test_iou_no_overlap():
    box_a = np.array([0, 0, 5, 5])
    box_b = np.array([10, 10, 20, 20])
    assert iou(box_a, box_b) == pytest.approx(0.0, abs=1e-6)


def test_iou_partial():
    box_a = np.array([0, 0, 10, 10])
    box_b = np.array([5, 5, 15, 15])
    # Intersection = 5*5 = 25, Union = 100+100-25 = 175
    assert iou(box_a, box_b) == pytest.approx(25 / 175, abs=1e-6)
