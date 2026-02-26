"""Unit tests for mass regression head."""

import torch

from edge_ai_mass.modules.mass.regression_estimator import MassRegressionHead


def test_forward_shape():
    model = MassRegressionHead(input_dim=32, hidden_dim=64, num_classes=9)
    features = torch.randn(4, 32)
    class_ids = torch.randint(0, 9, (4,))
    output = model(features, class_ids)
    assert output.shape == (4,)
    assert (output >= 0).all(), "Softplus should produce non-negative values"


def test_single_sample():
    model = MassRegressionHead(input_dim=10, hidden_dim=16, num_classes=3)
    features = torch.randn(1, 10)
    class_ids = torch.tensor([1])
    output = model(features, class_ids)
    assert output.shape == (1,)
