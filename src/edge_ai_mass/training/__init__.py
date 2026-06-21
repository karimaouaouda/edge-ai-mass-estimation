"""Reproducible training pipelines for edge-ai-mass model stages.

The training package deliberately keeps framework imports lazy.  Importing the
edge runtime CLI must not load PyTorch, Ultralytics, MLflow, or Optuna.
"""

from edge_ai_mass.training.pipeline import TrainingPipeline

__all__ = ["TrainingPipeline"]
