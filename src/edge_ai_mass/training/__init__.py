"""Reproducible training pipelines for edge-ai-mass model stages.

The training package deliberately keeps framework imports lazy.  Importing the
edge runtime CLI must not load PyTorch, Ultralytics, MLflow, or Optuna.
"""

from edge_ai_mass.training.checkpoints import (
    ModelSource,
    load_model_or_checkpoint,
    resume_target_epochs,
    resolve_model_source,
)
from edge_ai_mass.training.downloads import TACO_ANNOTATIONS_URL, download_taco_dataset
from edge_ai_mass.training.pipeline import TrainingPipeline
from edge_ai_mass.training.publication import (
    TrainingOutputPublisher,
    restore_training_outputs,
)

__all__ = [
    "ModelSource",
    "TACO_ANNOTATIONS_URL",
    "TrainingPipeline",
    "TrainingOutputPublisher",
    "download_taco_dataset",
    "load_model_or_checkpoint",
    "resume_target_epochs",
    "restore_training_outputs",
    "resolve_model_source",
]
