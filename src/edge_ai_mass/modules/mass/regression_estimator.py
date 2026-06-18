"""Regression-based mass estimator — learns mass from visual + depth features.

A lightweight MLP that takes a feature vector (bbox area, convex-hull area,
depth statistics, category embedding) and outputs predicted mass in kg.

This module can be trained on a custom dataset of (features, ground-truth mass)
pairs collected with a scale.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from edge_ai_mass.modules.base import BaseModule, ModuleResult

logger = logging.getLogger(__name__)


class MassRegressionHead(nn.Module):
    """Small MLP for mass prediction from a fixed-size feature vector."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, num_classes: int = 9) -> None:
        super().__init__()
        self.category_embed = nn.Embedding(num_classes, 8)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # mass is always positive
        )

    def forward(self, features: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        cat_emb = self.category_embed(class_ids)
        x = torch.cat([features, cat_emb], dim=-1)
        return self.net(x).squeeze(-1)


class RegressionMassEstimator(BaseModule):
    """Learned mass estimation via MLP regression."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.checkpoint_path: str | None = config.get("checkpoint_path")
        self.input_dim: int = config.get("input_dim", 32)
        self.hidden_dim: int = config.get("hidden_dim", 64)
        self.num_classes: int = config.get("num_classes", 9)
        self.device: str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.require_checkpoint: bool = config.get("require_checkpoint", False)
        self._model: MassRegressionHead | None = None

    def load(self) -> None:
        self._model = MassRegressionHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
        )
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            logger.info("Loading mass regression weights from %s", self.checkpoint_path)
            state = torch.load(self.checkpoint_path, map_location=self.device)
            self._model.load_state_dict(state)
        else:
            if self.require_checkpoint:
                raise FileNotFoundError(
                    f"Mass regression checkpoint is required but was not found: {self.checkpoint_path}"
                )
            logger.warning("No checkpoint found — using untrained regression head")
        self._model.to(self.device).eval()
        self._is_loaded = True

    def _forward(self, image: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        features_dict: dict[str, Any] = kwargs["features"]
        feature_vec = self._extract_features(features_dict)
        class_id = int(features_dict.get("class_id", 0))
        if class_id < 0 or class_id >= self.num_classes:
            logger.warning("Class id %s is outside regression embedding range; using 0", class_id)
            class_id = 0

        with torch.no_grad():
            feat_t = torch.tensor(feature_vec, dtype=torch.float32).unsqueeze(0)
            feat_tensor = feat_t.to(self.device)
            cls_tensor = torch.tensor([class_id], dtype=torch.long).to(self.device)
            mass_kg = float(self._model(feat_tensor, cls_tensor).item())

        return ModuleResult(
            data={
                "mass_kg": mass_kg,
                "volume_m3": features_dict.get("volume_m3"),
                "volume_method": "geometry" if features_dict.get("volume_m3") is not None else "",
                "material": features_dict.get("material") or features_dict.get("class_name", "unknown"),
                "warnings": [],
            },
            metadata={"method": "regression"},
        )

    def _extract_features(self, f: dict[str, Any]) -> list[float]:
        """Build a fixed-length feature vector from the detection context."""
        bbox = f.get("bbox", np.zeros(4))
        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        area = w * h

        ds = f.get("depth_stats", {})
        depth_feats = [
            ds.get("mean", 0.0),
            ds.get("median", 0.0),
            ds.get("std", 0.0),
            ds.get("min", 0.0),
            ds.get("max", 0.0),
        ]

        geometry = f.get("geometry") or {}
        geometry_feats = [
            geometry.get("width_m", 0.0),
            geometry.get("height_m", 0.0),
            geometry.get("projected_area_m2", 0.0),
            geometry.get("mean_height_m", 0.0),
            geometry.get("max_height_m", 0.0),
            geometry.get("volume_m3", f.get("volume_m3") or 0.0),
        ]

        mask = f.get("mask")
        mask_area = float(np.sum(mask > 0)) if mask is not None else area
        mask_ratio = mask_area / max(area, 1.0)

        # Pad / truncate to input_dim
        raw = [w, h, area, mask_area, mask_ratio] + depth_feats + geometry_feats
        raw = raw[: self.input_dim]
        raw += [0.0] * max(0, self.input_dim - len(raw))
        return raw
