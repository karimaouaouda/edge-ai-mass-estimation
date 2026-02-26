"""MobileNetV3-SSD lightweight detector — fallback for YOLO when latency is tight.

Uses torchvision's pre-trained MobileNetV3-SSD or a custom fine-tuned variant.
This module acts as the fast-but-less-accurate alternative in the detection
cascade.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from edge_ai_mass.modules.base import BaseModule
from edge_ai_mass.pipeline.pipeline import Detection

logger = logging.getLogger(__name__)


class MobileNetDetector(BaseModule):
    """MobileNetV3-based SSD detector (torchvision or custom checkpoint)."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model_path: str | None = config.get("model_path")
        self.conf_threshold: float = config.get("conf_threshold", 0.3)
        self.device: str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.class_map: dict[int, str] = config.get("class_map", {})
        self._model: Any = None

    def load(self) -> None:
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large

        logger.info("Loading MobileNetV3-SSD (device=%s)", self.device)
        self._model = ssdlite320_mobilenet_v3_large(pretrained=self.model_path is None)
        if self.model_path:
            state = torch.load(self.model_path, map_location=self.device)
            self._model.load_state_dict(state)
        self._model.to(self.device).eval()
        self._is_loaded = True

    @torch.no_grad()
    def _forward(self, image: np.ndarray, **kwargs: Any) -> list[Detection]:
        from torchvision import transforms

        transform = transforms.Compose([transforms.ToTensor()])
        tensor = transform(image).unsqueeze(0).to(self.device)
        preds = self._model(tensor)[0]

        detections: list[Detection] = []
        for i in range(len(preds["scores"])):
            score = float(preds["scores"][i])
            if score < self.conf_threshold:
                continue
            cls_id = int(preds["labels"][i])
            detections.append(
                Detection(
                    bbox=preds["boxes"][i].cpu().numpy(),
                    mask=None,
                    class_id=cls_id,
                    class_name=self.class_map.get(cls_id, f"class_{cls_id}"),
                    confidence=score,
                )
            )
        return detections
