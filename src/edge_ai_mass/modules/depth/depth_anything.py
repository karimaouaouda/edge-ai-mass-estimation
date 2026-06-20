"""Depth Anything monocular depth estimation module.

Supports both the HuggingFace transformers pipeline (for prototyping) and
direct loading of the DepthAnything model (for optimised inference / TensorRT).

Returns a metric-scale depth map (H, W) in metres.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import cv2
import numpy as np
import torch

from edge_ai_mass.modules.base import BaseModule

logger = logging.getLogger(__name__)


class DepthAnythingModule(BaseModule):
    """Wrapper around Depth Anything V2 for monocular metric depth."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model_name: str = config.get(
            "model_name", "depth-anything/Depth-Anything-V2-Small-hf"
        )
        self.input_size: int = config.get("input_size", 384)
        self.device: str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._pipe: Any = None

    def load(self) -> None:
        from transformers import pipeline as hf_pipeline

        logger.info("Loading Depth Anything model: %s", self.model_name)
        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=self.model_name,
            device=self.device,
        )
        self._is_loaded = True

    def _forward(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        from PIL import Image

        # Resize for speed while keeping aspect
        h, w = image.shape[:2]
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = self._pipe(pil_image)
        depth = np.array(result["depth"], dtype=np.float32)

        # save to specific path for debugging as depth image with model name and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = f"debug_depth_{self.model_name.replace('/', '_')}_{timestamp}.png"
        cv2.imwrite(debug_path, (depth / np.max(depth) * 255).astype(np.uint8))

        # Resize depth back to original resolution
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth


class MiDaSDepthModule(BaseModule):
    """MiDaS fallback — lighter and faster, but relative depth only."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model_type: str = config.get("model_type", "MiDaS_small")
        self.device: str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._model: Any = None
        self._transform: Any = None

    def load(self) -> None:
        logger.info("Loading MiDaS model: %s", self.model_type)
        self._model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self._model.to(self.device).eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if "small" in self.model_type.lower():
            self._transform = midas_transforms.small_transform
        else:
            self._transform = midas_transforms.dpt_transform
        self._is_loaded = True

    @torch.no_grad()
    def _forward(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        h, w = image.shape[:2]
        input_batch = self._transform(image).to(self.device)
        prediction = self._model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        #save as depth image with model name and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = f"debug_depth_{self.model_type.replace('/', '_')}_{timestamp}.png"
        cv2.imwrite(debug_path, (prediction.cpu().numpy() / np.max(prediction.cpu().numpy()) * 255).astype(np.uint8))
        return prediction.cpu().numpy()
