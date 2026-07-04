"""Depth Anything monocular depth estimation module.

Supports both the HuggingFace transformers pipeline (for prototyping) and
direct loading of the DepthAnything model (for optimised inference / TensorRT).

Returns a metric-scale depth map (H, W) in metres.
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from edge_ai_mass.modules.base import BaseModule, ModuleResult

logger = logging.getLogger(__name__)


class DepthAnythingModule(BaseModule):
    """Wrapper around Depth Anything V2 for monocular metric depth."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model_name: str = config.get(
            "model_name", "depth-anything/Depth-Anything-V2-Small-hf"
        )
        self.input_size: int = config.get("input_size", 384)
        self.device: str = _torch_device(config.get("device"))
        self.cache_dir = _resolve_model_dir(
            config.get("cache_dir") or config.get("model_cache_dir"),
            "depth",
            "huggingface",
        )
        self.local_files_only = _as_bool(config.get("local_files_only", False))
        self.debug_outputs = _as_bool(config.get("debug_outputs", False))
        self.debug_dir = _resolve_model_dir(config.get("debug_dir"), "debug", "depth")
        self._pipe: Any = None

    def load(self) -> None:
        from transformers import pipeline as hf_pipeline

        model_kwargs = dict(self.config.get("model_kwargs") or {})
        model_kwargs.setdefault("cache_dir", str(self.cache_dir))
        if self.local_files_only:
            model_kwargs.setdefault("local_files_only", True)

        logger.info(
            "Loading Depth Anything model: %s device=%s cache_dir=%s",
            self.model_name,
            self.device,
            self.cache_dir,
        )
        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=self.model_name,
            device=_transformers_device(self.device),
            model_kwargs=model_kwargs,
        )
        self._is_loaded = True

    def _forward(self, image: np.ndarray, **kwargs: Any) -> ModuleResult:
        from PIL import Image

        # Resize for speed while keeping aspect
        h, w = image.shape[:2]
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = self._pipe(pil_image)
        depth = np.array(result["depth"], dtype=np.float32)

        if self.debug_outputs:
            _write_debug_depth(
                depth,
                self.debug_dir,
                f"depth-anything_{self.model_name.replace('/', '_')}",
            )

        # Resize depth back to original resolution
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        return ModuleResult(
            data=depth,
            metadata={
                "model_name": self.model_name,
                "cache_dir": str(self.cache_dir),
                "device": self.device,
            },
        )


class MiDaSDepthModule(BaseModule):
    """MiDaS fallback — lighter and faster, but relative depth only."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model_type: str = config.get("model_type", "MiDaS_small")
        self.device: str = _torch_device(config.get("device"))
        self.torch_hub_dir = _resolve_model_dir(
            config.get("torch_hub_dir") or config.get("cache_dir"),
            "torch",
            "hub",
        )
        self.debug_outputs = _as_bool(config.get("debug_outputs", False))
        self.debug_dir = _resolve_model_dir(config.get("debug_dir"), "debug", "depth")
        self._model: Any = None
        self._transform: Any = None

    def load(self) -> None:
        self.torch_hub_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(self.torch_hub_dir))
        logger.info(
            "Loading MiDaS model: %s device=%s torch_hub_dir=%s",
            self.model_type,
            self.device,
            self.torch_hub_dir,
        )
        self._model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self._model.to(self.device).eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if "small" in self.model_type.lower():
            self._transform = midas_transforms.small_transform
        else:
            self._transform = midas_transforms.dpt_transform
        self._is_loaded = True

    @torch.no_grad()
    def _forward(self, image: np.ndarray, **kwargs: Any) -> ModuleResult:
        h, w = image.shape[:2]
        input_batch = self._transform(image).to(self.device)
        prediction = self._model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)
        if self.debug_outputs:
            _write_debug_depth(depth, self.debug_dir, f"midas_{self.model_type.replace('/', '_')}")
        return ModuleResult(
            data=depth,
            metadata={
                "model_type": self.model_type,
                "torch_hub_dir": str(self.torch_hub_dir),
                "device": self.device,
            },
        )


def _models_root() -> Path:
    root = Path(os.environ.get("EDGE_AI_MODELS_DIR", "models")).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def _resolve_model_dir(configured: Any, *default_parts: str) -> Path:
    if configured not in (None, ""):
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
    else:
        path = _models_root().joinpath(*default_parts)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _torch_device(configured: Any = None) -> str:
    device = str(configured or ("cuda" if torch.cuda.is_available() else "cpu")).strip()
    if device.lower().startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA depth device requested but CUDA is unavailable; using CPU")
        return "cpu"
    return device or "cpu"


def _transformers_device(device: str) -> int | str:
    normalized = str(device).strip().lower()
    if normalized in {"cpu", "-1"}:
        return -1
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA depth device requested but CUDA is unavailable; using CPU")
            return -1
        if ":" in normalized:
            try:
                return int(normalized.rsplit(":", 1)[1])
            except ValueError:
                return 0
        return 0
    return device


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _write_debug_depth(depth: np.ndarray, debug_dir: Path, name: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = debug_dir / f"debug_depth_{name}_{timestamp}.png"
    cv2.imwrite(str(debug_path), _depth_to_uint8(depth))


def _depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    min_val = float(np.min(arr[finite]))
    max_val = float(np.max(arr[finite]))
    if max_val <= min_val:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    normalized = (arr - min_val) / (max_val - min_val)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
