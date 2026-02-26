"""Base interface that every AI module must implement.

Keeping a uniform contract lets the pipeline swap models without touching
orchestration logic.  When you build a new model module, subclass ``BaseModule``
and implement ``load``, ``predict``, and optionally ``warmup``.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ModuleResult:
    """Standardised container returned by every module's ``predict``."""

    data: Any  # module-specific payload (detections, depth map, mass, …)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseModule(abc.ABC):
    """Abstract base for all AI modules (detection, depth, mass, …)."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights / engines into memory."""

    def warmup(self, input_shape: tuple[int, ...] = (480, 640, 3)) -> None:
        """Run a dummy forward pass to trigger JIT / TensorRT warm-up."""
        dummy = np.zeros(input_shape, dtype=np.uint8)
        self.predict(dummy)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _forward(self, image: np.ndarray, **kwargs: Any) -> Any:
        """Model-specific forward pass.  Return raw module payload."""

    def predict(self, image: np.ndarray, **kwargs: Any) -> ModuleResult:
        """Public entry point — wraps ``_forward`` with timing."""
        if not self._is_loaded:
            self.load()
        t0 = time.perf_counter()
        data = self._forward(image, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        return ModuleResult(data=data, latency_ms=elapsed)
