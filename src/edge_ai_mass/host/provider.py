"""BaseModule adapter that executes a stage on a configured host server."""

from __future__ import annotations

from typing import Any

import numpy as np

from edge_ai_mass.host.client import HostInferenceClient
from edge_ai_mass.modules.base import BaseModule, ModuleResult


class HostStageModule(BaseModule):
    """Proxy one pipeline stage to a remote host FastAPI server.

    ``load()`` is a cheap health check, not an inference call.  The edge agent
    uses this when the local primary model cannot load or crashes at runtime.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.stage_name = str(config["stage_name"])
        self.endpoint_stage = str(config.get("endpoint_stage") or self.stage_name)
        self.client = HostInferenceClient(
            base_url=str(config["base_url"]),
            timeout_seconds=float(config.get("timeout_seconds", 10.0)),
            health_timeout_seconds=float(config.get("health_timeout_seconds", 2.0)),
            load_timeout_seconds=float(config.get("load_timeout_seconds", 120.0)),
        )
        self._health: dict[str, Any] = {}

    def load(self) -> None:
        self._health = self.client.stage_health(self.endpoint_stage, load=True)
        if self._health.get("status") not in {"ok", "ready"}:
            raise RuntimeError(
                f"Host stage {self.endpoint_stage!r} is not ready: {self._health}"
            )
        self._is_loaded = True

    def _forward(self, image: np.ndarray, **kwargs: Any) -> ModuleResult:
        result = self.client.run_stage(self.endpoint_stage, image, kwargs=kwargs)
        result.metadata.setdefault("host_stage", self.endpoint_stage)
        result.metadata.setdefault("host_base_url", self.client.base_url)
        return result
