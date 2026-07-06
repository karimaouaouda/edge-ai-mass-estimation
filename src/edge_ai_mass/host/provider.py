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
        # ``load`` means "the proxy can reach the host stage endpoint", not
        # "eagerly load the remote model".  The edge fallback chain should be:
        #
        #   local primary -> host primary endpoint -> local fallback
        #
        # Asking the health endpoint to load/smoke-test the remote primary here
        # can incorrectly mark the host as unavailable during edge preload,
        # especially for large depth checkpoints.  The real POST endpoint still
        # loads the host primary just-in-time before inference.
        self._health = self.client.stage_health(self.endpoint_stage, load=False)
        if self._health.get("status") not in {"ok", "ready"}:
            raise RuntimeError(
                f"Host stage {self.endpoint_stage!r} is not ready: {self._health}"
            )
        self._is_loaded = True

    def _forward(self, image: np.ndarray, **kwargs: Any) -> ModuleResult:
        primary_loaded = bool(self._health.get("primary_loaded", False))
        first_load_timeout = (
            self.client.load_timeout_seconds if not primary_loaded else None
        )
        result = self.client.run_stage(
            self.endpoint_stage,
            image,
            kwargs=kwargs,
            timeout_seconds=first_load_timeout,
        )
        self._health["primary_loaded"] = True
        result.metadata.setdefault("host_stage", self.endpoint_stage)
        result.metadata.setdefault("host_base_url", self.client.base_url)
        return result
