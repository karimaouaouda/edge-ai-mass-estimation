"""HTTP client used by the edge agent to call host-executed stages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from edge_ai_mass.modules.base import ModuleResult
from edge_ai_mass.host.serialization import (
    decode_value,
    module_result_from_payload,
    pipeline_request_payload,
    stage_request_payload,
)
from edge_ai_mass.pipeline.pipeline import PipelineResult


class HostInferenceError(RuntimeError):
    """Raised when the host inference provider is unavailable or returns an error."""


@dataclass(frozen=True, slots=True)
class HostInferenceClient:
    """Small JSON client for the host FastAPI inference server."""

    base_url: str
    timeout_seconds: float = 10.0
    health_timeout_seconds: float = 2.0
    load_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    def health(self) -> dict[str, Any]:
        """Return server-level health."""

        return self._request_json("GET", "/health", timeout=self.health_timeout_seconds)

    def stage_health(self, stage_name: str, *, load: bool = True) -> dict[str, Any]:
        """Return stage health, optionally asking the host to load the stage primary."""

        query = urlencode({"load": str(bool(load)).lower()})
        return self._request_json(
            "GET",
            f"/v1/stages/{stage_name}/health?{query}",
            # ``load=true`` performs a real host-side primary load and smoke test.
            # That can include first-time Hugging Face / Torch Hub downloads, so
            # keep the cheap ping timeout separate from the model-load timeout.
            timeout=self.load_timeout_seconds if load else self.health_timeout_seconds,
        )

    def run_stage(
        self,
        stage_name: str,
        image: np.ndarray,
        *,
        kwargs: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> ModuleResult:
        """Run one host stage and restore its ``ModuleResult``."""

        payload = stage_request_payload(image, kwargs)
        response = self._request_json(
            "POST",
            f"/v1/stages/{stage_name}",
            payload=payload,
            timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds,
        )
        return module_result_from_payload(response)

    def run_pipeline(
        self,
        image: np.ndarray,
        *,
        include_depth_maps: bool = False,
    ) -> PipelineResult:
        """Run the full host pipeline and restore the ``PipelineResult``."""

        payload = pipeline_request_payload(image, include_depth_maps=include_depth_maps)
        response = self._request_json(
            "POST",
            "/v1/pipeline",
            payload=payload,
            timeout=self.timeout_seconds,
        )
        result = decode_value(response.get("result"))
        if not isinstance(result, PipelineResult):
            raise HostInferenceError("Host pipeline response did not contain a PipelineResult")
        return result

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise HostInferenceError(
                f"Host inference HTTP {exc.code} for {method} {path}: {detail}"
            ) from exc
        except URLError as exc:
            raise HostInferenceError(
                f"Host inference connection failed for {method} {path}: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise HostInferenceError(
                f"Host inference timed out after {timeout:.1f}s for {method} {path}"
            ) from exc

        try:
            decoded = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise HostInferenceError("Host inference returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise HostInferenceError("Host inference returned a non-object JSON payload")
        return decoded
