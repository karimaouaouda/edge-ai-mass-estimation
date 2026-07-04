"""Contracts for host-executed stage fallback."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from edge_ai_mass.agent.config import AgentConfig
from edge_ai_mass.host.client import HostInferenceClient
from edge_ai_mass.host.serialization import (
    decode_value,
    module_result_from_payload,
    module_result_to_payload,
    stage_request_payload,
)
from edge_ai_mass.host.server import prepare_host_model_environment
from edge_ai_mass.modules.base import BaseModule, ModuleResult
from edge_ai_mass.pipeline.pipeline import Detection, Stage


class FakeModule(BaseModule):
    def __init__(
        self,
        *,
        name: str,
        data: Any = None,
        fail_load: bool = False,
        fail_predict: bool = False,
    ) -> None:
        super().__init__({})
        self.name = name
        self.data = data if data is not None else name
        self.fail_load = fail_load
        self.fail_predict = fail_predict
        self.load_calls = 0
        self.predict_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        if self.fail_load:
            raise RuntimeError(f"{self.name} load failed")
        self._is_loaded = True

    def _forward(self, image: np.ndarray, **kwargs: Any) -> ModuleResult:
        self.predict_calls += 1
        if self.fail_predict:
            raise RuntimeError(f"{self.name} predict failed")
        return ModuleResult(data=self.data, metadata={"module": self.name})


def test_stage_tries_host_primary_before_edge_fallback_after_primary_failure():
    primary = FakeModule(name="primary", fail_predict=True)
    host = FakeModule(name="host", data={"ok": True})
    fallback = FakeModule(name="fallback", data={"fallback": True})
    stage = Stage("depth", primary, fallback=fallback, host_provider=host)

    result = stage.run(np.zeros((4, 4, 3), dtype=np.uint8))

    assert result.data == {"ok": True}
    assert result.metadata["source"] == "depth.host_primary"
    assert result.metadata["fallback_reason"] == "primary_error"
    assert result.metadata["primary_error"] == "primary predict failed"
    assert fallback.predict_calls == 0


def test_stage_uses_edge_fallback_when_primary_and_host_fail():
    primary = FakeModule(name="primary", fail_predict=True)
    host = FakeModule(name="host", fail_predict=True)
    fallback = FakeModule(name="fallback", data={"fallback": True})
    stage = Stage("mass", primary, fallback=fallback, host_provider=host)

    result = stage.run(np.zeros((4, 4, 3), dtype=np.uint8))

    assert result.data == {"fallback": True}
    assert result.metadata["source"] == "mass.fallback"
    assert result.metadata["fallback_reason"] == "primary_error"
    assert "host predict failed" in result.metadata["host_error"]


def test_host_serialization_roundtrip_restores_detection_and_arrays():
    image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    detection = Detection(
        bbox=np.array([1, 2, 3, 4], dtype=np.float32),
        mask=np.ones((3, 3), dtype=np.uint8),
        class_id=2,
        class_name="metal_can",
        confidence=0.91,
    )
    result = ModuleResult(
        data=[detection],
        latency_ms=12.5,
        metadata={"depth": np.ones((2, 2), dtype=np.float32)},
    )

    request = stage_request_payload(image, {"features": {"mask": detection.mask}})
    restored_image = decode_value(request["image"])
    restored_kwargs = decode_value(request["kwargs"])
    restored_result = module_result_from_payload(module_result_to_payload(result))

    assert np.array_equal(restored_image, image)
    assert np.array_equal(restored_kwargs["features"]["mask"], detection.mask)
    restored_detection = restored_result.data[0]
    assert isinstance(restored_detection, Detection)
    assert restored_detection.class_name == "metal_can"
    assert np.array_equal(restored_detection.mask, detection.mask)
    assert np.array_equal(restored_result.metadata["depth"], np.ones((2, 2), dtype=np.float32))


def test_agent_config_parses_host_inference_settings(tmp_path):
    config = AgentConfig.from_mapping(
        {
            "device": {"id": "jetson-01"},
            "host_inference": {
                "enabled": True,
                "host": "gpu-host.local",
                "port": 8099,
                "request_timeout_seconds": 22,
                "stages": {
                    "detection": {"enabled": True},
                    "depth": {"enabled": False},
                    "mass": {"enabled": True, "endpoint_stage": "mass"},
                },
            },
        },
        base_dir=tmp_path,
    )

    assert config.host_inference.enabled is True
    assert config.host_inference.resolved_base_url == "http://gpu-host.local:8099"
    assert config.host_inference.request_timeout_seconds == 22
    assert config.host_inference.load_timeout_seconds == 120.0
    assert config.host_inference.stage_enabled("detection") is True
    assert config.host_inference.stage_enabled("depth") is False
    assert config.host_inference.endpoint_stage("mass") == "mass"


def test_prepare_host_model_environment_keeps_caches_under_models(tmp_path):
    models_dir = tmp_path / "models"
    tracked_env = [
        "EDGE_AI_MODELS_DIR",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TORCH_HOME",
        "YOLO_CONFIG_DIR",
        "ULTRALYTICS_CACHE_DIR",
    ]
    previous = {key: os.environ.get(key) for key in tracked_env}
    try:
        root = prepare_host_model_environment(models_dir)

        assert root == models_dir.resolve()
        assert (models_dir / "weights").is_dir()
        assert (models_dir / "huggingface" / "hub").is_dir()
        assert (models_dir / "torch" / "hub").is_dir()
        assert (models_dir / "ultralytics").is_dir()
        assert os.environ["EDGE_AI_MODELS_DIR"] == str(models_dir.resolve())
        assert os.environ["HF_HOME"].startswith(str(models_dir.resolve()))
        assert os.environ["TORCH_HOME"].startswith(str(models_dir.resolve()))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_host_client_uses_longer_timeout_when_stage_health_loads(monkeypatch):
    observed: list[float] = []

    def fake_request_json(
        self: HostInferenceClient,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float,
    ) -> dict[str, Any]:
        observed.append(timeout)
        return {"status": "ready"}

    monkeypatch.setattr(HostInferenceClient, "_request_json", fake_request_json)
    client = HostInferenceClient(
        "http://host",
        health_timeout_seconds=2.0,
        load_timeout_seconds=99.0,
    )

    client.stage_health("depth", load=False)
    client.stage_health("depth", load=True)

    assert observed == [2.0, 99.0]
