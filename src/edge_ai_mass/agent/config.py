"""Configuration models for the backend-facing edge device agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edge_ai_mass import __version__
from edge_ai_mass.agent.secrets import SecretStore, require_secret
from edge_ai_mass.utils.config import load_config

VALID_TELEMETRY_TRANSPORTS = {"http", "mqtt", "both"}
VALID_PREVIEW_BACKENDS = {"aiortc", "simulated"}


@dataclass(frozen=True, slots=True)
class DeviceSettings:
    """Stable identity and capability metadata for one edge device."""

    id: str
    name: str = ""
    type: str = "jetson_nano"
    firmware_version: str = "unknown"
    agent_version: str = __version__
    capabilities: dict[str, bool] = field(default_factory=dict)
    active_models: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendSettings:
    """Laravel edge API connection settings."""

    base_url: str = "http://localhost:8000"
    device_token_env: str = "DROVENAI_DEVICE_TOKEN"
    request_timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5


@dataclass(frozen=True, slots=True)
class MQTTSettings:
    """MQTT broker and topic settings for backend command exchange."""

    host: str = "127.0.0.1"
    port: int = 1883
    topic_prefix: str = "drovenai"
    username_env: str = "DROVENAI_MQTT_USERNAME"
    password_env: str = "DROVENAI_MQTT_PASSWORD"
    tls: bool = False
    keepalive_seconds: int = 30
    client_id: str = ""
    qos: int = 0


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Local runtime paths, intervals, and feature configuration."""

    telemetry_interval_seconds: float = 30.0
    telemetry_transport: str = "http"
    preview_backend: str = "aiortc"
    media_upload_retries: int = 5
    outbox_path: Path = Path("/var/lib/drovenai-agent/outbox")
    model_dir: Path = Path("/var/lib/drovenai-agent/models")
    media_dir: Path = Path("/var/lib/drovenai-agent/media")
    pipeline_config: str = "configs/pipeline/jetson_nano.yaml"
    preload_inference: bool = True
    camera_source: str = "camera:0"
    outbox_max_attempts: int = 8
    outbox_retention_seconds: int = 7 * 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Top-level edge agent configuration."""

    device: DeviceSettings
    backend: BackendSettings = field(default_factory=BackendSettings)
    mqtt: MQTTSettings = field(default_factory=MQTTSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    @classmethod
    def from_file(cls, path: str | Path) -> "AgentConfig":
        raw = load_config(path) or {}
        return cls.from_mapping(raw, base_dir=Path(path).resolve().parent)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], base_dir: Path | None = None) -> "AgentConfig":
        base_dir = base_dir or Path.cwd()
        device_raw = raw.get("device") or {}
        backend_raw = raw.get("backend") or {}
        mqtt_raw = raw.get("mqtt") or {}
        runtime_raw = raw.get("runtime") or {}

        device_id = str(device_raw.get("id") or raw.get("device_id") or "").strip()
        if not device_id:
            raise ValueError("Agent config requires device.id")

        device = DeviceSettings(
            id=device_id,
            name=str(device_raw.get("name") or ""),
            type=str(device_raw.get("type") or "jetson_nano"),
            firmware_version=str(device_raw.get("firmware_version") or "unknown"),
            agent_version=str(device_raw.get("agent_version") or __version__),
            capabilities=_bool_map(
                device_raw.get(
                    "capabilities",
                    {"inference": True, "preview": True, "model_update": True},
                )
            ),
            active_models={
                str(key): str(value)
                for key, value in (device_raw.get("active_models") or {}).items()
            },
        )

        backend = BackendSettings(
            base_url=str(backend_raw.get("base_url") or "http://localhost:8000").rstrip("/"),
            device_token_env=str(
                backend_raw.get("device_token_env") or "DROVENAI_DEVICE_TOKEN"
            ),
            request_timeout_seconds=float(backend_raw.get("request_timeout_seconds", 10)),
            max_retries=int(backend_raw.get("max_retries", 3)),
            retry_backoff_seconds=float(backend_raw.get("retry_backoff_seconds", 0.5)),
        )

        mqtt = MQTTSettings(
            host=str(mqtt_raw.get("host") or "127.0.0.1"),
            port=int(mqtt_raw.get("port", 1883)),
            topic_prefix=str(mqtt_raw.get("topic_prefix") or "drovenai").strip("/"),
            username_env=str(mqtt_raw.get("username_env") or "DROVENAI_MQTT_USERNAME"),
            password_env=str(mqtt_raw.get("password_env") or "DROVENAI_MQTT_PASSWORD"),
            tls=_as_bool(mqtt_raw.get("tls", False)),
            keepalive_seconds=int(mqtt_raw.get("keepalive_seconds", 30)),
            client_id=str(mqtt_raw.get("client_id") or f"{device_id}-agent"),
            qos=int(mqtt_raw.get("qos", 0)),
        )

        telemetry_transport = str(
            runtime_raw.get("telemetry_transport") or "http"
        ).lower()
        if telemetry_transport not in VALID_TELEMETRY_TRANSPORTS:
            raise ValueError(
                "runtime.telemetry_transport must be one of "
                f"{sorted(VALID_TELEMETRY_TRANSPORTS)}, got {telemetry_transport!r}"
            )

        preview_backend = str(runtime_raw.get("preview_backend") or "aiortc").lower()
        if preview_backend not in VALID_PREVIEW_BACKENDS:
            raise ValueError(
                "runtime.preview_backend must be one of "
                f"{sorted(VALID_PREVIEW_BACKENDS)}, got {preview_backend!r}"
            )

        runtime = RuntimeSettings(
            telemetry_interval_seconds=float(runtime_raw.get("telemetry_interval_seconds", 30)),
            telemetry_transport=telemetry_transport,
            preview_backend=preview_backend,
            media_upload_retries=int(runtime_raw.get("media_upload_retries", 5)),
            outbox_path=_resolve_path(
                runtime_raw.get("outbox_path", "/var/lib/drovenai-agent/outbox"),
                base_dir,
            ),
            model_dir=_resolve_path(
                runtime_raw.get("model_dir", "/var/lib/drovenai-agent/models"),
                base_dir,
            ),
            media_dir=_resolve_path(
                runtime_raw.get("media_dir", "/var/lib/drovenai-agent/media"),
                base_dir,
            ),
            pipeline_config=str(
                runtime_raw.get("pipeline_config") or "configs/pipeline/jetson_nano.yaml"
            ),
            preload_inference=_as_bool(runtime_raw.get("preload_inference", True)),
            camera_source=str(runtime_raw.get("camera_source") or "camera:0"),
            outbox_max_attempts=int(runtime_raw.get("outbox_max_attempts", 8)),
            outbox_retention_seconds=int(
                runtime_raw.get("outbox_retention_seconds", 7 * 24 * 60 * 60)
            ),
        )

        return cls(device=device, backend=backend, mqtt=mqtt, runtime=runtime)

    def require_device_token(self, secret_store: SecretStore) -> str:
        """Return the HTTP provisioning token or fail startup clearly."""

        return require_secret(
            secret_store,
            self.backend.device_token_env,
            label="HTTP device provisioning token",
        )

    def mqtt_credentials(self, secret_store: SecretStore) -> tuple[str, str]:
        """Return optional MQTT username/password from configured secret keys."""

        username = secret_store.get(self.mqtt.username_env) if self.mqtt.username_env else ""
        password = secret_store.get(self.mqtt.password_env) if self.mqtt.password_env else ""
        return username, password


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _bool_map(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _as_bool(val) for key, val in value.items()}
