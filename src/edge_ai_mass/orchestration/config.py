"""Configuration models for the Jetson orchestrator process."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from edge_ai_mass.utils.config import load_config


VALID_MODES = {"poll", "mqtt", "hybrid", "manual"}


@dataclass(slots=True)
class GitHubSettings:
    """GitHub Releases source used by the updater."""

    owner: str
    repo: str
    token: str = ""
    release: str = "latest"
    api_base_url: str = "https://api.github.com"
    manifest_asset: str = "edge-ai-update-manifest.json"
    channel: str = "stable"
    include_prereleases: bool = False
    timeout_seconds: int = 30


@dataclass(slots=True)
class MQTTSettings:
    """Mosquitto-compatible MQTT trigger settings."""

    enabled: bool = False
    host: str = "localhost"
    port: int = 1883
    username: str = ""
    password: str = ""
    tls: bool = False
    client_id: str = "edge-ai-mass-orchestrator"
    topics: list[str] = field(default_factory=list)
    status_topic: str = "edge-ai-mass/{device_id}/status"
    keepalive_seconds: int = 60


@dataclass(slots=True)
class RuntimeSettings:
    """Local process and post-update commands."""

    manage_app_process: bool = False
    app_command: list[str] = field(default_factory=list)
    restart_on_exit: bool = True
    restart_app_on_update: bool = True
    restart_command: list[str] = field(default_factory=list)
    health_check_command: list[str] = field(default_factory=list)
    command_timeout_seconds: int = 300


@dataclass(slots=True)
class InstallRule:
    """Mapping from a release asset to a local install action."""

    name: str
    asset: str = ""
    source: str = "github"
    target: str = "model"
    destination: str = ""
    sha256: str = ""
    unpack: str = "none"
    install_command: list[str] = field(default_factory=list)
    restart_app: bool = True
    executable: bool = False
    required: bool = True
    kaggle_model: str = ""
    kaggle_model_version: str = ""
    kaggle_model_file: str = ""

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "InstallRule":
        source = str(data.get("source") or data.get("source_type") or "github").lower()
        source = source.replace("-", "_")
        kaggle_model_file = str(data.get("kaggle_model_file") or data.get("model_path") or "")
        asset = str(
            data.get("asset")
            or data.get("asset_name")
            or data.get("file")
            or _asset_name_from_source_path(kaggle_model_file)
            or ""
        )
        name = str(data.get("name") or asset)
        if not asset:
            raise ValueError(f"Install rule {name!r} is missing an asset name")
        return cls(
            name=name,
            asset=asset,
            source=source,
            target=str(data.get("target") or data.get("kind") or "model"),
            destination=str(data.get("destination") or data.get("dest") or ""),
            sha256=str(data.get("sha256") or ""),
            unpack=str(data.get("unpack") or "none").lower(),
            install_command=_as_command(data.get("install_command")),
            restart_app=_as_bool(data.get("restart_app", True)),
            executable=_as_bool(data.get("executable", False)),
            required=_as_bool(data.get("required", True)),
            kaggle_model=str(data.get("kaggle_model") or data.get("model") or ""),
            kaggle_model_version=str(
                data.get("kaggle_model_version") or data.get("model_version") or ""
            ),
            kaggle_model_file=kaggle_model_file,
        )


@dataclass(slots=True)
class OrchestratorConfig:
    """Top-level settings for the long-running orchestrator process."""

    project_root: Path
    device_id: str
    mode: str
    poll_interval_seconds: int
    run_on_start: bool
    state_path: Path
    cache_dir: Path
    backup_dir: Path
    github: GitHubSettings
    mqtt: MQTTSettings
    runtime: RuntimeSettings
    install_plan: list[InstallRule]

    @classmethod
    def from_file(cls, path: str | Path) -> "OrchestratorConfig":
        raw = load_config(path) or {}
        return cls.from_mapping(raw, base_dir=Path(path).resolve().parent)

    @classmethod
    def from_mapping(
        cls, raw: dict[str, Any], base_dir: Path | None = None
    ) -> "OrchestratorConfig":
        base_dir = base_dir or Path.cwd()
        project_root = _resolve_path(raw.get("project_root", "."), base_dir).resolve()
        updates = raw.get("updates", {})
        mode = str(updates.get("mode", raw.get("mode", "poll"))).lower()
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid orchestrator mode {mode!r}; expected one of {VALID_MODES}")

        device_id = str(
            raw.get("device_id")
            or os.environ.get("EDGE_AI_DEVICE_ID")
            or os.environ.get("HOSTNAME")
            or os.environ.get("COMPUTERNAME")
            or "jetson-nano"
        )

        github_raw = updates.get("github", raw.get("github", {}))
        github = GitHubSettings(
            owner=str(github_raw.get("owner") or "drovenai"),
            repo=str(github_raw.get("repo") or "edge-ai-mass-estimation"),
            token=str(github_raw.get("token") or ""),
            release=str(github_raw.get("release") or "latest"),
            api_base_url=str(github_raw.get("api_base_url") or "https://api.github.com"),
            manifest_asset=str(
                github_raw.get("manifest_asset") or "edge-ai-update-manifest.json"
            ),
            channel=str(github_raw.get("channel") or "stable"),
            include_prereleases=_as_bool(github_raw.get("include_prereleases", False)),
            timeout_seconds=int(github_raw.get("timeout_seconds", 30)),
        )

        mqtt_raw = raw.get("mqtt", {})
        topic_defaults = [
            f"edge-ai-mass/{device_id}/updates",
            "edge-ai-mass/all/updates",
        ]
        mqtt = MQTTSettings(
            enabled=_as_bool(mqtt_raw.get("enabled", mode in {"mqtt", "hybrid"})),
            host=str(mqtt_raw.get("host") or "localhost"),
            port=int(mqtt_raw.get("port", 1883)),
            username=str(mqtt_raw.get("username") or ""),
            password=str(mqtt_raw.get("password") or ""),
            tls=_as_bool(mqtt_raw.get("tls", False)),
            client_id=str(mqtt_raw.get("client_id") or f"edge-ai-mass-{device_id}"),
            topics=[str(t) for t in mqtt_raw.get("topics", topic_defaults)],
            status_topic=str(
                mqtt_raw.get("status_topic") or "edge-ai-mass/{device_id}/status"
            ),
            keepalive_seconds=int(mqtt_raw.get("keepalive_seconds", 60)),
        )

        runtime_raw = raw.get("runtime", {})
        runtime = RuntimeSettings(
            manage_app_process=_as_bool(runtime_raw.get("manage_app_process", False)),
            app_command=_as_command(runtime_raw.get("app_command")),
            restart_on_exit=_as_bool(runtime_raw.get("restart_on_exit", True)),
            restart_app_on_update=_as_bool(runtime_raw.get("restart_app_on_update", True)),
            restart_command=_as_command(runtime_raw.get("restart_command")),
            health_check_command=_as_command(runtime_raw.get("health_check_command")),
            command_timeout_seconds=int(runtime_raw.get("command_timeout_seconds", 300)),
        )

        plan_raw = updates.get("install_plan", raw.get("install_plan", {}))
        assets_raw = plan_raw.get("assets", plan_raw if isinstance(plan_raw, list) else [])
        install_plan = [InstallRule.from_mapping(item) for item in assets_raw]

        return cls(
            project_root=project_root,
            device_id=device_id,
            mode=mode,
            poll_interval_seconds=int(updates.get("poll_interval_seconds", 6 * 60 * 60)),
            run_on_start=_as_bool(updates.get("run_on_start", True)),
            state_path=_resolve_path(
                raw.get("state_path", ".edge_ai_mass/orchestrator/state.json"), project_root
            ),
            cache_dir=_resolve_path(
                raw.get("cache_dir", ".edge_ai_mass/orchestrator/cache"), project_root
            ),
            backup_dir=_resolve_path(
                raw.get("backup_dir", ".edge_ai_mass/orchestrator/backups"), project_root
            ),
            github=github,
            mqtt=mqtt,
            runtime=runtime,
            install_plan=install_plan,
        )


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return base_dir / path


def _as_command(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, (list, tuple)):
        return [str(part) for part in value]
    raise TypeError(f"Command must be a string or list, got {type(value).__name__}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _asset_name_from_source_path(source_path: str) -> str:
    if not source_path:
        return ""
    normalized = source_path.replace("\\", "/").rstrip("/")
    return PurePosixPath(normalized).name
