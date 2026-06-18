"""Device identity and host metadata helpers."""

from __future__ import annotations

import platform
import socket
from dataclasses import asdict, dataclass
from typing import Any

from edge_ai_mass.agent.config import AgentConfig


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    """Immutable edge device identity exposed to telemetry and logs."""

    device_id: str
    name: str
    type: str
    firmware_version: str
    agent_version: str
    capabilities: dict[str, bool]
    hostname: str
    platform: str
    python_version: str

    @classmethod
    def from_config(cls, config: AgentConfig) -> "DeviceIdentity":
        device = config.device
        return cls(
            device_id=device.id,
            name=device.name,
            type=device.type,
            firmware_version=device.firmware_version,
            agent_version=device.agent_version,
            capabilities=dict(device.capabilities),
            hostname=socket.gethostname(),
            platform=platform.platform(),
            python_version=platform.python_version(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
