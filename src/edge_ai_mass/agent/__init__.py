"""Backend-facing Jetson device agent integration.

This package contains the device/runtime side of the Laravel supervision
contract: HTTP telemetry/media upload, MQTT command/event envelopes, local
outbox persistence, and adapters around the existing inference pipeline.
"""

from edge_ai_mass.agent.config import AgentConfig
from edge_ai_mass.agent.envelopes import Envelope, new_envelope, parse_envelope

__all__ = [
    "AgentConfig",
    "Envelope",
    "new_envelope",
    "parse_envelope",
]
