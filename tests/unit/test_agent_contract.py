"""Tests for backend edge-agent config and MQTT envelopes."""

from __future__ import annotations

import pytest

from edge_ai_mass.agent.config import AgentConfig
from edge_ai_mass.agent.envelopes import (
    CommandValidationError,
    MessageDeduper,
    expected_event_name_for_topic,
    new_envelope,
    parse_envelope,
    required_payload_fields,
    validate_command,
)
from edge_ai_mass.agent.secrets import EnvironmentSecretStore, MissingSecretError


def test_agent_config_loads_identity_and_token_from_env(tmp_path):
    config = AgentConfig.from_mapping(
        {
            "device": {
                "id": "jetson-nano-line-a-01",
                "capabilities": {"inference": "true", "preview": "false"},
            },
            "backend": {"device_token_env": "TEST_DEVICE_TOKEN"},
            "runtime": {
                "outbox_path": "outbox",
                "telemetry_transport": "mqtt",
            },
        },
        base_dir=tmp_path,
    )
    store = EnvironmentSecretStore({"TEST_DEVICE_TOKEN": "raw-token"})

    assert config.device.id == "jetson-nano-line-a-01"
    assert config.device.capabilities == {"inference": True, "preview": False}
    assert config.runtime.outbox_path == tmp_path / "outbox"
    assert config.runtime.telemetry_transport == "mqtt"
    assert config.require_device_token(store) == "raw-token"


def test_agent_config_requires_device_id_and_http_token():
    with pytest.raises(ValueError, match="device.id"):
        AgentConfig.from_mapping({"device": {}})

    config = AgentConfig.from_mapping({"device": {"id": "jetson-01"}})
    with pytest.raises(MissingSecretError):
        config.require_device_token(EnvironmentSecretStore({}))

    with pytest.raises(ValueError, match="telemetry_transport"):
        AgentConfig.from_mapping(
            {
                "device": {"id": "jetson-01"},
                "runtime": {"telemetry_transport": "carrier-pigeon"},
            }
        )


def test_envelope_round_trip_and_command_validation():
    envelope = new_envelope(
        device_id="jetson-01",
        event_name="inference.requested",
        payload={
            "request_id": "request-1",
            "source_type": "camera",
            "source_reference": "camera:0",
        },
        correlation_id="correlation-1",
    )

    parsed = parse_envelope(envelope.to_json())
    validated = validate_command(
        parsed,
        expected_device_id="jetson-01",
        expected_event_name="inference.requested",
        required_payload_fields=("source_type", "source_reference"),
        deduper=MessageDeduper(),
    )

    assert validated.correlation_id == "correlation-1"
    assert validated.request_id == "request-1"
    assert validated.payload["source_reference"] == "camera:0"


def test_validate_command_rejects_wrong_device_and_duplicates():
    envelope = new_envelope(
        device_id="other-device",
        event_name="preview.start_requested",
        payload={"camera_source": "camera:0"},
    )

    with pytest.raises(CommandValidationError) as exc:
        validate_command(envelope, expected_device_id="jetson-01")
    assert exc.value.field == "device_id"

    deduper = MessageDeduper()
    matching = new_envelope(
        device_id="jetson-01",
        event_name="preview.start_requested",
        payload={"camera_source": "camera:0"},
    )
    validate_command(matching, expected_device_id="jetson-01", deduper=deduper)
    with pytest.raises(CommandValidationError) as duplicate:
        validate_command(matching, expected_device_id="jetson-01", deduper=deduper)
    assert duplicate.value.field == "message_id"


def test_preview_signal_topic_and_payload_requirements_are_registered():
    topic = "drovenai/devices/jetson-01/commands/preview/signal"

    assert expected_event_name_for_topic(topic) == "preview.webrtc_signal"
    assert required_payload_fields("preview.webrtc_signal") == (
        "request_id",
        "signal_type",
        "peer_id",
    )
