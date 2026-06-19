"""Tests for agent inference normalization and deployment helpers."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from edge_ai_mass.agent.inference import normalize_pipeline_result
from edge_ai_mass.agent.model_manager import ModelDeploymentError, ModelManager
from edge_ai_mass.agent.outbox import FileOutbox
from edge_ai_mass.agent.preview import PreviewError, PreviewManager
from edge_ai_mass.agent.runner import EdgeDeviceAgent
from edge_ai_mass.agent.config import AgentConfig
from edge_ai_mass.agent.envelopes import new_envelope
from edge_ai_mass.pipeline.pipeline import Detection, ObjectEstimate, PipelineResult


def test_normalize_pipeline_result_matches_backend_inference_payload():
    detection = Detection(
        bbox=np.array([10, 20, 110, 160], dtype=np.float32),
        mask=None,
        class_id=1,
        class_name="plastic_bottle",
        confidence=0.94,
    )
    obj = ObjectEstimate(
        detection=detection,
        geometry={"area_px": 14000},
        material="plastic",
        mass_kg=0.1205,
        mass_method="depth_estimate",
    )
    result = PipelineResult(objects=[obj], frame_time_ms=142.0)

    payload = normalize_pipeline_result(
        result,
        request_id="request-1",
        correlation_id="correlation-1",
        model_versions={"detector": "yolo-waste-v1", "mass": "mass-v1"},
    )

    assert payload["request_id"] == "request-1"
    assert payload["correlation_id"] == "correlation-1"
    assert payload["latency_ms"] == 142
    assert payload["object_count"] == 1
    assert payload["total_estimated_mass_grams"] == pytest.approx(120.5)
    assert payload["dominant_material"] == "plastic"
    assert payload["objects"][0]["class_label"] == "plastic_bottle"
    assert payload["objects"][0]["bbox"] == {"x": 10.0, "y": 20.0, "w": 100.0, "h": 140.0}
    assert payload["objects"][0]["estimated_mass_grams"] == pytest.approx(120.5)


def test_model_manager_activates_local_component_and_updates_versions(tmp_path):
    source = tmp_path / "detector.engine"
    source.write_bytes(b"engine")
    checksum = hashlib.sha256(b"engine").hexdigest()
    manager = ModelManager(tmp_path / "models")

    result = manager.deploy(
        {
            "deployment_id": 1,
            "target_device_id": "jetson-01",
            "components": [
                {
                    "task": "detector",
                    "version": "v2.1.0",
                    "source_type": "local",
                    "source_uri": str(source),
                    "artifact": {"checksum": checksum},
                }
            ],
        },
        device_id="jetson-01",
    )

    active_path = tmp_path / "models" / "detector" / "v2.1.0" / "detector.engine"
    assert active_path.read_bytes() == b"engine"
    assert result["active_models"]["detector"] == "v2.1.0"
    assert result["activated_components"][0]["active_path"] == str(active_path)


def test_model_manager_refuses_wrong_target_device(tmp_path):
    manager = ModelManager(tmp_path / "models")

    with pytest.raises(ModelDeploymentError, match="does not match"):
        manager.deploy(
            {
                "target_device_id": "other-device",
                "components": [{"task": "detector", "version": "v1", "source_uri": "x"}],
            },
            device_id="jetson-01",
        )


def test_preview_start_returns_ready_payload_with_webrtc_options():
    manager = PreviewManager(camera_available=lambda _source: True, time_fn=lambda: 100.0)

    payload = manager.start(
        {
            "mode": "low_fps",
            "camera_source": "camera:0",
            "ttl_seconds": 60,
            "webrtc": {
                "video": {
                    "codec_preferences": ["VP8", "H264"],
                    "max_width": 640,
                    "max_height": 360,
                    "max_fps": 5,
                }
            },
        },
        request_id="request-1",
        correlation_id="correlation-1",
    )

    assert payload["request_id"] == "request-1"
    assert payload["correlation_id"] == "correlation-1"
    assert payload["expires_at"] == "1970-01-01T00:02:40Z"
    assert payload["webrtc"]["role"] == "answerer"
    assert payload["webrtc"]["supported_codecs"] == ["VP8", "H264"]
    assert payload["webrtc"]["max_width"] == 640
    assert manager.sessions["request-1"].state == "ready"


def test_preview_signal_requires_active_matching_session():
    manager = PreviewManager(camera_available=lambda _source: True, time_fn=lambda: 100.0)

    with pytest.raises(PreviewError) as missing:
        manager.handle_signal(
            {"signal_type": "offer", "peer_id": "peer-1", "sdp": "v=0\r\n"},
            request_id="unknown",
            correlation_id="correlation-1",
        )
    assert missing.value.error_type == "session_expired"

    manager.start(
        {"camera_source": "camera:0"},
        request_id="request-1",
        correlation_id="correlation-1",
    )

    with pytest.raises(PreviewError) as wrong_correlation:
        manager.handle_signal(
            {"signal_type": "offer", "peer_id": "peer-1", "sdp": "v=0\r\n"},
            request_id="request-1",
            correlation_id="other-correlation",
        )
    assert wrong_correlation.value.error_type == "permission_denied"


def test_preview_offer_returns_answer_ice_and_started_events():
    manager = PreviewManager(camera_available=lambda _source: True, time_fn=lambda: 100.0)
    manager.start(
        {"camera_source": "camera:0", "ttl_seconds": 60},
        request_id="request-1",
        correlation_id="correlation-1",
    )

    events = manager.handle_signal(
        {
            "request_id": "request-1",
            "correlation_id": "correlation-1",
            "signal_type": "offer",
            "peer_id": "operator-browser-1",
            "sdp": "v=0\r\n",
        },
        request_id="request-1",
        correlation_id="correlation-1",
    )

    assert [event.event_name for event in events] == [
        "preview.webrtc_answer",
        "preview.webrtc_ice_candidate",
        "preview.started",
    ]
    assert events[0].payload["peer_id"] == "operator-browser-1"
    assert events[0].payload["sdp"].startswith("v=0")
    assert events[1].payload["candidate"]["candidate"].startswith("candidate:")
    assert events[2].payload["selected_codec"] == "H264"
    assert manager.sessions["request-1"].state == "active"


def test_preview_ttl_expiry_emits_stopped_event():
    now = {"value": 100.0}
    manager = PreviewManager(
        camera_available=lambda _source: True,
        time_fn=lambda: now["value"],
    )
    manager.start(
        {"camera_source": "camera:0", "ttl_seconds": 5},
        request_id="request-1",
        correlation_id="correlation-1",
    )

    now["value"] = 106.0
    events = manager.expire_sessions()

    assert len(events) == 1
    assert events[0].event_name == "preview.stopped"
    assert events[0].payload["reason"] == "ttl_expired"
    assert "request-1" not in manager.sessions


class RecordingMqtt:
    is_started = True
    is_connected = True

    def __init__(self):
        self.events = []
        self.telemetry_envelopes = []

    def publish_envelope(self, envelope):
        self.events.append(
            {
                "event_name": envelope.event_name,
                "payload": envelope.payload,
                "correlation_id": envelope.correlation_id,
                "request_id": envelope.request_id,
            }
        )

    def publish_telemetry_envelope(self, envelope):
        self.telemetry_envelopes.append(envelope)

    def publish_event(self, event_name, payload, *, correlation_id=None, request_id=None):
        self.events.append(
            {
                "event_name": event_name,
                "payload": payload,
                "correlation_id": correlation_id,
                "request_id": request_id,
            }
        )
        return new_envelope(
            device_id="jetson-01",
            event_name=event_name,
            payload=payload,
            correlation_id=correlation_id,
            request_id=request_id,
        )


def test_agent_preview_command_flow_publishes_ready_then_signaling_events():
    config = AgentConfig.from_mapping({"device": {"id": "jetson-01"}})
    mqtt = RecordingMqtt()
    agent = EdgeDeviceAgent(
        config,
        mqtt_client=mqtt,
        preview_manager=PreviewManager(camera_available=lambda _source: True),
    )

    start = new_envelope(
        device_id="jetson-01",
        event_name="preview.start_requested",
        payload={
            "request_id": "request-1",
            "correlation_id": "correlation-1",
            "camera_source": "camera:0",
            "ttl_seconds": 60,
        },
        correlation_id="correlation-1",
        request_id="request-1",
    )
    signal = new_envelope(
        device_id="jetson-01",
        event_name="preview.webrtc_signal",
        payload={
            "request_id": "request-1",
            "correlation_id": "correlation-1",
            "signal_type": "offer",
            "peer_id": "operator-browser-1",
            "sdp": "v=0\r\n",
        },
        correlation_id="correlation-1",
        request_id="request-1",
    )

    agent.handle_command(start)
    agent.handle_command(signal)

    assert [event["event_name"] for event in mqtt.events] == [
        "preview.ready",
        "preview.webrtc_answer",
        "preview.webrtc_ice_candidate",
        "preview.started",
    ]
    assert mqtt.events[0]["payload"]["webrtc"]["role"] == "answerer"
    assert mqtt.events[1]["payload"]["peer_id"] == "operator-browser-1"


class FixedTelemetry:
    def sample(self, *, status="online"):
        return {
            "status": status,
            "health_status": "healthy",
            "reported_at": "2026-06-19T12:00:00Z",
        }


def test_agent_can_send_telemetry_through_mqtt():
    config = AgentConfig.from_mapping(
        {
            "device": {"id": "jetson-01"},
            "runtime": {"telemetry_transport": "mqtt"},
        }
    )
    mqtt = RecordingMqtt()
    agent = EdgeDeviceAgent(config, mqtt_client=mqtt, telemetry=FixedTelemetry())

    response = agent.send_telemetry(
        status="online",
        correlation_id="correlation-1",
        request_id="request-1",
    )

    assert response is None
    assert len(mqtt.telemetry_envelopes) == 1
    envelope = mqtt.telemetry_envelopes[0]
    assert envelope.event_name == "telemetry.reported"
    assert envelope.correlation_id == "correlation-1"
    assert envelope.request_id == "request-1"
    assert envelope.payload["status"] == "online"


class DisconnectedMqtt(RecordingMqtt):
    is_connected = False


def test_disconnected_mqtt_telemetry_is_persisted_to_outbox(tmp_path):
    config = AgentConfig.from_mapping(
        {
            "device": {"id": "jetson-01"},
            "runtime": {
                "telemetry_transport": "mqtt",
                "outbox_path": str(tmp_path),
            },
        }
    )
    outbox = FileOutbox(tmp_path)
    agent = EdgeDeviceAgent(
        config,
        mqtt_client=DisconnectedMqtt(),
        telemetry=FixedTelemetry(),
        outbox=outbox,
    )

    agent.send_telemetry(status="online")

    records = outbox.due_records()
    assert len(records) == 1
    assert records[0].kind == "mqtt_telemetry"
    assert records[0].payload["envelope"]["event_name"] == "telemetry.reported"


def test_mqtt_telemetry_outbox_record_replays_with_same_message_id(tmp_path):
    config = AgentConfig.from_mapping(
        {
            "device": {"id": "jetson-01"},
            "runtime": {
                "telemetry_transport": "mqtt",
                "outbox_path": str(tmp_path),
            },
        }
    )
    outbox = FileOutbox(tmp_path)
    original = new_envelope(
        device_id="jetson-01",
        event_name="telemetry.reported",
        payload={"status": "online"},
        correlation_id="correlation-1",
    )
    outbox.enqueue("mqtt_telemetry", {"envelope": original.to_dict()})
    mqtt = RecordingMqtt()
    agent = EdgeDeviceAgent(config, mqtt_client=mqtt, outbox=outbox)

    agent.flush_outbox()

    assert len(mqtt.telemetry_envelopes) == 1
    assert mqtt.telemetry_envelopes[0].message_id == original.message_id
    assert outbox.due_records() == []
