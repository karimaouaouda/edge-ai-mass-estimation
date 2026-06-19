"""Tests for edge-agent HTTP, outbox, and MQTT transport behavior."""

from __future__ import annotations

import json

import pytest

from edge_ai_mass.agent.config import AgentConfig
from edge_ai_mass.agent.envelopes import new_envelope
from edge_ai_mass.agent.http_client import EdgeHttpClient, HttpRequest, HttpResponse
from edge_ai_mass.agent.mqtt_client import EdgeMqttClient, MqttPublishError
from edge_ai_mass.agent.outbox import FileOutbox


class RecordingTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[HttpRequest] = []

    def __call__(self, request: HttpRequest, timeout: float) -> HttpResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _json_response(status: int, payload: dict) -> HttpResponse:
    return HttpResponse(
        status_code=status,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )


def test_http_submit_telemetry_uses_backend_contract_headers():
    transport = RecordingTransport([_json_response(201, {"telemetry": {"id": 1}})])
    client = EdgeHttpClient(
        base_url="http://localhost:8000",
        device_id="jetson-01",
        device_token="token-123",
        transport=transport,
        sleep=lambda _seconds: None,
    )

    response = client.submit_telemetry({"status": "online"})

    request = transport.requests[0]
    assert response["telemetry"]["id"] == 1
    assert request.url == "http://localhost:8000/api/edge/devices/jetson-01/telemetry"
    assert request.headers["Accept"] == "application/json"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["X-Device-Token"] == "token-123"
    assert json.loads(request.body.decode("utf-8"))["status"] == "online"


def test_http_retries_5xx_but_not_forbidden():
    transport = RecordingTransport(
        [
            _json_response(500, {"message": "down"}),
            _json_response(201, {"telemetry": {"id": 2}}),
        ]
    )
    client = EdgeHttpClient(
        base_url="http://localhost:8000",
        device_id="jetson-01",
        device_token="token",
        transport=transport,
        sleep=lambda _seconds: None,
    )

    assert client.submit_telemetry({"status": "online"})["telemetry"]["id"] == 2
    assert len(transport.requests) == 2

    forbidden = RecordingTransport([_json_response(403, {"message": "Forbidden"})])
    client = EdgeHttpClient(
        base_url="http://localhost:8000",
        device_id="jetson-01",
        device_token="bad-token",
        transport=forbidden,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(Exception) as exc_info:
        client.submit_telemetry({"status": "online"})
    assert "HTTP 403" in str(exc_info.value)
    assert len(forbidden.requests) == 1


def test_http_upload_media_builds_multipart_with_sha_and_metadata(tmp_path):
    image = tmp_path / "annotated.jpg"
    image.write_bytes(b"fake-image")
    transport = RecordingTransport([_json_response(201, {"media_file": {"id": 3}})])
    client = EdgeHttpClient(
        base_url="http://localhost:8000",
        device_id="jetson-01",
        device_token="token",
        transport=transport,
        sleep=lambda _seconds: None,
    )

    response = client.upload_media(
        image,
        media_type="annotated",
        request_id="request-1",
        correlation_id="correlation-1",
        metadata={"object_count": 2},
    )

    request = transport.requests[0]
    body = request.body.decode("utf-8", errors="replace")
    assert response["media_file"]["id"] == 3
    assert request.url.endswith("/api/edge/devices/jetson-01/media")
    assert "multipart/form-data; boundary=" in request.headers["Content-Type"]
    assert 'name="media_type"' in body
    assert "annotated" in body
    assert 'name="metadata[object_count]"' in body
    assert "fake-image" in body


def test_file_outbox_retries_and_dead_letters(tmp_path):
    outbox = FileOutbox(tmp_path, max_attempts=2)
    record = outbox.enqueue("telemetry", {"payload": {"status": "online"}}, now=100.0)

    due = outbox.due_records(now=100.0)
    assert due[0].id == record.id

    failed = outbox.mark_failed(due[0], "network", now=100.0)
    assert failed.attempts == 1
    assert outbox.due_records(now=100.5) == []
    due_again = outbox.due_records(now=102.0)[0]
    outbox.mark_failed(due_again, "still down", now=102.0)

    assert not list((tmp_path / "pending").glob("*.json"))
    assert len(list((tmp_path / "dead").glob("*.json"))) == 1


class FakePahoClient:
    def __init__(self, *, publish_rc=0):
        self.published = []
        self.publish_rc = publish_rc

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append(
            {"topic": topic, "payload": payload, "qos": qos, "retain": retain}
        )
        return FakeMessageInfo(rc=self.publish_rc, mid=len(self.published))


class FakeMessageInfo:
    def __init__(self, *, rc, mid):
        self.rc = rc
        self.mid = mid


def test_mqtt_dispatches_valid_commands_and_publishes_error_for_duplicate():
    config = AgentConfig.from_mapping({"device": {"id": "jetson-01"}})
    handled = []
    mqtt = EdgeMqttClient(config=config, on_command=handled.append)
    mqtt._client = FakePahoClient()
    mqtt._connected.set()

    envelope = new_envelope(
        device_id="jetson-01",
        event_name="inference.requested",
        payload={"source_type": "camera", "source_reference": "camera:0"},
        correlation_id="correlation-1",
        request_id="request-1",
    )
    topic = "drovenai/devices/jetson-01/commands/inference"

    mqtt.handle_message(topic, envelope.to_json())
    mqtt.handle_message(topic, envelope.to_json())

    assert handled == [envelope]
    assert mqtt._client.published[0]["topic"] == "drovenai/devices/jetson-01/events/device.error"
    payload = json.loads(mqtt._client.published[0]["payload"])
    assert payload["event_name"] == "device.error"
    assert payload["payload"]["error_type"] == "invalid_command"


def test_mqtt_publish_preserves_dotted_backend_event_topic():
    config = AgentConfig.from_mapping({"device": {"id": "jetson-01"}})
    mqtt = EdgeMqttClient(config=config, on_command=lambda _envelope: None)
    mqtt._client = FakePahoClient()
    mqtt._connected.set()

    envelope = mqtt.publish_event(
        "preview.ready",
        {"request_id": "request-1"},
        correlation_id="correlation-1",
        request_id="request-1",
    )

    published = mqtt._client.published[0]
    assert published["topic"] == "drovenai/devices/jetson-01/events/preview.ready"
    assert json.loads(published["payload"])["message_id"] == envelope.message_id


def test_mqtt_publish_raises_when_paho_rejects_message():
    config = AgentConfig.from_mapping({"device": {"id": "jetson-01"}})
    mqtt = EdgeMqttClient(config=config, on_command=lambda _envelope: None)
    mqtt._client = FakePahoClient(publish_rc=4)
    mqtt._connected.set()

    with pytest.raises(MqttPublishError, match="rc=4"):
        mqtt.publish_event("device.error", {"error_type": "test"})
