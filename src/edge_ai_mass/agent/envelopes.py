"""MQTT message envelope helpers for backend/device workflows."""

from __future__ import annotations

import json
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


COMMAND_EVENT_BY_SUFFIX = {
    "inference": "inference.requested",
    "preview/start": "preview.start_requested",
    "preview/signal": "preview.webrtc_signal",
    "preview/stop": "preview.stop_requested",
    "model-deployment": "model_deployment.requested",
}


class EnvelopeError(ValueError):
    """Base class for MQTT envelope parse and validation errors."""


class CommandValidationError(EnvelopeError):
    """Raised when an inbound command envelope violates the backend contract."""

    def __init__(
        self,
        summary: str,
        *,
        source_message_id: str | None = None,
        field: str | None = None,
    ) -> None:
        super().__init__(summary)
        self.summary = summary
        self.source_message_id = source_message_id
        self.field = field

    def to_error_payload(self) -> dict[str, Any]:
        details = {}
        if self.field:
            details["field"] = self.field
        return {
            "error_type": "invalid_command",
            "summary": self.summary,
            "source_message_id": self.source_message_id,
            "retryable": False,
            "details": details,
        }


@dataclass(frozen=True, slots=True)
class Envelope:
    """Backend MQTT envelope shared by commands and events."""

    message_id: str
    correlation_id: str
    request_id: str | None
    event_name: str
    device_id: str
    issued_at: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


class MessageDeduper:
    """Small in-memory duplicate detector for recently processed message ids."""

    def __init__(self, *, max_size: int = 2048, ttl_seconds: float = 24 * 60 * 60) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, float] = OrderedDict()

    def check_and_remember(self, message_id: str) -> bool:
        """Return True when the id is new; False when it was seen recently."""

        now = time.time()
        self._purge(now)
        if message_id in self._items:
            return False
        self._items[message_id] = now
        if len(self._items) > self.max_size:
            self._items.popitem(last=False)
        return True

    def _purge(self, now: float) -> None:
        expired = [
            message_id
            for message_id, seen_at in self._items.items()
            if now - seen_at > self.ttl_seconds
        ]
        for message_id in expired:
            self._items.pop(message_id, None)


def utc_now_iso() -> str:
    """Return an ISO 8601 UTC timestamp suitable for backend envelopes."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_envelope(
    *,
    device_id: str,
    event_name: str,
    payload: dict[str, Any] | None = None,
    correlation_id: str | None = None,
    request_id: str | None = None,
) -> Envelope:
    """Build a new outbound MQTT envelope while preserving workflow ids."""

    payload = payload or {}
    if not isinstance(payload, dict):
        raise EnvelopeError("Envelope payload must be a JSON object")
    workflow_correlation_id = correlation_id or payload.get("correlation_id") or str(uuid.uuid4())
    workflow_request_id = request_id if request_id is not None else payload.get("request_id")
    return Envelope(
        message_id=str(uuid.uuid4()),
        correlation_id=str(workflow_correlation_id),
        request_id=str(workflow_request_id) if workflow_request_id is not None else None,
        event_name=event_name,
        device_id=device_id,
        issued_at=utc_now_iso(),
        payload=payload,
    )


def parse_envelope(raw: bytes | str | dict[str, Any]) -> Envelope:
    """Parse and minimally validate a JSON MQTT envelope."""

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CommandValidationError("Payload is not valid JSON") from exc
    else:
        data = raw

    if not isinstance(data, dict):
        raise CommandValidationError("Envelope must be a JSON object")

    payload = data.get("payload")
    if not isinstance(payload, dict):
        raise CommandValidationError(
            "Envelope payload must be a JSON object",
            source_message_id=_string_or_none(data.get("message_id")),
            field="payload",
        )

    required = ["message_id", "correlation_id", "event_name", "device_id", "issued_at"]
    for field_name in required:
        if not str(data.get(field_name) or "").strip():
            raise CommandValidationError(
                f"Envelope missing required field: {field_name}",
                source_message_id=_string_or_none(data.get("message_id")),
                field=field_name,
            )

    request_id = data.get("request_id")
    return Envelope(
        message_id=str(data["message_id"]),
        correlation_id=str(data["correlation_id"]),
        request_id=str(request_id) if request_id is not None else None,
        event_name=str(data["event_name"]),
        device_id=str(data["device_id"]),
        issued_at=str(data["issued_at"]),
        payload=payload,
    )


def validate_command(
    envelope: Envelope,
    *,
    expected_device_id: str,
    expected_event_name: str | None = None,
    required_payload_fields: list[str] | tuple[str, ...] = (),
    deduper: MessageDeduper | None = None,
) -> Envelope:
    """Validate an inbound command envelope against the local device identity."""

    if envelope.device_id != expected_device_id:
        raise CommandValidationError(
            "Command envelope device_id does not match this device",
            source_message_id=envelope.message_id,
            field="device_id",
        )
    if expected_event_name is not None and envelope.event_name != expected_event_name:
        raise CommandValidationError(
            "Command envelope event_name does not match the command topic",
            source_message_id=envelope.message_id,
            field="event_name",
        )
    if deduper is not None and not deduper.check_and_remember(envelope.message_id):
        raise CommandValidationError(
            "Duplicate command message_id",
            source_message_id=envelope.message_id,
            field="message_id",
        )
    for field_name in required_payload_fields:
        if field_name not in envelope.payload:
            raise CommandValidationError(
                f"Command payload missing required field: {field_name}",
                source_message_id=envelope.message_id,
                field=field_name,
            )
    return envelope


def expected_event_name_for_topic(topic: str) -> str | None:
    """Return the expected command event name for a backend command topic."""

    marker = "/commands/"
    if marker not in topic:
        return None
    suffix = topic.split(marker, 1)[1].strip("/")
    return COMMAND_EVENT_BY_SUFFIX.get(suffix)


def required_payload_fields(event_name: str) -> tuple[str, ...]:
    """Return command-specific payload requirements from the backend contract."""

    if event_name == "inference.requested":
        return ("source_type", "source_reference")
    if event_name == "preview.start_requested":
        return ("camera_source",)
    if event_name == "preview.webrtc_signal":
        return ("request_id", "signal_type", "peer_id")
    if event_name == "model_deployment.requested":
        return ("target_device_id", "components")
    return ()


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
