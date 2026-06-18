"""MQTT command subscriber and event publisher for the Laravel contract."""

from __future__ import annotations

import logging
from typing import Any, Callable

from edge_ai_mass.agent.config import AgentConfig
from edge_ai_mass.agent.envelopes import (
    CommandValidationError,
    Envelope,
    MessageDeduper,
    expected_event_name_for_topic,
    new_envelope,
    parse_envelope,
    required_payload_fields,
    validate_command,
)

logger = logging.getLogger(__name__)

CommandHandler = Callable[[Envelope], None]


class EdgeMqttClient:
    """MQTT backend adapter for device command and event exchange."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        on_command: CommandHandler,
        username: str = "",
        password: str = "",
        deduper: MessageDeduper | None = None,
    ) -> None:
        self.config = config
        self.on_command = on_command
        self.username = username
        self.password = password
        self.deduper = deduper or MessageDeduper()
        self._client: Any = None

    @property
    def command_topic(self) -> str:
        return self.topic("commands/#")

    def topic(self, suffix: str) -> str:
        suffix = suffix.strip("/")
        return f"{self.config.mqtt.topic_prefix}/devices/{self.config.device.id}/{suffix}"

    def start(self) -> None:
        """Connect to the broker and subscribe to backend command topics."""

        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError(
                "Backend MQTT mode requires paho-mqtt. Install with: "
                "pip install -e '.[orchestration]'"
            ) from exc

        try:
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.config.mqtt.client_id,
            )
        except AttributeError:
            client = mqtt.Client(client_id=self.config.mqtt.client_id)

        if self.username:
            client.username_pw_set(self.username, self.password or None)
        if self.config.mqtt.tls:
            client.tls_set()

        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.connect(
            self.config.mqtt.host,
            self.config.mqtt.port,
            keepalive=self.config.mqtt.keepalive_seconds,
        )
        client.loop_start()
        self._client = client

    def stop(self) -> None:
        if self._client is None:
            return
        self._client.loop_stop()
        self._client.disconnect()
        self._client = None

    def publish_event(
        self,
        event_name: str,
        payload: dict[str, Any],
        *,
        correlation_id: str | None = None,
        request_id: str | None = None,
    ) -> Envelope:
        envelope = new_envelope(
            device_id=self.config.device.id,
            event_name=event_name,
            payload=payload,
            correlation_id=correlation_id,
            request_id=request_id,
        )
        self._publish_envelope(f"events/{event_name}", envelope)
        return envelope

    def publish_envelope(self, envelope: Envelope) -> None:
        """Publish a previously persisted event envelope."""

        self._publish_envelope(f"events/{envelope.event_name}", envelope)

    @property
    def is_started(self) -> bool:
        return self._client is not None

    def publish_error(
        self,
        payload: dict[str, Any],
        *,
        correlation_id: str | None = None,
        request_id: str | None = None,
    ) -> Envelope:
        return self.publish_event(
            "device.error",
            payload,
            correlation_id=correlation_id,
            request_id=request_id,
        )

    def handle_message(self, topic: str, payload: bytes | str) -> None:
        """Parse, validate, and dispatch a backend command message."""

        envelope: Envelope | None = None
        try:
            envelope = parse_envelope(payload)
            expected_event_name = expected_event_name_for_topic(topic)
            if expected_event_name is None:
                raise CommandValidationError(
                    "MQTT topic is not a supported command topic",
                    source_message_id=envelope.message_id,
                    field="topic",
                )
            validate_command(
                envelope,
                expected_device_id=self.config.device.id,
                expected_event_name=expected_event_name,
                required_payload_fields=required_payload_fields(expected_event_name),
                deduper=self.deduper,
            )
        except CommandValidationError as exc:
            logger.warning("Rejected MQTT command: %s", exc.summary)
            self.publish_error(
                exc.to_error_payload(),
                correlation_id=envelope.correlation_id if envelope is not None else None,
                request_id=envelope.request_id if envelope is not None else None,
            )
            return

        self.on_command(envelope)

    def _publish_envelope(self, suffix: str, envelope: Envelope) -> None:
        if self._client is None:
            logger.info("MQTT client is not connected; event not published: %s", envelope.event_name)
            return
        self._client.publish(
            self.topic(suffix),
            envelope.to_json(),
            qos=self.config.mqtt.qos,
            retain=False,
        )

    def _on_connect(
        self,
        client: Any,
        _userdata: Any,
        _flags: Any,
        reason_code: Any,
        *_extra: Any,
    ) -> None:
        if int(reason_code) != 0:
            logger.error("MQTT connection failed with rc=%s", reason_code)
            return
        client.subscribe(self.command_topic, qos=self.config.mqtt.qos)
        logger.info("Subscribed to backend MQTT commands on %s", self.command_topic)

    def _on_message(self, _client: Any, _userdata: Any, message: Any) -> None:
        self.handle_message(str(message.topic), message.payload)
