"""MQTT command subscriber and event publisher for the Laravel contract."""

from __future__ import annotations

import logging
import threading
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


class MqttPublishError(RuntimeError):
    """Raised when an MQTT event cannot be queued for delivery."""


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
        self._connected = threading.Event()
        self._pending_publishes: dict[int, dict[str, str]] = {}

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
        client.on_connect_fail = self._on_connect_fail
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        client.on_publish = self._on_publish
        client.on_subscribe = self._on_subscribe
        client.on_log = self._on_log
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client = client

        logger.info(
            "Starting MQTT client: broker=%s:%s client_id=%s tls=%s qos=%s "
            "username_configured=%s command_topic=%s",
            self.config.mqtt.host,
            self.config.mqtt.port,
            self.config.mqtt.client_id,
            self.config.mqtt.tls,
            self.config.mqtt.qos,
            bool(self.username),
            self.command_topic,
        )
        try:
            result_code = client.connect(
                self.config.mqtt.host,
                self.config.mqtt.port,
                keepalive=self.config.mqtt.keepalive_seconds,
            )
            logger.debug("MQTT connect() queued with rc=%s", result_code)
            client.loop_start()
        except Exception:
            self._client = None
            self._connected.clear()
            logger.exception(
                "MQTT startup failed for broker %s:%s",
                self.config.mqtt.host,
                self.config.mqtt.port,
            )
            raise

    def stop(self) -> None:
        if self._client is None:
            return
        logger.info("Stopping MQTT client %s", self.config.mqtt.client_id)
        self._connected.clear()
        self._client.disconnect()
        self._client.loop_stop()
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

    @property
    def is_connected(self) -> bool:
        """Return whether the broker has accepted the current connection."""

        return self._client is not None and self._connected.is_set()

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
            logger.info(
                "Received MQTT command: topic=%s event=%s message_id=%s "
                "correlation_id=%s request_id=%s",
                topic,
                envelope.event_name,
                envelope.message_id,
                envelope.correlation_id,
                envelope.request_id,
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
        topic = self.topic(suffix)
        if self._client is None:
            raise MqttPublishError(
                f"MQTT client is not started; event {envelope.event_name!r} "
                f"was not published to {topic}"
            )
        if not self._connected.is_set():
            raise MqttPublishError(
                f"MQTT broker is not connected; event {envelope.event_name!r} "
                f"was not published to {topic}"
            )

        encoded = envelope.to_json()
        logger.debug(
            "MQTT publish attempt: topic=%s event=%s message_id=%s qos=%s bytes=%s",
            topic,
            envelope.event_name,
            envelope.message_id,
            self.config.mqtt.qos,
            len(encoded.encode("utf-8")),
        )
        result = self._client.publish(
            topic,
            encoded,
            qos=self.config.mqtt.qos,
            retain=False,
        )
        result_code = int(getattr(result, "rc", 0))
        message_id = int(getattr(result, "mid", 0))
        if result_code != 0:
            logger.error(
                "MQTT publish rejected locally: topic=%s event=%s message_id=%s rc=%s mid=%s",
                topic,
                envelope.event_name,
                envelope.message_id,
                result_code,
                message_id,
            )
            raise MqttPublishError(
                f"MQTT publish failed with rc={result_code} for topic {topic}"
            )

        self._pending_publishes[message_id] = {
            "topic": topic,
            "event_name": envelope.event_name,
            "message_id": envelope.message_id,
        }
        logger.info(
            "MQTT publish queued: topic=%s event=%s message_id=%s mid=%s "
            "correlation_id=%s request_id=%s",
            topic,
            envelope.event_name,
            envelope.message_id,
            message_id,
            envelope.correlation_id,
            envelope.request_id,
        )

    def _on_connect(
        self,
        client: Any,
        _userdata: Any,
        _flags: Any,
        reason_code: Any,
        *_extra: Any,
    ) -> None:
        try:
            numeric_reason = _reason_code_value(reason_code)
            if numeric_reason != 0:
                self._connected.clear()
                logger.error(
                    "MQTT connection rejected: broker=%s:%s client_id=%s reason=%s value=%s",
                    self.config.mqtt.host,
                    self.config.mqtt.port,
                    self.config.mqtt.client_id,
                    reason_code,
                    numeric_reason,
                )
                return
            self._connected.set()
            logger.info(
                "MQTT connected: broker=%s:%s client_id=%s reason=%s",
                self.config.mqtt.host,
                self.config.mqtt.port,
                self.config.mqtt.client_id,
                reason_code,
            )
            subscribe_result = client.subscribe(
                self.command_topic,
                qos=self.config.mqtt.qos,
            )
            subscribe_code, subscribe_mid = _subscribe_result(subscribe_result)
            if subscribe_code != 0:
                logger.error(
                    "MQTT subscription failed: topic=%s rc=%s mid=%s",
                    self.command_topic,
                    subscribe_code,
                    subscribe_mid,
                )
                return
            logger.info(
                "MQTT subscription queued: topic=%s qos=%s mid=%s",
                self.command_topic,
                self.config.mqtt.qos,
                subscribe_mid,
            )
        except Exception as exc:
            self._connected.clear()
            logger.exception("Error in MQTT on_connect handler: %s", exc)

    def _on_message(self, _client: Any, _userdata: Any, message: Any) -> None:
        logger.debug(
            "MQTT message received: topic=%s qos=%s retain=%s bytes=%s",
            message.topic,
            getattr(message, "qos", None),
            getattr(message, "retain", None),
            len(message.payload),
        )
        self.handle_message(str(message.topic), message.payload)

    def _on_publish(
        self,
        _client: Any,
        _userdata: Any,
        mid: int,
        reason_code: Any = None,
        *_extra: Any,
    ) -> None:
        details = self._pending_publishes.pop(int(mid), {})
        logger.info(
            "MQTT publish completed: topic=%s event=%s message_id=%s mid=%s reason=%s",
            details.get("topic", "unknown"),
            details.get("event_name", "unknown"),
            details.get("message_id", "unknown"),
            mid,
            reason_code,
        )

    def _on_subscribe(
        self,
        _client: Any,
        _userdata: Any,
        mid: int,
        reason_codes: Any = None,
        *_extra: Any,
    ) -> None:
        logger.info(
            "MQTT subscription acknowledged: topic=%s mid=%s reason_codes=%s",
            self.command_topic,
            mid,
            reason_codes,
        )

    def _on_connect_fail(self, _client: Any, _userdata: Any) -> None:
        self._connected.clear()
        logger.error(
            "MQTT TCP connection failed: broker=%s:%s client_id=%s",
            self.config.mqtt.host,
            self.config.mqtt.port,
            self.config.mqtt.client_id,
        )

    def _on_disconnect(self, _client: Any, _userdata: Any, *args: Any) -> None:
        self._connected.clear()
        reason_code = _disconnect_reason(args)
        numeric_reason = _reason_code_value(reason_code)
        log = logger.info if numeric_reason == 0 else logger.warning
        log(
            "MQTT disconnected: broker=%s:%s client_id=%s reason=%s value=%s",
            self.config.mqtt.host,
            self.config.mqtt.port,
            self.config.mqtt.client_id,
            reason_code,
            numeric_reason,
        )

    def _on_log(
        self,
        _client: Any,
        _userdata: Any,
        level: Any,
        message: str,
    ) -> None:
        logger.debug("Paho MQTT: level=%s message=%s", level, message)


def _reason_code_value(reason_code: Any) -> int:
    value = getattr(reason_code, "value", reason_code)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0 if str(reason_code).lower() == "success" else -1


def _subscribe_result(result: Any) -> tuple[int, int]:
    if isinstance(result, tuple) and len(result) >= 2:
        return int(result[0]), int(result[1])
    return int(getattr(result, "rc", -1)), int(getattr(result, "mid", 0))


def _disconnect_reason(args: tuple[Any, ...]) -> Any:
    if len(args) >= 2:
        return args[1]
    if args:
        return args[0]
    return "unknown"
