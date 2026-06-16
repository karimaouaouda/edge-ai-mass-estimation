"""MQTT trigger listener for the orchestrator."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from edge_ai_mass.orchestration.config import MQTTSettings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BrokerCommand:
    action: str = "check"
    target: str | None = None
    force: bool = False
    release_tag: str | None = None
    source_topic: str = ""


class MQTTUpdateListener:
    """Subscribe to update commands from a Mosquitto-compatible broker."""

    def __init__(
        self,
        settings: MQTTSettings,
        device_id: str,
        on_command: Callable[[BrokerCommand], None],
    ) -> None:
        self.settings = settings
        self.device_id = device_id
        self.on_command = on_command
        self._client: Any = None

    def start(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError(
                "MQTT mode requires paho-mqtt. Install with: pip install -e '.[orchestration]'"
            ) from exc

        client = mqtt.Client(client_id=self.settings.client_id)
        if self.settings.username:
            client.username_pw_set(self.settings.username, self.settings.password or None)
        if self.settings.tls:
            client.tls_set()

        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.connect(
            self.settings.host,
            self.settings.port,
            keepalive=self.settings.keepalive_seconds,
        )
        client.loop_start()
        self._client = client
        logger.info("MQTT listener connected to %s:%s", self.settings.host, self.settings.port)

    def stop(self) -> None:
        if self._client is None:
            return
        self._client.loop_stop()
        self._client.disconnect()
        self._client = None

    def publish_status(self, payload: dict[str, Any]) -> None:
        if self._client is None:
            return
        topic = self.settings.status_topic.format(device_id=self.device_id)
        self._client.publish(topic, json.dumps(payload), qos=1, retain=False)

    def _on_connect(self, client: Any, _userdata: Any, _flags: Any, rc: int, *_extra: Any) -> None:
        if rc != 0:
            logger.error("MQTT connection failed with rc=%s", rc)
            return
        for topic in self.settings.topics:
            client.subscribe(topic, qos=1)
            logger.info("Subscribed to MQTT update topic %s", topic)

    def _on_message(self, _client: Any, _userdata: Any, message: Any) -> None:
        payload = message.payload.decode("utf-8", errors="replace").strip()
        try:
            data = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            data = {"action": payload}

        command = BrokerCommand(
            action=str(data.get("action") or data.get("command") or "check").lower(),
            target=data.get("target"),
            force=_as_bool(data.get("force", False)),
            release_tag=data.get("release") or data.get("release_tag"),
            source_topic=message.topic,
        )
        logger.info("Received MQTT update command: %s", command)
        self.on_command(command)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
