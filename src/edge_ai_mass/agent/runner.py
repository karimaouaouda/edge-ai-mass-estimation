"""Runtime coordinator for the backend-facing edge device agent."""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import Any, Callable

from edge_ai_mass.agent.camera import CaptureAdapter
from edge_ai_mass.agent.config import AgentConfig
from edge_ai_mass.agent.envelopes import Envelope, new_envelope
from edge_ai_mass.agent.http_client import EdgeHttpClient
from edge_ai_mass.agent.inference import InferenceRunner
from edge_ai_mass.agent.media import MediaRenderer, MediaUploadJob
from edge_ai_mass.agent.model_manager import ModelDeploymentError, ModelManager
from edge_ai_mass.agent.mqtt_client import EdgeMqttClient
from edge_ai_mass.agent.outbox import FileOutbox
from edge_ai_mass.agent.preview import PreviewError, PreviewManager
from edge_ai_mass.agent.secrets import EnvironmentSecretStore, SecretStore
from edge_ai_mass.agent.telemetry import TelemetrySampler

logger = logging.getLogger(__name__)


class EdgeDeviceAgent:
    """Coordinate backend HTTP, MQTT, telemetry, and inference workflows."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        secret_store: SecretStore | None = None,
        http_client: EdgeHttpClient | None = None,
        mqtt_client: EdgeMqttClient | None = None,
        outbox: FileOutbox | None = None,
        telemetry: TelemetrySampler | None = None,
        inference_runner_factory: Callable[[], InferenceRunner] | None = None,
        preview_manager: PreviewManager | None = None,
        preview_camera_available: Callable[[str], bool] | None = None,
        model_manager: ModelManager | None = None,
        media_renderer: MediaRenderer | None = None,
    ) -> None:
        self.config = config
        self.secret_store = secret_store or EnvironmentSecretStore()
        self.http = http_client
        self.mqtt = mqtt_client
        self.outbox = outbox or FileOutbox(
            config.runtime.outbox_path,
            max_attempts=config.runtime.outbox_max_attempts,
            retention_seconds=config.runtime.outbox_retention_seconds,
        )
        self.telemetry = telemetry
        self.inference_runner_factory = inference_runner_factory
        camera_probe = preview_camera_available or CaptureAdapter().is_camera_connected
        self.preview_manager = preview_manager or PreviewManager(camera_available=camera_probe)
        self.active_models = dict(config.device.active_models)
        self.model_manager = model_manager or ModelManager(
            config.runtime.model_dir,
            active_models=self.active_models,
        )
        self.media_renderer = media_renderer or MediaRenderer(config.runtime.media_dir)
        self._inference_busy = False
        self._stop_requested = False
        self._inference_runner: InferenceRunner | None = None

    def start(self) -> None:
        """Initialize network clients and send the first telemetry payload."""

        self.outbox.start()
        if self.http is None:
            token = self.config.require_device_token(self.secret_store)
            self.http = EdgeHttpClient(
                base_url=self.config.backend.base_url,
                device_id=self.config.device.id,
                device_token=token,
                timeout_seconds=self.config.backend.request_timeout_seconds,
                max_retries=self.config.backend.max_retries,
                retry_backoff_seconds=self.config.backend.retry_backoff_seconds,
            )
        if self.telemetry is None:
            self.telemetry = TelemetrySampler(
                inference_busy=lambda: self._inference_busy,
                active_models=lambda: dict(self.active_models),
                disk_path=Path("."),
            )
        if self.mqtt is None:
            username, password = self.config.mqtt_credentials(self.secret_store)
            self.mqtt = EdgeMqttClient(
                config=self.config,
                on_command=self.handle_command,
                username=username,
                password=password,
            )
            self.mqtt.start()
        self.send_telemetry(status="online")

    def run_forever(self) -> None:
        """Run the long-lived edge agent loop."""

        self._install_signal_handlers()
        self.start()
        next_telemetry = time.monotonic() + self.config.runtime.telemetry_interval_seconds
        while not self._stop_requested:
            self._publish_preview_expiry_events()
            self.flush_outbox()
            now = time.monotonic()
            if now >= next_telemetry:
                self.send_telemetry(status="online")
                next_telemetry = now + self.config.runtime.telemetry_interval_seconds
            time.sleep(1.0)
        self.shutdown()

    def shutdown(self) -> None:
        self.preview_manager.stop_all()
        self.send_telemetry(status="offline")
        self.flush_outbox()
        if self.mqtt is not None:
            self.mqtt.stop()

    def send_telemetry(
        self,
        *,
        status: str = "online",
        correlation_id: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any] | None:
        logger.debug(
            "Preparing telemetry: status=%s correlation_id=%s request_id=%s",
            status,
            correlation_id,
            request_id,
        )
        if self.telemetry is None:
            self.telemetry = TelemetrySampler(
                inference_busy=lambda: self._inference_busy,
                active_models=lambda: dict(self.active_models),
            )
        payload = self.telemetry.sample(status=status)
        metadata = {
            "payload": payload,
            "correlation_id": correlation_id,
            "request_id": request_id,
        }
        try:
            assert self.http is not None
            return self.http.submit_telemetry(payload)
        except Exception as exc:
            logger.warning("Telemetry submission failed; queued locally: %s", exc)
            workflow_id = correlation_id or request_id or ""
            self.outbox.enqueue("telemetry", metadata, workflow_id=workflow_id)
            return None

    def handle_command(self, envelope: Envelope) -> None:
        logger.info(
            "Handling command: event=%s message_id=%s correlation_id=%s request_id=%s",
            envelope.event_name,
            envelope.message_id,
            envelope.correlation_id,
            envelope.request_id,
        )
        if envelope.event_name == "inference.requested":
            self._handle_inference(envelope)
        elif envelope.event_name == "preview.start_requested":
            self._handle_preview_start(envelope)
        elif envelope.event_name == "preview.webrtc_signal":
            self._handle_preview_signal(envelope)
        elif envelope.event_name == "preview.stop_requested":
            self._handle_preview_stop(envelope)
        elif envelope.event_name == "model_deployment.requested":
            self._handle_model_deployment(envelope)
        else:
            self.publish_event(
                "device.error",
                _error_payload(
                    "invalid_command",
                    f"Unsupported command event: {envelope.event_name}",
                    retryable=False,
                ),
                correlation_id=envelope.correlation_id,
                request_id=envelope.request_id,
            )

    def publish_event(
        self,
        event_name: str,
        payload: dict[str, Any],
        *,
        correlation_id: str | None,
        request_id: str | None,
    ) -> Envelope | None:
        envelope = new_envelope(
            device_id=self.config.device.id,
            event_name=event_name,
            payload=payload,
            correlation_id=correlation_id,
            request_id=request_id,
        )
        if self.mqtt is not None and self.mqtt.is_connected:
            try:
                self.mqtt.publish_envelope(envelope)
                return envelope
            except Exception as exc:
                logger.exception(
                    "MQTT event publish failed; queueing event=%s message_id=%s: %s",
                    event_name,
                    envelope.message_id,
                    exc,
                )
        else:
            logger.warning(
                "MQTT unavailable; queueing event=%s message_id=%s started=%s connected=%s",
                event_name,
                envelope.message_id,
                self.mqtt.is_started if self.mqtt is not None else False,
                self.mqtt.is_connected if self.mqtt is not None else False,
            )
        self.outbox.enqueue(
            "mqtt_event",
            {"envelope": envelope.to_dict()},
            workflow_id=correlation_id or request_id or "",
        )
        logger.info(
            "MQTT event persisted to outbox: event=%s message_id=%s",
            event_name,
            envelope.message_id,
        )
        return envelope

    def flush_outbox(self) -> None:
        for record in self.outbox.due_records():
            try:
                if record.kind == "telemetry":
                    assert self.http is not None
                    self.http.submit_telemetry(record.payload["payload"])
                elif record.kind == "media_upload":
                    assert self.http is not None
                    job = record.payload
                    self.http.upload_media(
                        job["path"],
                        media_type=job["media_type"],
                        request_id=job.get("request_id"),
                        correlation_id=job.get("correlation_id"),
                        metadata=job.get("metadata") or {},
                    )
                elif record.kind == "mqtt_event":
                    if self.mqtt is None or not self.mqtt.is_connected:
                        raise RuntimeError("MQTT client is not connected")
                    logger.debug(
                        "Retrying MQTT outbox event: record_id=%s event=%s message_id=%s",
                        record.id,
                        record.payload["envelope"].get("event_name"),
                        record.payload["envelope"].get("message_id"),
                    )
                    self.mqtt.publish_envelope(Envelope(**record.payload["envelope"]))
                else:
                    self.outbox.mark_failed(
                        record,
                        f"unknown outbox kind {record.kind}",
                        permanent=True,
                    )
                    continue
            except Exception as exc:
                logger.warning(
                    "Outbox delivery failed: record_id=%s kind=%s attempts=%s error=%s",
                    record.id,
                    record.kind,
                    record.attempts,
                    exc,
                )
                self.outbox.mark_failed(record, str(exc))
            else:
                self.outbox.mark_sent(record)
                logger.info(
                    "Outbox delivery completed: record_id=%s kind=%s",
                    record.id,
                    record.kind,
                )

    def _handle_inference(self, envelope: Envelope) -> None:
        request_id = _request_id(envelope)
        correlation_id = envelope.correlation_id
        self._inference_busy = True
        self.send_telemetry(
            status="online",
            correlation_id=correlation_id,
            request_id=request_id,
        )
        try:
            runner = self._get_inference_runner()
            result = runner.run_command(
                envelope.payload,
                request_id=request_id,
                correlation_id=correlation_id,
            )
            self.publish_event(
                "inference.result",
                result.payload,
                correlation_id=correlation_id,
                request_id=request_id,
            )
            self._upload_requested_media(envelope, result)
        except Exception as exc:
            logger.exception("Inference command failed")
            self.publish_event(
                "inference.failed",
                _error_payload(
                    "inference_failed",
                    str(exc),
                    request_id=request_id,
                    correlation_id=correlation_id,
                    retryable=False,
                ),
                correlation_id=correlation_id,
                request_id=request_id,
            )
        finally:
            self._inference_busy = False
            self.send_telemetry(
                status="online",
                correlation_id=correlation_id,
                request_id=request_id,
            )

    def _handle_preview_start(self, envelope: Envelope) -> None:
        request_id = _request_id(envelope)
        try:
            payload = self.preview_manager.start(
                envelope.payload,
                request_id=request_id,
                correlation_id=envelope.correlation_id,
                preview_enabled=bool(self.config.device.capabilities.get("preview", False)),
            )
            self.publish_event(
                "preview.ready",
                payload,
                correlation_id=envelope.correlation_id,
                request_id=request_id,
            )
        except PreviewError as exc:
            self.publish_event(
                "preview.failed",
                _preview_failure_payload(
                    exc.error_type,
                    exc.summary,
                    request_id=request_id,
                    correlation_id=envelope.correlation_id,
                    retryable=exc.retryable,
                ),
                correlation_id=envelope.correlation_id,
                request_id=request_id,
            )
        except Exception as exc:
            self.publish_event(
                "preview.failed",
                _preview_failure_payload(
                    "preview_internal_error",
                    str(exc),
                    request_id=request_id,
                    correlation_id=envelope.correlation_id,
                    retryable=False,
                ),
                correlation_id=envelope.correlation_id,
                request_id=request_id,
            )

    def _handle_preview_signal(self, envelope: Envelope) -> None:
        request_id = _request_id(envelope)
        try:
            events = self.preview_manager.handle_signal(
                envelope.payload,
                request_id=request_id,
                correlation_id=envelope.correlation_id,
            )
            for event in events:
                self.publish_event(
                    event.event_name,
                    event.payload,
                    correlation_id=envelope.correlation_id,
                    request_id=request_id,
                )
        except PreviewError as exc:
            self.publish_event(
                "preview.failed",
                _preview_failure_payload(
                    exc.error_type,
                    exc.summary,
                    request_id=request_id,
                    correlation_id=envelope.correlation_id,
                    retryable=exc.retryable,
                ),
                correlation_id=envelope.correlation_id,
                request_id=request_id,
            )
        except Exception as exc:
            self.publish_event(
                "preview.failed",
                _preview_failure_payload(
                    "preview_internal_error",
                    str(exc),
                    request_id=request_id,
                    correlation_id=envelope.correlation_id,
                    retryable=False,
                ),
                correlation_id=envelope.correlation_id,
                request_id=request_id,
            )

    def _handle_preview_stop(self, envelope: Envelope) -> None:
        request_id = _request_id(envelope)
        payload = self.preview_manager.stop(
            envelope.payload,
            request_id=request_id,
            correlation_id=envelope.correlation_id,
        )
        self.publish_event(
            "preview.stopped",
            payload,
            correlation_id=envelope.correlation_id,
            request_id=request_id,
        )

    def _publish_preview_expiry_events(self) -> None:
        for event in self.preview_manager.expire_sessions():
            self.publish_event(
                event.event_name,
                event.payload,
                correlation_id=event.payload.get("correlation_id"),
                request_id=event.payload.get("request_id"),
            )

    def _handle_model_deployment(self, envelope: Envelope) -> None:
        request_id = _request_id(envelope)
        correlation_id = envelope.correlation_id
        self.publish_event(
            "model_deployment.started",
            {
                "deployment_id": envelope.payload.get("deployment_id"),
                "request_id": request_id,
                "correlation_id": correlation_id,
            },
            correlation_id=correlation_id,
            request_id=request_id,
        )
        try:
            payload = self.model_manager.deploy(envelope.payload, device_id=self.config.device.id)
        except ModelDeploymentError as exc:
            self.publish_event(
                "model_deployment.failed",
                _error_payload(
                    "model_download_failed",
                    str(exc),
                    request_id=request_id,
                    correlation_id=correlation_id,
                    retryable=False,
                ),
                correlation_id=correlation_id,
                request_id=request_id,
            )
            return

        self.active_models.update(payload.get("active_models") or {})
        self.publish_event(
            "model_deployment.succeeded",
            payload,
            correlation_id=correlation_id,
            request_id=request_id,
        )
        self.send_telemetry(
            status="online",
            correlation_id=correlation_id,
            request_id=request_id,
        )

    def _upload_requested_media(self, envelope: Envelope, result: Any) -> None:
        options = envelope.payload.get("options")
        options = options if isinstance(options, dict) else {}
        jobs = self.media_renderer.render_requested(
            image=result.captured.image,
            result=result.pipeline_result,
            request_id=_request_id(envelope),
            correlation_id=envelope.correlation_id,
            return_annotated_image=bool(options.get("return_annotated_image")),
            return_depth_preview=bool(options.get("return_depth_preview")),
        )
        for job in jobs:
            self._upload_media_job(job)

    def _upload_media_job(self, job: MediaUploadJob) -> None:
        try:
            assert self.http is not None
            self.http.upload_media(
                job.path,
                media_type=job.media_type,
                request_id=job.request_id,
                correlation_id=job.correlation_id,
                metadata=job.metadata,
            )
        except Exception as exc:
            logger.warning("Media upload failed; queued locally: %s", exc)
            self.outbox.enqueue(
                "media_upload",
                {
                    "path": str(job.path),
                    "media_type": job.media_type,
                    "request_id": job.request_id,
                    "correlation_id": job.correlation_id,
                    "metadata": job.metadata,
                },
                workflow_id=job.correlation_id or job.request_id or "",
            )

    def _get_inference_runner(self) -> InferenceRunner:
        if self._inference_runner is None:
            if self.inference_runner_factory is not None:
                self._inference_runner = self.inference_runner_factory()
            else:
                from edge_ai_mass.pipeline.factory import build_pipeline

                pipeline = build_pipeline(self.config.runtime.pipeline_config)
                pipeline.load_all()
                self._inference_runner = InferenceRunner(
                    pipeline=pipeline,
                    active_models=self.active_models,
                )
        return self._inference_runner

    def _install_signal_handlers(self) -> None:
        def _request_stop(_signum: int, _frame: object) -> None:
            self._stop_requested = True

        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)


def _request_id(envelope: Envelope) -> str:
    return str(envelope.request_id or envelope.payload.get("request_id") or "")


def _error_payload(
    error_type: str,
    summary: str,
    *,
    request_id: str | None = None,
    correlation_id: str | None = None,
    retryable: bool,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error_type": error_type,
        "summary": summary[:500],
        "retryable": retryable,
        "details": details or {},
    }
    if request_id:
        payload["request_id"] = request_id
    if correlation_id:
        payload["correlation_id"] = correlation_id
    return payload


def _preview_failure_payload(
    error_type: str,
    summary: str,
    *,
    request_id: str | None,
    correlation_id: str | None,
    retryable: bool,
) -> dict[str, Any]:
    from edge_ai_mass.agent.envelopes import utc_now_iso

    payload = _error_payload(
        error_type,
        summary,
        request_id=request_id,
        correlation_id=correlation_id,
        retryable=retryable,
    )
    payload["failed_at"] = utc_now_iso()
    return payload
