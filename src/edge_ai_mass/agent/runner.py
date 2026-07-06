"""Runtime coordinator for the backend-facing edge device agent."""

from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path
from typing import Any, Callable

from edge_ai_mass.agent.camera import CaptureAdapter
from edge_ai_mass.agent.config import AgentConfig
from edge_ai_mass.agent.console import print_panel, print_stage_update
from edge_ai_mass.agent.envelopes import Envelope, new_envelope
from edge_ai_mass.agent.firmware import FirmwareUpdateError, FirmwareUpdateManager
from edge_ai_mass.agent.http_client import EdgeHttpClient
from edge_ai_mass.agent.inference import InferenceRunner, InferenceStageReporter
from edge_ai_mass.agent.media import MediaRenderer, MediaUploadJob
from edge_ai_mass.agent.model_manager import ModelDeploymentError, ModelManager
from edge_ai_mass.agent.mqtt_client import EdgeMqttClient
from edge_ai_mass.agent.outbox import FileOutbox
from edge_ai_mass.agent.preview import (
    DeferredAiortcPreviewPeerFactory,
    PreviewError,
    PreviewManager,
    SimulatedPreviewPeerFactory,
)
from edge_ai_mass.agent.secrets import EnvironmentSecretStore, SecretStore
from edge_ai_mass.agent.telemetry import TelemetrySampler

logger = logging.getLogger(__name__)

PIPELINE_MODEL_TASKS = {"detector", "depth", "material", "mass"}


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
        auto_updater_factory: Callable[[], Any] | None = None,
        preview_manager: PreviewManager | None = None,
        preview_camera_available: Callable[[str], bool] | None = None,
        model_manager: ModelManager | None = None,
        media_renderer: MediaRenderer | None = None,
        firmware_manager: FirmwareUpdateManager | None = None,
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
        self.auto_updater_factory = auto_updater_factory
        camera_probe = preview_camera_available or CaptureAdapter().is_camera_connected
        self.preview_manager = preview_manager or self._build_preview_manager(camera_probe)
        self.active_models = dict(config.device.active_models)
        self.model_manager = model_manager or ModelManager(
            config.runtime.model_dir,
            active_models=self.active_models,
        )
        self.media_renderer = media_renderer or MediaRenderer(config.runtime.media_dir)
        self.firmware_manager = firmware_manager or FirmwareUpdateManager(
            device_type=config.device.type,
            current_version=config.device.firmware_version,
        )
        self._inference_busy = False
        self._update_check_busy = False
        self._stop_requested = False
        self._inference_runner: InferenceRunner | None = None
        self._auto_updater: Any | None = None
        self._auto_update_interval_seconds = 0.0
        self._next_update_check: float | None = None

    def start(self) -> None:
        """Initialize network clients and send the first telemetry payload."""

        self._start_auto_update_runtime()
        if self.config.updates.enabled and self.config.updates.run_on_start:
            self._check_for_updates(reason="startup")
        self._preload_inference_runtime()
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
        if not self.mqtt.is_started:
            self.mqtt.start()
        self.send_telemetry(status="online")

    def run_forever(self) -> None:
        """Run the long-lived edge agent loop."""

        self._install_signal_handlers()
        self.start()
        next_telemetry = time.monotonic() + self.config.runtime.telemetry_interval_seconds
        while not self._stop_requested:
            self._maybe_check_for_updates()
            self._publish_preview_runtime_events()
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
        transport = self.config.runtime.telemetry_transport
        http_response = None
        if transport in {"http", "both"}:
            http_response = self._submit_http_telemetry(
                payload,
                correlation_id=correlation_id,
                request_id=request_id,
            )
        if transport in {"mqtt", "both"}:
            self._submit_mqtt_telemetry(
                payload,
                correlation_id=correlation_id,
                request_id=request_id,
            )
        return http_response

    def _submit_http_telemetry(
        self,
        payload: dict[str, Any],
        *,
        correlation_id: str | None,
        request_id: str | None,
    ) -> dict[str, Any] | None:
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

    def _submit_mqtt_telemetry(
        self,
        payload: dict[str, Any],
        *,
        correlation_id: str | None,
        request_id: str | None,
    ) -> None:
        envelope = new_envelope(
            device_id=self.config.device.id,
            event_name="telemetry.reported",
            payload=payload,
            correlation_id=correlation_id,
            request_id=request_id,
        )
        if self.mqtt is not None and self.mqtt.is_connected:
            try:
                self.mqtt.publish_telemetry_envelope(envelope)
                return
            except Exception as exc:
                logger.exception(
                    "MQTT telemetry publish failed; queueing message_id=%s: %s",
                    envelope.message_id,
                    exc,
                )
        else:
            logger.warning(
                "MQTT unavailable; queueing telemetry message_id=%s connected=%s",
                envelope.message_id,
                self.mqtt.is_connected if self.mqtt is not None else False,
            )
        self.outbox.enqueue(
            "mqtt_telemetry",
            {"envelope": envelope.to_dict()},
            workflow_id=correlation_id or request_id or "",
        )
        logger.info(
            "MQTT telemetry persisted to outbox: message_id=%s",
            envelope.message_id,
        )

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
        elif envelope.event_name == "firmware.update_requested":
            self._handle_firmware_update(envelope)
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
        workflow_id = correlation_id or request_id or ""
        pending_predecessor = self.outbox.has_pending(
            "mqtt_event",
            workflow_id=workflow_id,
        )
        if pending_predecessor:
            logger.info(
                "MQTT event follows pending workflow event; queueing event=%s message_id=%s",
                event_name,
                envelope.message_id,
            )
        elif self.mqtt is not None and self.mqtt.is_connected:
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
            workflow_id=workflow_id,
        )
        logger.info(
            "MQTT event persisted to outbox: event=%s message_id=%s",
            event_name,
            envelope.message_id,
        )
        return envelope

    def flush_outbox(self) -> None:
        blocked_workflows: set[str] = set()
        for record in self.outbox.due_records():
            if (
                record.kind == "mqtt_event"
                and record.workflow_id
                and record.workflow_id in blocked_workflows
            ):
                continue
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
                elif record.kind in {"mqtt_event", "mqtt_telemetry"}:
                    if self.mqtt is None or not self.mqtt.is_connected:
                        raise RuntimeError("MQTT client is not connected")
                    logger.debug(
                        "Retrying MQTT outbox message: record_id=%s event=%s message_id=%s",
                        record.id,
                        record.payload["envelope"].get("event_name"),
                        record.payload["envelope"].get("message_id"),
                    )
                    envelope = Envelope(**record.payload["envelope"])
                    if record.kind == "mqtt_telemetry":
                        self.mqtt.publish_telemetry_envelope(envelope)
                    else:
                        self.mqtt.publish_envelope(envelope)
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
                if record.kind == "mqtt_event" and record.workflow_id:
                    blocked_workflows.add(record.workflow_id)
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
        stage_reporter = InferenceStageReporter(
            request_id=request_id,
            correlation_id=correlation_id,
            publish=lambda payload: self.publish_event(
                "inference.stage",
                payload,
                correlation_id=correlation_id,
                request_id=request_id,
            ),
            console=print_stage_update,
        )
        self._inference_busy = True
        self.send_telemetry(
            status="online",
            correlation_id=correlation_id,
            request_id=request_id,
        )

        print_panel(
            "Inference request",
            {
                "correlation_id": correlation_id,
                "request_id": request_id,
                "source_type": envelope.payload.get("source_type", "camera"),
                "source_reference": envelope.payload.get("source_reference", "camera:0"),
            },
            status="received",
        )
        try:
            runner = self._get_inference_runner()
            result = runner.run_command(
                envelope.payload,
                request_id=request_id,
                correlation_id=correlation_id,
                stage_reporter=stage_reporter,
            )
            stage_reporter.update("media-output", "running")
            media_summary = self._upload_requested_media(envelope, result)
            stage_reporter.update("media-output", "completed", media_summary)
            self.publish_event(
                "inference.result",
                result.payload,
                correlation_id=correlation_id,
                request_id=request_id,
            )
        except Exception as exc:
            logger.exception("Inference command failed")
            print_panel(
                "Inference command",
                {
                    "correlation_id": correlation_id,
                    "request_id": request_id,
                    "error": exc,
                },
                status="failed",
            )
            stage_reporter.fail_active(exc)
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

    def _handle_firmware_update(self, envelope: Envelope) -> None:
        request_id = _request_id(envelope)
        correlation_id = envelope.correlation_id

        def publish_progress(payload: dict[str, Any]) -> None:
            self.publish_event(
                "firmware.update.progress",
                payload,
                correlation_id=correlation_id,
                request_id=request_id,
            )

        try:
            payload = self.firmware_manager.apply(
                envelope.payload,
                progress=publish_progress,
            )
        except FirmwareUpdateError as exc:
            self.publish_event(
                "firmware.update.failed",
                _error_payload(
                    exc.error_type,
                    exc.summary,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    retryable=exc.retryable,
                ),
                correlation_id=correlation_id,
                request_id=request_id,
            )
            return
        except Exception as exc:
            logger.exception("Firmware update command failed")
            self.publish_event(
                "firmware.update.failed",
                _error_payload(
                    "firmware_update_failed",
                    str(exc),
                    request_id=request_id,
                    correlation_id=correlation_id,
                    retryable=False,
                ),
                correlation_id=correlation_id,
                request_id=request_id,
            )
            return

        self.publish_event(
            "firmware.update.succeeded",
            payload,
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

    def _publish_preview_runtime_events(self) -> None:
        events = self.preview_manager.poll_events()
        events.extend(self.preview_manager.expire_sessions())
        for event in events:
            self.publish_event(
                event.event_name,
                event.payload,
                correlation_id=event.payload.get("correlation_id"),
                request_id=event.payload.get("request_id"),
            )

    def _build_preview_manager(
        self,
        camera_probe: Callable[[str], bool],
    ) -> PreviewManager:
        if self.config.runtime.preview_backend == "simulated":
            peer_factory = SimulatedPreviewPeerFactory()
        else:
            peer_factory = DeferredAiortcPreviewPeerFactory()
        return PreviewManager(
            peer_factory=peer_factory,
            camera_available=camera_probe,
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
        activated_components = payload.get("activated_components") or []
        self._apply_active_model_path_overrides(activated_components)
        if any(
            str(component.get("task") or "") in PIPELINE_MODEL_TASKS
            for component in activated_components
            if isinstance(component, dict)
        ):
            self._inference_runner = None
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

    def _start_auto_update_runtime(self) -> None:
        if not self.config.updates.enabled:
            return
        if self._auto_updater is None:
            self._auto_updater = self._build_auto_updater()
        self._auto_update_interval_seconds = self._resolve_auto_update_interval()
        self._next_update_check = time.monotonic() + self._auto_update_interval_seconds

    def _build_auto_updater(self) -> Any:
        if self.auto_updater_factory is not None:
            return self.auto_updater_factory()

        from edge_ai_mass.orchestration.config import OrchestratorConfig
        from edge_ai_mass.orchestration.updater import Updater

        config_path = Path(self.config.updates.config_path)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        update_config = OrchestratorConfig.from_file(config_path)
        return Updater(update_config, restart_callback=self._on_update_restart_requested)

    def _resolve_auto_update_interval(self) -> float:
        configured = self.config.updates.poll_interval_seconds
        if configured is not None and configured > 0:
            return float(configured)
        updater_config = getattr(self._auto_updater, "config", None)
        interval = getattr(updater_config, "poll_interval_seconds", 6 * 60 * 60)
        return float(max(interval, 60))

    def _maybe_check_for_updates(self) -> None:
        if not self.config.updates.enabled or self._next_update_check is None:
            return
        now = time.monotonic()
        if now < self._next_update_check:
            return
        self._check_for_updates(reason="periodic")
        self._next_update_check = time.monotonic() + self._auto_update_interval_seconds

    def _check_for_updates(self, *, reason: str) -> None:
        if not self.config.updates.enabled or self._update_check_busy:
            return
        if self._auto_updater is None:
            self._auto_updater = self._build_auto_updater()
        self._update_check_busy = True
        logger.info("Checking for agent updates: reason=%s", reason)
        try:
            result = self._auto_updater.check_for_updates(
                target=self.config.updates.target,
                force=self.config.updates.force,
                release_tag=self.config.updates.release_tag,
            )
        except Exception as exc:
            logger.exception("Automatic update check failed")
            print_panel(
                "Automatic update check",
                {"reason": reason, "error": exc},
                status="failed",
            )
            return
        finally:
            self._update_check_busy = False

        if getattr(result, "changed", False):
            logger.info("Automatic update installed: %s", getattr(result, "message", ""))
            print_panel(
                "Automatic update",
                {"message": getattr(result, "message", "")},
                status="installed",
            )
            self._on_update_restart_requested()
            if self.http is not None or self.mqtt is not None:
                self.send_telemetry(status="online")
        else:
            logger.info("Automatic update check completed: %s", getattr(result, "message", ""))

    def _on_update_restart_requested(self) -> bool:
        self._inference_runner = None
        self._apply_active_model_path_overrides()
        return True

    def _upload_requested_media(self, envelope: Envelope, result: Any) -> dict[str, Any]:
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
        uploaded = 0
        queued = 0
        for job in jobs:
            if self._upload_media_job(job) == "uploaded":
                uploaded += 1
            else:
                queued += 1
        return {
            "requested_count": len(jobs),
            "uploaded_count": uploaded,
            "queued_count": queued,
        }

    def _upload_media_job(self, job: MediaUploadJob) -> str:
        try:
            assert self.http is not None
            self.http.upload_media(
                job.path,
                media_type=job.media_type,
                request_id=job.request_id,
                correlation_id=job.correlation_id,
                metadata=job.metadata,
            )
            return "uploaded"
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
            return "queued"

    def _get_inference_runner(self) -> InferenceRunner:
        if self._inference_runner is None:
            if self.inference_runner_factory is not None:
                self._inference_runner = self.inference_runner_factory()
            else:
                from edge_ai_mass.pipeline.factory import build_pipeline

                self._apply_active_model_path_overrides()
                pipeline = build_pipeline(self.config.runtime.pipeline_config)
                self._attach_host_stage_providers(pipeline)
                print_panel(
                    "Inference runtime",
                    {
                        "pipeline_config": self.config.runtime.pipeline_config,
                        "host_inference": self.config.host_inference.enabled,
                        "active_models": self.active_models,
                    },
                    status="loading",
                )
                pipeline.load_all()
                print_panel(
                    "Inference runtime",
                    {
                        "pipeline_config": self.config.runtime.pipeline_config,
                        "stage_count": len(getattr(pipeline, "stages", {})),
                    },
                    status="loaded",
                )
                self._inference_runner = InferenceRunner(
                    pipeline=pipeline,
                    active_models=self.active_models,
                )

        print_panel(
            "Inference runner",
            {
                "initialized": True,
                "active_models": self.active_models,
            },
            status="ready",
        )
        return self._inference_runner

    def _attach_host_stage_providers(self, pipeline: Any) -> None:
        """Attach host-primary providers that run only after local primary failure."""

        settings = self.config.host_inference
        if not settings.enabled:
            return
        from edge_ai_mass.host.provider import HostStageModule

        for stage_name, stage in pipeline.stages.items():
            if not settings.stage_enabled(stage_name):
                continue
            stage.host_provider = HostStageModule(
                {
                    "stage_name": stage_name,
                    "endpoint_stage": settings.endpoint_stage(stage_name),
                    "base_url": settings.resolved_base_url,
                    "timeout_seconds": settings.request_timeout_seconds,
                    "health_timeout_seconds": settings.health_timeout_seconds,
                    "load_timeout_seconds": settings.load_timeout_seconds,
                }
            )
            logger.info(
                "Attached host inference provider for stage=%s endpoint_stage=%s base_url=%s",
                stage_name,
                settings.endpoint_stage(stage_name),
                settings.resolved_base_url,
            )

    def _preload_inference_runtime(self) -> None:
        if not self.config.runtime.preload_inference:
            logger.warning(
                "Inference preloading is disabled; native model libraries may load on first use"
            )
            return
        if not self.config.device.capabilities.get("inference", False):
            logger.info("Inference preloading skipped because the device capability is disabled")
            return
        if self._inference_runner is not None:
            return

        started = time.perf_counter()
        logger.info(
            "Preloading inference runtime before MQTT startup: pipeline_config=%s",
            self.config.runtime.pipeline_config,
        )
        try:
            self._get_inference_runner()
        except Exception as exc:
            if "static TLS block" in str(exc):
                logger.error(
                    "The platform loader could not reserve static TLS for a model library. "
                    "If eager loading is still too late on this Jetson image, preload the "
                    "Torch-bundled libgomp with LD_PRELOAD before starting Python."
                )
            logger.exception(
                "Inference runtime preload failed; MQTT will not start with an unready pipeline"
            )
            raise
        logger.info(
            "Inference runtime ready before MQTT startup: load_time_ms=%.1f active_models=%s",
            (time.perf_counter() - started) * 1000.0,
            self.active_models,
        )

    def _apply_active_model_path_overrides(
        self,
        activated_components: list[dict[str, Any]] | None = None,
    ) -> None:
        """Expose deployed model artifacts to YAML configs through env overrides."""

        mass_path = None
        for component in activated_components or []:
            if not isinstance(component, dict) or component.get("task") != "mass":
                continue
            active_path = component.get("active_path")
            if active_path:
                mass_path = Path(str(active_path))
                break

        if mass_path is None:
            active_artifact_path = getattr(self.model_manager, "active_artifact_path", None)
            if callable(active_artifact_path):
                mass_path = active_artifact_path("mass")

        if mass_path is not None and Path(mass_path).is_file():
            os.environ["EDGE_AI_MASS_MODEL_PATH"] = str(Path(mass_path))

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
