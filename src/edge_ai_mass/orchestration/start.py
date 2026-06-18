"""Long-running Jetson orchestrator process."""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from edge_ai_mass.orchestration.broker import BrokerCommand, MQTTUpdateListener
from edge_ai_mass.orchestration.config import OrchestratorConfig
from edge_ai_mass.orchestration.updater import Updater
from edge_ai_mass.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class ApplicationSupervisor:
    """Optional child-process supervisor for the inference application."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if not self.config.runtime.manage_app_process or not self.config.runtime.app_command:
            return
        if self.process is not None and self.process.poll() is None:
            return
        logger.info("Starting app process: %s", " ".join(self.config.runtime.app_command))
        self.process = subprocess.Popen(
            self.config.runtime.app_command,
            cwd=self.config.project_root,
            text=True,
        )

    def restart(self) -> bool:
        if self.config.runtime.restart_command:
            subprocess.run(
                self.config.runtime.restart_command,
                cwd=self.config.project_root,
                timeout=self.config.runtime.command_timeout_seconds,
                check=True,
            )
            return True
        if not self.config.runtime.manage_app_process or not self.config.runtime.app_command:
            logger.info("Restart requested, but no app process or restart_command is configured")
            return False
        self.stop()
        self.start()
        return True

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        logger.info("Stopping app process")
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            logger.warning("App process did not stop cleanly; killing it")
            self.process.kill()
            self.process.wait(timeout=10)

    def maintain(self) -> None:
        if not self.config.runtime.manage_app_process or not self.config.runtime.restart_on_exit:
            return
        if self.process is not None and self.process.poll() is not None:
            logger.warning("App process exited with code %s; restarting", self.process.returncode)
            self.process = None
            self.start()


class Orchestrator:
    """Coordinates update checks, MQTT triggers, and optional app supervision."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.commands: queue.Queue[BrokerCommand] = queue.Queue()
        self.stop_requested = False
        self.supervisor = ApplicationSupervisor(config)
        self.updater = Updater(config, restart_callback=self.supervisor.restart)
        self.mqtt: MQTTUpdateListener | None = None

    @classmethod
    def from_config_file(cls, path: str | Path) -> "Orchestrator":
        return cls(OrchestratorConfig.from_file(path))

    def run_once(
        self,
        *,
        target: str | None = None,
        force: bool = False,
        release_tag: str | None = None,
    ):
        return self.updater.check_for_updates(
            target=target,
            force=force,
            release_tag=release_tag,
        )

    def rollback(self, *, target: str | None = None):
        return self.updater.rollback(target=target)

    def run_forever(self) -> None:
        self._install_signal_handlers()
        self.supervisor.start()
        self._start_mqtt_if_enabled()

        next_poll = (
            0.0
            if self.config.run_on_start
            else time.monotonic() + self.config.poll_interval_seconds
        )
        logger.info(
            "Orchestrator started in %s mode for device %s",
            self.config.mode,
            self.config.device_id,
        )

        while not self.stop_requested:
            now = time.monotonic()
            if self.config.mode in {"poll", "hybrid"} and now >= next_poll:
                self._handle_update_command(BrokerCommand(action="check"))
                next_poll = time.monotonic() + self.config.poll_interval_seconds

            self._drain_commands()
            self.supervisor.maintain()

            wait_seconds = 5.0
            if self.config.mode in {"poll", "hybrid"}:
                wait_seconds = max(1.0, min(wait_seconds, next_poll - time.monotonic()))
            time.sleep(wait_seconds)

        self._shutdown()

    def _start_mqtt_if_enabled(self) -> None:
        if self.config.mode not in {"mqtt", "hybrid"} or not self.config.mqtt.enabled:
            return
        self.mqtt = MQTTUpdateListener(
            self.config.mqtt,
            self.config.device_id,
            on_command=self.commands.put,
        )
        try:
            self.mqtt.start()
        except RuntimeError:
            if self.config.mode == "mqtt":
                raise
            logger.exception("MQTT listener failed; continuing with periodic polling")
            self.mqtt = None

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return
            self._handle_update_command(command)

    def _handle_update_command(self, command: BrokerCommand) -> None:
        try:
            if command.action in {"check", "update", "upgrade"}:
                result = self.updater.check_for_updates(
                    target=command.target,
                    force=command.force,
                    release_tag=command.release_tag,
                )
            elif command.action == "rollback":
                result = self.updater.rollback(target=command.target)
            elif command.action in {"restart", "restart_app"}:
                self.supervisor.restart()
                result = {"message": "Application restart requested"}
            else:
                result = {"message": f"Unknown action {command.action!r}"}

            logger.info("Update command result: %s", result)
            self._publish_status("ok", result)
        except Exception as exc:
            logger.exception("Update command failed")
            self._publish_status("error", {"message": str(exc), "command": asdict(command)})

    def _publish_status(self, status: str, payload) -> None:
        if self.mqtt is None:
            return
        if is_dataclass(payload):
            payload = asdict(payload)
        self.mqtt.publish_status({"status": status, "payload": payload})

    def _install_signal_handlers(self) -> None:
        def _request_stop(_signum, _frame) -> None:
            self.stop_requested = True

        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)

    def _shutdown(self) -> None:
        logger.info("Stopping orchestrator")
        if self.mqtt is not None:
            self.mqtt.stop()
        self.supervisor.stop()


# Backwards-compatible name for the placeholder class that used to live here.
Orchestration = Orchestrator


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="edge-ai-orchestrator",
        description="Run the Edge AI Mass Estimation Jetson orchestrator",
    )
    parser.add_argument("--config", default="configs/orchestration/jetson_nano.yaml")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--once", action="store_true", help="Check/apply updates once and exit")
    parser.add_argument(
        "--target",
        default=None,
        help="Target to update: all, model, config, program",
    )
    parser.add_argument("--force", action="store_true", help="Reinstall even if state says current")
    parser.add_argument("--release-tag", default=None, help="Specific GitHub release tag to use")
    parser.add_argument("--rollback", action="store_true", help="Rollback the last file update")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    orchestrator = Orchestrator.from_config_file(args.config)

    if args.rollback:
        result = orchestrator.rollback(target=args.target)
        print(result.message)
        return

    if args.once:
        try:
            result = orchestrator.run_once(
                target=args.target,
                force=args.force,
                release_tag=args.release_tag,
            )
        except Exception as exc:
            if args.log_level.upper() == "DEBUG":
                logger.exception("One-shot update failed")
            print(f"Update failed: {exc}", file=sys.stderr)
            sys.exit(2)
        print(result.message)
        return

    orchestrator.run_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
