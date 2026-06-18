"""Telemetry sampler for Jetson/backend state updates."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from edge_ai_mass.agent.envelopes import utc_now_iso


BoolCallback = Callable[[], bool]
ModelsCallback = Callable[[], dict[str, str]]


class TelemetrySampler:
    """Collect backend telemetry payloads with optional psutil support."""

    def __init__(
        self,
        *,
        camera_connected: BoolCallback | None = None,
        inference_busy: BoolCallback | None = None,
        active_models: ModelsCallback | None = None,
        disk_path: str | Path = ".",
        metrics_probe: Callable[[], dict[str, float | None]] | None = None,
    ) -> None:
        self.camera_connected = camera_connected or (lambda: False)
        self.inference_busy = inference_busy or (lambda: False)
        self.active_models = active_models or (lambda: {})
        self.disk_path = Path(disk_path)
        self.metrics_probe = metrics_probe or self._probe_metrics

    def sample(self, *, status: str = "online") -> dict[str, Any]:
        metrics = self.metrics_probe()
        payload: dict[str, Any] = {
            "status": status,
            "health_status": _health_status(metrics),
            "camera_connected": bool(self.camera_connected()),
            "inference_busy": bool(self.inference_busy()),
            "active_models": self.active_models(),
            "reported_at": utc_now_iso(),
        }
        for key in (
            "cpu_percent",
            "memory_percent",
            "temperature_celsius",
            "disk_percent",
        ):
            value = metrics.get(key)
            if value is not None:
                payload[key] = float(value)
        return payload

    def _probe_metrics(self) -> dict[str, float | None]:
        cpu_percent = None
        memory_percent = None
        try:
            import psutil
        except ImportError:
            psutil = None

        if psutil is not None:
            cpu_percent = float(psutil.cpu_percent(interval=None))
            memory_percent = float(psutil.virtual_memory().percent)

        disk_percent = _disk_percent(self.disk_path)
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "temperature_celsius": _temperature_celsius(psutil),
            "disk_percent": disk_percent,
        }


def _health_status(metrics: dict[str, float | None]) -> str:
    cpu = metrics.get("cpu_percent")
    memory = metrics.get("memory_percent")
    disk = metrics.get("disk_percent")
    temp = metrics.get("temperature_celsius")
    if any(value is not None and value >= 98.0 for value in (cpu, memory, disk)):
        return "critical"
    if temp is not None and temp >= 95.0:
        return "critical"
    if any(value is not None and value >= 90.0 for value in (cpu, memory, disk)):
        return "warning"
    if temp is not None and temp >= 80.0:
        return "warning"
    return "healthy"


def _disk_percent(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    if usage.total <= 0:
        return None
    return usage.used / usage.total * 100.0


def _temperature_celsius(psutil_module: Any) -> float | None:
    if psutil_module is not None and hasattr(psutil_module, "sensors_temperatures"):
        try:
            temps = psutil_module.sensors_temperatures()
        except Exception:
            temps = {}
        for entries in temps.values():
            for entry in entries:
                current = getattr(entry, "current", None)
                if current is not None:
                    return float(current)

    for path in (
        Path("/sys/class/thermal/thermal_zone0/temp"),
        Path("/sys/devices/virtual/thermal/thermal_zone0/temp"),
    ):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw:
            continue
        value = float(raw)
        return value / 1000.0 if value > 1000 else value
    return None
