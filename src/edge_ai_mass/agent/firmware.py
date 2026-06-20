"""Fail-closed firmware update command validation and execution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from edge_ai_mass.agent.envelopes import utc_now_iso


_VERSION_PATTERN = re.compile(
    r"^v?\d+\.\d+\.\d+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?"
    r"(?:\+[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?$"
)


class FirmwareUpdateError(RuntimeError):
    """Raised when a firmware update cannot proceed securely."""

    def __init__(self, error_type: str, summary: str, *, retryable: bool = False) -> None:
        super().__init__(summary)
        self.error_type = error_type
        self.summary = summary
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class FirmwareUpdateRequest:
    target: str
    version: str | None
    current_version: str
    requested_at: str | None


@dataclass(frozen=True, slots=True)
class PreparedFirmwareUpdate:
    """An inert update package that has not yet been installed."""

    version: str
    signature_verified: bool
    compatible: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class FirmwareUpdateExecutor(Protocol):
    """Secure, platform-specific firmware installer implemented by deployment code."""

    def prepare(
        self,
        *,
        version: str | None,
        device_type: str,
        current_version: str,
    ) -> PreparedFirmwareUpdate:
        """Download and verify a package without installing it."""

    def install(self, prepared: PreparedFirmwareUpdate) -> dict[str, Any]:
        """Install a package previously returned by ``prepare``."""


FirmwareProgressCallback = Callable[[dict[str, Any]], None]


class FirmwareUpdateManager:
    """Validate backend firmware commands and enforce verify-before-install."""

    def __init__(
        self,
        *,
        device_type: str,
        current_version: str,
        executor: FirmwareUpdateExecutor | None = None,
    ) -> None:
        self.device_type = device_type
        self.current_version = current_version
        self.executor = executor

    def apply(
        self,
        payload: dict[str, Any],
        *,
        progress: FirmwareProgressCallback,
    ) -> dict[str, Any]:
        request = parse_firmware_update_request(
            payload,
            local_current_version=self.current_version,
        )
        progress(_progress_payload(request, "validating", 10))

        if self.executor is None:
            raise FirmwareUpdateError(
                "firmware_updater_unavailable",
                "No signature-verifying firmware executor is configured on this device.",
            )

        progress(_progress_payload(request, "preparing", 30))
        prepared = self.executor.prepare(
            version=request.version,
            device_type=self.device_type,
            current_version=self.current_version,
        )
        if not prepared.signature_verified:
            raise FirmwareUpdateError(
                "firmware_signature_invalid",
                "Firmware package signature verification failed.",
            )
        if not prepared.compatible:
            raise FirmwareUpdateError(
                "firmware_incompatible",
                f"Firmware {prepared.version!r} is not compatible with {self.device_type!r}.",
            )

        progress(
            _progress_payload(
                request,
                "verified",
                60,
                version=prepared.version,
                metadata=prepared.metadata,
            )
        )
        progress(_progress_payload(request, "installing", 75, version=prepared.version))
        result = self.executor.install(prepared)
        progress(_progress_payload(request, "completed", 100, version=prepared.version))
        return {
            "status": "completed",
            "target": request.target,
            "version": prepared.version,
            "previous_version": self.current_version,
            "completed_at": utc_now_iso(),
            "result": result,
        }


def parse_firmware_update_request(
    payload: dict[str, Any],
    *,
    local_current_version: str,
) -> FirmwareUpdateRequest:
    """Normalize only selectors; versions are never treated as commands or URLs."""

    target = str(payload.get("target") or "").strip().lower()
    if target not in {"latest", "specific"}:
        raise FirmwareUpdateError(
            "invalid_firmware_request",
            "Firmware target must be 'latest' or 'specific'.",
        )

    raw_version = payload.get("version")
    version = str(raw_version).strip() if raw_version is not None else None
    if target == "specific":
        if not version or not _VERSION_PATTERN.fullmatch(version):
            raise FirmwareUpdateError(
                "invalid_firmware_version",
                "Specific firmware versions must be semantic version tags such as v1.8.0.",
            )
    else:
        version = None

    return FirmwareUpdateRequest(
        target=target,
        version=version,
        current_version=local_current_version,
        requested_at=(
            str(payload["requested_at"])
            if payload.get("requested_at") is not None
            else None
        ),
    )


def _progress_payload(
    request: FirmwareUpdateRequest,
    status: str,
    progress_percent: int,
    *,
    version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "progress_percent": progress_percent,
        "target": request.target,
        "version": version if version is not None else request.version,
        "current_version": request.current_version,
        "reported_at": utc_now_iso(),
        "metadata": dict(metadata or {}),
    }
