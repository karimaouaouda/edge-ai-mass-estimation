"""Secret loading and redaction helpers for the edge device agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


class MissingSecretError(RuntimeError):
    """Raised when a required runtime secret is not available."""


class SecretStore(Protocol):
    """Minimal interface for device secret stores."""

    def get(self, name: str) -> str:
        """Return a secret value, or an empty string when missing."""


@dataclass(slots=True)
class EnvironmentSecretStore:
    """Load secrets from process environment variables."""

    environ: dict[str, str] | None = None

    def get(self, name: str) -> str:
        source = self.environ if self.environ is not None else os.environ
        return str(source.get(name, "")).strip()


def require_secret(store: SecretStore, name: str, *, label: str | None = None) -> str:
    """Load a required secret and raise a clear startup error when absent."""

    value = store.get(name)
    if not value:
        display = label or name
        raise MissingSecretError(f"Missing required secret: {display} ({name})")
    return value


def redact(value: str, *, visible: int = 4) -> str:
    """Return a log-safe representation of a secret-like value."""

    if not value:
        return ""
    if len(value) <= visible:
        return "***"
    return f"{value[:visible]}***"


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Redact sensitive headers before logging or error reporting."""

    secret_headers = {"authorization", "x-device-token"}
    return {
        key: "***" if key.lower() in secret_headers else value
        for key, value in headers.items()
    }
