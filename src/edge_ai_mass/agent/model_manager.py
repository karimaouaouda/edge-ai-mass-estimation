"""Model deployment command support for local artifact activation."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class ModelDeploymentError(RuntimeError):
    """Raised when a deployment command cannot be applied safely."""


@dataclass(frozen=True, slots=True)
class ActivatedComponent:
    task: str
    version: str
    source: str
    active_path: str

    def to_payload(self) -> dict[str, str]:
        return {
            "task": self.task,
            "version": self.version,
            "source": self.source,
            "active_path": self.active_path,
        }


class ModelManager:
    """Stage, verify, and activate deployment components."""

    def __init__(self, model_dir: str | Path, *, active_models: dict[str, str] | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.active_models = active_models if active_models is not None else {}

    def deploy(self, payload: dict[str, Any], *, device_id: str) -> dict[str, Any]:
        target = str(payload.get("target_device_id") or "")
        if target and target != device_id:
            raise ModelDeploymentError(
                f"Deployment target {target!r} does not match local device {device_id!r}"
            )

        components = payload.get("components")
        if not isinstance(components, list) or not components:
            raise ModelDeploymentError("Deployment payload requires at least one component")

        activated: list[ActivatedComponent] = []
        for item in components:
            if not isinstance(item, dict):
                raise ModelDeploymentError("Deployment component must be an object")
            activated_component = self._activate_component(item)
            activated.append(activated_component)
            self.active_models[activated_component.task] = activated_component.version

        return {
            "deployment_id": payload.get("deployment_id"),
            "request_id": payload.get("request_id"),
            "activated_components": [item.to_payload() for item in activated],
            "active_models": dict(self.active_models),
            "restart_agent": bool(payload.get("restart_agent", False)),
        }

    def active_artifact_path(self, task: str, version: str | None = None) -> Path | None:
        """Return the active artifact file for a deployed task/version if present."""

        active_version = str(version or self.active_models.get(task) or "").strip()
        if not active_version:
            return None
        directory = self.model_dir / task / active_version
        if not directory.is_dir():
            return None
        candidates = sorted(
            path for path in directory.iterdir() if path.is_file() and not path.name.startswith(".")
        )
        if not candidates:
            return None
        preferred_extensions = _preferred_extensions(task)
        for extension in preferred_extensions:
            for candidate in candidates:
                if candidate.suffix.lower() == extension:
                    return candidate
        return candidates[0]

    def _activate_component(self, component: dict[str, Any]) -> ActivatedComponent:
        task = str(component.get("task") or "").strip()
        version = str(component.get("version") or "").strip()
        if task not in {"detector", "depth", "material", "mass"}:
            raise ModelDeploymentError(f"Unsupported component task: {task!r}")
        if not version:
            raise ModelDeploymentError(f"Component {task!r} is missing version")

        source_path = self._resolve_source(component)
        checksum = _component_checksum(component)
        if checksum:
            _verify_sha256(source_path, checksum)

        destination = self.model_dir / task / version / source_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_copy(source_path, destination)
        return ActivatedComponent(
            task=task,
            version=version,
            source=str(source_path),
            active_path=str(destination),
        )

    def _resolve_source(self, component: dict[str, Any]) -> Path:
        source_type = str(
            component.get("source_type")
            or (component.get("artifact") or {}).get("source_type")
            or ""
        ).lower()
        source_uri = str(component.get("source_uri") or "")
        artifact = component.get("artifact") if isinstance(component.get("artifact"), dict) else {}
        candidate = str(artifact.get("path") or artifact.get("local_path") or source_uri or "")

        if source_uri.startswith("file://"):
            candidate = urlparse(source_uri).path

        if source_type in {"local", "manual", ""} or source_uri.startswith("file://"):
            path = Path(candidate)
            if not path.exists():
                raise ModelDeploymentError(f"Local model artifact not found: {candidate}")
            if not path.is_file():
                raise ModelDeploymentError(f"Local model artifact is not a file: {candidate}")
            return path

        raise ModelDeploymentError(
            "Remote artifact source is not directly resolvable by this agent yet: "
            f"{source_type or source_uri}"
        )


def _component_checksum(component: dict[str, Any]) -> str:
    artifact = component.get("artifact") if isinstance(component.get("artifact"), dict) else {}
    value = str(component.get("checksum") or artifact.get("checksum") or "")
    if value.startswith("sha256:"):
        return value.split(":", 1)[1]
    return value


def _verify_sha256(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise ModelDeploymentError(
            f"Checksum mismatch for {path.name}: expected {expected}, got {actual}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, tmp)
    os.replace(tmp, destination)


def _preferred_extensions(task: str) -> tuple[str, ...]:
    if task == "mass":
        return (".joblib", ".pkl", ".pickle", ".onnx", ".pt")
    if task == "detector":
        return (".engine", ".pt", ".onnx")
    if task == "depth":
        return (".engine", ".pt", ".onnx")
    return (".joblib", ".pkl", ".pt", ".onnx")
