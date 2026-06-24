"""Package and publish reusable training outputs without redistributing data."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from edge_ai_mass.training.config import TrainingConfig
from edge_ai_mass.training.state import PipelineState, git_metadata


ALLOWED_ARTIFACT_ROOTS = {
    "checkpoints",
    "evaluation",
    "exports",
    "models",
    "optimization",
    "reports",
    "training",
    "zenml",
}
EXCLUDED_DIRECTORY_NAMES = {
    "__pycache__",
    "annotated_batches",
    "annotated_samples",
    "dataset_visualizations",
    "images",
    "labels",
    "mlruns",
    "training_runs",
    "tuning_runs",
}
EXCLUDED_FILE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".webp",
}
ALLOWED_CHART_SUFFIXES = {".png", ".svg", ".html"}


class TrainingOutputPublisher:
    """Create/version a Kaggle dataset containing only reusable model outputs."""

    def __init__(self, config: TrainingConfig, state: PipelineState):
        self.config = config
        self.state = state
        self.settings = config.payload.get("publication", {})
        self.bundle_dir = config.path(
            self.settings.get(
                "bundle_dir",
                f"builds/kaggle/training-outputs/{config.run_name}",
            )
        )

    def publish(self) -> dict[str, Any]:
        """Build the governed bundle and create or version its Kaggle dataset."""
        if not self.settings or not self.settings.get("enabled", True):
            return {"enabled": False, "reason": "publication is not configured or disabled"}
        provider = str(self.settings.get("provider", "kaggle")).lower()
        if provider != "kaggle":
            raise ValueError(f"Unsupported output publication provider: {provider}")

        bundle = self.prepare_bundle()
        api = _authenticate_kaggle()
        reference = str(self.settings["dataset"])
        notes_prefix = str(
            self.settings.get(
                "version_notes",
                f"Training outputs for {self.config.run_name}",
            )
        )
        notes = (
            f"{notes_prefix} | run={self.config.run_name} | "
            f"dataset={self.state.data.get('dataset_fingerprint', 'unknown')}"
        )
        if _dataset_exists(api, reference):
            response = api.dataset_create_version(
                str(self.bundle_dir),
                version_notes=notes,
                quiet=False,
                convert_to_csv=False,
                delete_old_versions=False,
                dir_mode="skip",
            )
            operation = "versioned"
        else:
            response = api.dataset_create_new(
                str(self.bundle_dir),
                public=bool(self.settings.get("public", False)),
                quiet=False,
                convert_to_csv=False,
                dir_mode="skip",
            )
            operation = "created"
        _wait_for_dataset(
            api,
            reference,
            timeout_seconds=int(self.settings.get("wait_timeout_seconds", 300)),
        )
        remote = _owned_dataset(api, reference)
        result = {
            **bundle,
            "provider": "kaggle",
            "dataset": reference,
            "operation": operation,
            "url": f"https://www.kaggle.com/datasets/{reference}",
            "version": getattr(remote, "current_version_number", None),
            "response": str(response),
            "status": "ready",
        }
        report = self.config.artifacts_dir / "reports" / "publication.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.state.update("publish", publication=result)
        return result

    def prepare_bundle(self) -> dict[str, Any]:
        """Build a deterministic ZIP and manifest with strict data exclusions."""
        _reset_bundle_directory(self.bundle_dir)
        selected = collect_reusable_outputs(self.config)
        if self.settings.get("require_checkpoint", True) and not any(
            relative.parts and relative.parts[0] in {"checkpoints", "models"}
            for _, relative in selected
        ):
            raise FileNotFoundError(
                "No reusable checkpoints/models were found. Run the train stage before publish."
            )

        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project": self.config.payload["project"].get("name"),
            "run_name": self.config.run_name,
            "config_digest": self.config.digest,
            "data_config_digest": self.config.data_digest,
            "dataset_fingerprint": self.state.data.get("dataset_fingerprint"),
            "git": git_metadata(self.config.project_root),
            "exclusions": {
                "raw_data": True,
                "processed_dataset": True,
                "dataset_images": True,
                "annotated_samples": True,
                "scratch_runs": True,
                "mlflow_database": True,
            },
            "files": {
                relative.as_posix(): {
                    "sha256": _sha256(source),
                    "size_bytes": source.stat().st_size,
                }
                for source, relative in selected
            },
        }
        generated = {
            "metadata/training-output-manifest.json": (
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            ).encode(),
            "metadata/resolved-config.yaml": yaml.safe_dump(
                self.config.resolved_copy(),
                sort_keys=False,
            ).encode(),
            "metadata/pipeline-state.json": (
                json.dumps(self.state.data, indent=2, sort_keys=True) + "\n"
            ).encode(),
        }
        dataset_manifest = self.config.dataset_dir / "dataset_manifest.json"
        if dataset_manifest.is_file():
            generated["metadata/dataset-manifest.json"] = dataset_manifest.read_bytes()

        archive = self.bundle_dir / "training_outputs.zip"
        _deterministic_zip(archive, selected, generated)
        manifest_path = self.bundle_dir / "training-output-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        owner, slug = str(self.settings["dataset"]).split("/", 1)
        metadata = {
            "title": str(
                self.settings.get(
                    "title",
                    f"Edge AI Mass Training Outputs - {self.config.run_name}",
                )
            ),
            "id": f"{owner}/{slug}",
            "licenses": [{"name": str(self.settings.get("license", "MIT"))}],
        }
        (self.bundle_dir / "dataset-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "enabled": True,
            "bundle_dir": str(self.bundle_dir),
            "archive": str(archive),
            "archive_sha256": _sha256(archive),
            "archive_size_bytes": archive.stat().st_size,
            "file_count": len(selected),
            "manifest": str(manifest_path),
        }


def collect_reusable_outputs(config: TrainingConfig) -> list[tuple[Path, Path]]:
    """Return allowed artifact files as ``(source, archive-relative path)``."""
    root = config.artifacts_dir
    selected: list[tuple[Path, Path]] = []
    if not root.is_dir():
        return selected
    for top_level in sorted(ALLOWED_ARTIFACT_ROOTS):
        source_root = root / top_level
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
                continue
            suffix = path.suffix.lower()
            if suffix in EXCLUDED_FILE_SUFFIXES:
                continue
            # PNG is retained only for plots/diagnostics, never qualitative data
            # samples. The excluded directory rules above enforce that boundary.
            if suffix in ALLOWED_CHART_SUFFIXES or suffix not in {".png", ".svg", ".html"}:
                selected.append((path, relative))
    state = root / "pipeline_state.json"
    if state.is_file():
        selected.append((state, Path("pipeline_state.json")))
    return selected


def restore_training_outputs(
    archive: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Safely restore a published output bundle into a run artifact directory."""
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    restored: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = Path(member.filename)
            if not relative.parts or relative.parts[0] == "metadata":
                continue
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe path in training-output archive: {member.filename}")
            target = (destination / relative).resolve()
            if destination.resolve() not in target.parents:
                raise ValueError(f"Archive path escapes destination: {member.filename}")
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            restored.append(relative.as_posix())
    _sanitize_restored_state(destination)
    return {"archive": str(archive), "destination": str(destination), "restored": restored}


def _sanitize_restored_state(destination: Path) -> None:
    """Remove machine-local tracking IDs and rewrite important artifact paths."""
    state_path = destination / "pipeline_state.json"
    if not state_path.is_file():
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("mlflow_run_id", None)
    path_updates = {
        "best_weights": destination / "models" / "best.pt",
        "last_weights": destination / "models" / "last.pt",
    }
    latest_path = destination / "checkpoints" / "latest.json"
    if latest_path.is_file():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        path_updates["latest_checkpoint"] = (
            destination / "checkpoints" / str(latest.get("weights", ""))
        )
        path_updates["latest_checkpoint_manifest"] = (
            destination / "checkpoints" / str(latest.get("manifest", ""))
        )
        state["checkpoint_history"] = [
            {
                "completed_epochs": json.loads(manifest.read_text(encoding="utf-8")).get(
                    "completed_epochs"
                ),
                "weights": str((manifest.parent / "weights.pt").resolve()),
                "manifest": str(manifest.resolve()),
            }
            for manifest in sorted(
                (destination / "checkpoints").glob(
                    "epoch_*/checkpoint_manifest.json"
                )
            )
        ]
    for key, path in path_updates.items():
        if path.is_file():
            state[key] = str(path.resolve())
        else:
            state.pop(key, None)
    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _reset_bundle_directory(path: Path) -> None:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor) or len(resolved.parts) < 3:
        raise ValueError(f"Refusing unsafe publication directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _deterministic_zip(
    archive_path: Path,
    files: list[tuple[Path, Path]],
    generated: dict[str, bytes],
) -> None:
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source, relative in files:
            _write_zip_member(archive, relative.as_posix(), source.read_bytes())
        for relative, payload in sorted(generated.items()):
            _write_zip_member(archive, relative, payload)


def _write_zip_member(archive: zipfile.ZipFile, relative: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload)


def _authenticate_kaggle() -> Any:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'Kaggle publishing requires: pip install -e ".[orchestration]"'
        ) from exc
    api = KaggleApi()
    api.authenticate()
    return api


def _owned_dataset(api: Any, reference: str) -> Any | None:
    slug = reference.split("/", 1)[-1]
    datasets = api.dataset_list(mine=True, search=slug) or []
    return next((item for item in datasets if item and item.ref == reference), None)


def _dataset_exists(api: Any, reference: str) -> bool:
    try:
        api.dataset_status(reference)
        return True
    except Exception as exc:
        message = str(exc).lower()
        if "404" in message or "not found" in message:
            return False
        if any(value in message for value in ("401", "403", "unauthorized", "forbidden")):
            return _owned_dataset(api, reference) is not None
        raise


def _wait_for_dataset(api: Any, reference: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            status = str(api.dataset_status(reference)).lower()
            print(f"Training-output dataset status: {status}")
            if "ready" in status:
                return
            if any(value in status for value in ("error", "failed")):
                raise RuntimeError(f"Kaggle dataset processing failed: {status}")
        except Exception as exc:
            message = str(exc).lower()
            if not any(value in message for value in ("401", "403", "unauthorized", "forbidden")):
                raise
            owned = _owned_dataset(api, reference)
            if owned is not None and bool(getattr(owned, "_is_frozen", False)):
                return
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for Kaggle dataset: {reference}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "TrainingOutputPublisher",
    "collect_reusable_outputs",
    "restore_training_outputs",
]
