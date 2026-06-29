"""Periodic YOLO checkpoint storage and model-source resolution."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edge_ai_mass.training.config import TrainingConfig
from edge_ai_mass.training.state import PipelineState


@dataclass(frozen=True)
class ModelSource:
    """Resolved source used to construct YOLO for a training invocation."""

    path: str
    kind: str
    resume: bool
    completed_epochs: int = 0
    manifest: str | None = None
    selected_epoch: int | None = None
    rollback: dict[str, Any] | None = None


def resolve_model_source(
    config: TrainingConfig,
    *,
    artifacts_dir: Path | None = None,
    prefer_resume: bool = True,
    rollback_to_selected: bool = False,
) -> ModelSource:
    """Resolve an explicit/automatic checkpoint or the configured base model."""
    checkpointing = config.payload["training"].get("checkpointing", {})
    resume_cfg = checkpointing.get("resume", {})
    mode = str(resume_cfg.get("mode", "auto")).strip().lower()
    root = (artifacts_dir or config.artifacts_dir) / "checkpoints"

    if prefer_resume and checkpointing.get("enabled", True) and mode != "never":
        explicit = resume_cfg.get("checkpoint")
        if explicit:
            path = config.path(str(explicit))
            if not path.is_file():
                raise FileNotFoundError(f"Configured resume checkpoint does not exist: {path}")
            return ModelSource(
                path=str(path),
                kind="explicit_checkpoint",
                resume=True,
                completed_epochs=_checkpoint_epoch_from_name(path),
            )

        selected_epoch = _selected_resume_epoch(resume_cfg)
        if selected_epoch is not None:
            selected = _managed_checkpoint_payload(root, selected_epoch)
            if selected is not None:
                rollback = None
                if rollback_to_selected and bool(
                    resume_cfg.get("prune_after_selected", True)
                ):
                    rollback = _rollback_managed_checkpoints(root, selected_epoch)
                    selected = _managed_checkpoint_payload(root, selected_epoch)
                    if selected is None:  # pragma: no cover - defensive filesystem guard
                        raise FileNotFoundError(
                            f"Selected checkpoint disappeared during rollback: "
                            f"epoch_{selected_epoch:06d}"
                        )
                return ModelSource(
                    path=str(selected["weights_path"].resolve()),
                    kind="selected_checkpoint",
                    resume=True,
                    completed_epochs=int(selected["latest"].get("completed_epochs", 0)),
                    manifest=str(selected["manifest_path"].resolve()),
                    selected_epoch=selected_epoch,
                    rollback=rollback,
                )
            if mode == "required":
                raise FileNotFoundError(
                    f"Configured resume epoch {selected_epoch} was not found under {root}"
                )
            if _managed_checkpoint_dirs(root) or (root / "latest.json").is_file():
                raise FileNotFoundError(
                    f"Configured resume epoch {selected_epoch} was not found under "
                    f"{root}; refusing to resume a different checkpoint"
                )

        latest = root / "latest.json"
        if latest.is_file():
            payload = json.loads(latest.read_text(encoding="utf-8"))
            relative = payload.get("weights")
            path = root / str(relative)
            if path.is_file():
                return ModelSource(
                    path=str(path.resolve()),
                    kind="automatic_checkpoint",
                    resume=True,
                    completed_epochs=int(payload.get("completed_epochs", 0)),
                    manifest=str(latest.resolve()),
                )

        legacy_last = (artifacts_dir or config.artifacts_dir) / "models" / "last.pt"
        if legacy_last.is_file():
            return ModelSource(
                path=str(legacy_last.resolve()),
                kind="legacy_last_checkpoint",
                resume=True,
            )

        if mode == "required":
            raise FileNotFoundError(
                f"Checkpoint resume is required, but no checkpoint was found under {root}"
            )

    value = str(config.payload["model"]["checkpoint"])
    candidate = config.path(value)
    path = str(candidate) if candidate.exists() else value
    return ModelSource(path=path, kind="base_model", resume=False)


def load_model_or_checkpoint(
    config: TrainingConfig,
    *,
    artifacts_dir: Path | None = None,
    prefer_resume: bool = True,
) -> tuple[Any, ModelSource]:
    """Load YOLO from either a base model name/path or a resumable checkpoint."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Ultralytics is required for YOLO training. Install the project dependencies."
        ) from exc

    source = resolve_model_source(
        config,
        artifacts_dir=artifacts_dir,
        prefer_resume=prefer_resume,
    )
    model = YOLO(source.path, task=config.payload["model"].get("task", "segment"))
    return model, source


def resume_target_epochs(config: TrainingConfig, source: ModelSource) -> int:
    """Return the total epoch target for fresh or resumed training."""
    configured_target = int(config.payload["training"].get("epochs", 100))
    if not source.resume:
        return configured_target
    resume_cfg = (
        config.payload["training"].get("checkpointing", {}).get("resume", {})
    )
    additional = int(resume_cfg.get("additional_epochs", 0))
    if additional > 0 and source.completed_epochs > 0:
        return source.completed_epochs + additional
    if source.completed_epochs > 0 and configured_target <= source.completed_epochs:
        raise RuntimeError(
            f"Checkpoint already contains {source.completed_epochs} completed epochs, "
            f"but training.epochs={configured_target}. Increase training.epochs or set "
            "training.checkpointing.resume.additional_epochs to continue in chunks."
        )
    return configured_target


class CheckpointStore:
    """Atomically persist resumable weights and metric history during training."""

    def __init__(
        self,
        config: TrainingConfig,
        state: PipelineState,
        *,
        dataset_fingerprint: str,
        model_source: ModelSource,
    ):
        self.config = config
        self.state = state
        self.settings = config.payload["training"].get("checkpointing", {})
        self.root = config.artifacts_dir / "checkpoints"
        self.dataset_fingerprint = dataset_fingerprint
        self.model_source = model_source

    @property
    def enabled(self) -> bool:
        return bool(self.settings.get("enabled", True))

    @property
    def interval(self) -> int:
        return int(self.settings.get("interval_epochs", 10))

    def callback(self, trainer: Any) -> None:
        """Ultralytics ``on_model_save`` callback."""
        if not self.enabled:
            return
        completed_epochs = int(getattr(trainer, "epoch", -1)) + 1
        total_epochs = int(getattr(trainer, "epochs", completed_epochs))
        is_final = completed_epochs >= total_epochs or bool(getattr(trainer, "stop", False))
        if completed_epochs % self.interval != 0 and not (
            is_final and self.settings.get("save_final", True)
        ):
            return
        self.capture(trainer, completed_epochs=completed_epochs, is_final=is_final)

    def capture(
        self,
        trainer: Any,
        *,
        completed_epochs: int,
        is_final: bool,
    ) -> dict[str, Any]:
        """Capture one checkpoint directory and update its durable latest pointer."""
        source_last = Path(getattr(trainer, "last", ""))
        if not source_last.is_file():
            raise FileNotFoundError(
                f"Ultralytics callback ran without a resumable last checkpoint: {source_last}"
            )

        self.root.mkdir(parents=True, exist_ok=True)
        directory_name = f"epoch_{completed_epochs:06d}"
        destination = self.root / directory_name
        temporary = self.root / f".{directory_name}.building"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)

        weights = temporary / "weights.pt"
        shutil.copy2(source_last, weights)
        source_best = Path(getattr(trainer, "best", ""))
        best_weights = None
        if source_best.is_file():
            best_weights = temporary / "best.pt"
            shutil.copy2(source_best, best_weights)

        results_csv = _trainer_results_csv(trainer)
        history_csv = None
        history = {"columns": [], "rows": []}
        if results_csv and results_csv.is_file():
            history_csv = temporary / "results.csv"
            shutil.copy2(results_csv, history_csv)
            history = _read_metric_history(history_csv)
        history_json = temporary / "metrics_history.json"
        history_json.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        curves = _render_metric_curves(history, temporary / "metrics_curves.png")

        current_metrics = _json_safe(
            {
                "metrics": getattr(trainer, "metrics", {}),
                "fitness": getattr(trainer, "fitness", None),
                "best_fitness": getattr(trainer, "best_fitness", None),
                "learning_rates": getattr(trainer, "lr", {}),
                "losses": _trainer_losses(trainer),
            }
        )
        metrics_path = temporary / "metrics.json"
        metrics_path.write_text(
            json.dumps(current_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "epoch": completed_epochs - 1,
            "completed_epochs": completed_epochs,
            "target_epochs": int(getattr(trainer, "epochs", completed_epochs)),
            "is_final": is_final,
            "weights": "weights.pt",
            "weights_sha256": _sha256(weights),
            "best_weights": "best.pt" if best_weights else None,
            "best_weights_sha256": _sha256(best_weights) if best_weights else None,
            "results_csv": "results.csv" if history_csv else None,
            "metrics_history": history_json.name,
            "metrics_curves": curves.name if curves else None,
            "metrics": current_metrics,
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_digest": self.config.digest,
            "source": asdict(self.model_source),
            "ultralytics_run_dir": str(getattr(trainer, "save_dir", "")),
        }
        (temporary / "checkpoint_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)

        latest = {
            **manifest,
            "checkpoint_dir": directory_name,
            "weights": f"{directory_name}/weights.pt",
            "best_weights": (
                f"{directory_name}/best.pt" if manifest["best_weights"] else None
            ),
            "manifest": f"{directory_name}/checkpoint_manifest.json",
        }
        _write_json_atomic(self.root / "latest.json", latest)
        self._apply_retention()
        history_entries = self._checkpoint_history()
        self.state.patch(
            latest_checkpoint=str((self.root / latest["weights"]).resolve()),
            latest_checkpoint_manifest=str((self.root / latest["manifest"]).resolve()),
            checkpoint_history=history_entries,
        )
        return latest

    def _checkpoint_history(self) -> list[dict[str, Any]]:
        return checkpoint_history(self.root)

    def _apply_retention(self) -> None:
        keep_last = int(self.settings.get("keep_last", 10))
        if keep_last <= 0:
            return
        directories = sorted(path for path in self.root.glob("epoch_*") if path.is_dir())
        for path in directories[:-keep_last]:
            shutil.rmtree(path)


def _trainer_results_csv(trainer: Any) -> Path | None:
    value = getattr(trainer, "csv", None)
    if value:
        return Path(value)
    save_dir = Path(getattr(trainer, "save_dir", ""))
    candidate = save_dir / "results.csv"
    return candidate if candidate.is_file() else None


def _trainer_losses(trainer: Any) -> Any:
    values = getattr(trainer, "tloss", None)
    if values is None:
        return {}
    try:
        return trainer.label_loss_items(values)
    except (AttributeError, TypeError, ValueError):
        return values


def _read_metric_history(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {str(key).strip(): _parse_number(value) for key, value in row.items()}
            for row in reader
        ]
    return {"columns": list(rows[0]) if rows else [], "rows": rows}


def _render_metric_curves(history: dict[str, Any], destination: Path) -> Path | None:
    rows = history.get("rows", [])
    if not rows:
        return None
    columns = [
        column
        for column in history.get("columns", [])
        if column not in {"epoch", "time"} and any(_is_number(row.get(column)) for row in rows)
    ]
    if not columns:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None

    column_count = 3
    row_count = math.ceil(len(columns) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(column_count * 5, row_count * 3.2),
        squeeze=False,
    )
    x_values = [row.get("epoch", index) for index, row in enumerate(rows)]
    for axis, column in zip(axes.flat, columns):
        y_values = [row.get(column) for row in rows]
        axis.plot(x_values, y_values, linewidth=1.5)
        axis.set_title(column, fontsize=9)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    for axis in list(axes.flat)[len(columns) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def _parse_number(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _selected_resume_epoch(resume_cfg: dict[str, Any]) -> int | None:
    """Return the configured managed checkpoint epoch, if one was selected."""
    raw_value = resume_cfg.get("selected_epoch", resume_cfg.get("epoch"))
    if raw_value in (None, ""):
        return None
    epoch = int(raw_value)
    if epoch <= 0:
        raise ValueError("training.checkpointing.resume.selected_epoch must be positive")
    return epoch


def _managed_checkpoint_payload(root: Path, completed_epochs: int) -> dict[str, Any] | None:
    """Read one managed checkpoint directory as a latest.json-compatible payload."""
    checkpoint_dir = root / f"epoch_{completed_epochs:06d}"
    weights = checkpoint_dir / "weights.pt"
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    if not weights.is_file() or not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    latest = _latest_payload_from_manifest(root, checkpoint_dir, manifest)
    return {
        "checkpoint_dir": checkpoint_dir,
        "weights_path": weights,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "latest": latest,
    }


def _latest_payload_from_manifest(
    root: Path,
    checkpoint_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Convert a checkpoint manifest into the durable latest.json pointer."""
    directory_name = checkpoint_dir.name
    best_weights = checkpoint_dir / "best.pt"
    latest = {
        **manifest,
        "checkpoint_dir": directory_name,
        "weights": f"{directory_name}/weights.pt",
        "best_weights": f"{directory_name}/best.pt" if best_weights.is_file() else None,
        "manifest": f"{directory_name}/checkpoint_manifest.json",
    }
    latest["resolved_weights"] = str((root / latest["weights"]).resolve())
    latest["resolved_manifest"] = str((root / latest["manifest"]).resolve())
    return latest


def _rollback_managed_checkpoints(
    root: Path,
    selected_epoch: int,
) -> dict[str, Any]:
    """Delete managed checkpoints newer than selected_epoch and rewrite latest.json."""
    selected = _managed_checkpoint_payload(root, selected_epoch)
    if selected is None:
        raise FileNotFoundError(
            f"Cannot rollback because checkpoint epoch_{selected_epoch:06d} is missing"
        )
    deleted = []
    root_resolved = root.resolve()
    for checkpoint_dir in _managed_checkpoint_dirs(root):
        completed = _checkpoint_epoch_from_name(checkpoint_dir)
        if completed <= selected_epoch:
            continue
        resolved = checkpoint_dir.resolve()
        if root_resolved not in resolved.parents:
            raise RuntimeError(f"Refusing to delete checkpoint outside {root}: {resolved}")
        shutil.rmtree(resolved)
        deleted.append(checkpoint_dir.name)

    latest = selected["latest"]
    latest["rollback"] = {
        "selected_epoch": selected_epoch,
        "deleted_checkpoints": deleted,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(root / "latest.json", latest)
    return {
        "selected_epoch": selected_epoch,
        "latest": str((root / "latest.json").resolve()),
        "deleted_checkpoints": deleted,
        "checkpoint_history": checkpoint_history(root),
    }


def _managed_checkpoint_dirs(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.glob("epoch_*") if path.is_dir()),
        key=_checkpoint_epoch_from_name,
    )


def checkpoint_history(root: Path) -> list[dict[str, Any]]:
    """Return the durable managed checkpoint history under a checkpoint root."""
    result = []
    for manifest_path in sorted(root.glob("epoch_*/checkpoint_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        result.append(
            {
                "completed_epochs": payload.get("completed_epochs"),
                "weights": str((manifest_path.parent / "weights.pt").resolve()),
                "manifest": str(manifest_path.resolve()),
            }
        )
    return result


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    return str(value)


def _checkpoint_epoch_from_name(path: Path) -> int:
    label = path.parent.name if path.parent.name.startswith("epoch_") else path.stem
    digits = "".join(character for character in label if character.isdigit())
    return int(digits) if digits else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "CheckpointStore",
    "ModelSource",
    "checkpoint_history",
    "load_model_or_checkpoint",
    "resume_target_epochs",
    "resolve_model_source",
]
