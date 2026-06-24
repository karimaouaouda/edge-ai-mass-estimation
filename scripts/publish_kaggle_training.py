"""Prepare and publish the Kaggle module dataset and training kernel.

Examples:
    python scripts/publish_kaggle_training.py prepare --force
    python scripts/publish_kaggle_training.py publish --message "Update training pipeline"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "builds" / "kaggle" / "edge-ai-mass-module"
DEFAULT_MODULE_DATASET = "karimaouaouda/edge-ai-mass-module"
DEFAULT_KERNEL = "karimaouaouda/train-pipeline"
ARCHIVE_NAME = "edge_ai_mass_module.zip"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "runs"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pt", ".onnx", ".engine", ".db"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def safe_reset_directory(path: Path, *, force: bool) -> None:
    resolved = path.resolve()
    builds_root = (PROJECT_ROOT / "builds" / "kaggle").resolve()
    if builds_root not in resolved.parents:
        raise ValueError(f"Bundle directory must stay under {builds_root}: {resolved}")
    if resolved.exists():
        if not force:
            raise FileExistsError(f"Bundle already exists: {resolved}. Pass --force to replace it.")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def include_file(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    return not (
        any(part in EXCLUDED_PARTS for part in relative.parts)
        or path.suffix.lower() in EXCLUDED_SUFFIXES
    )


def collect_module_files() -> tuple[list[Path], list[str]]:
    candidates = [
        *sorted((PROJECT_ROOT / "src" / "edge_ai_mass").rglob("*")),
        *sorted((PROJECT_ROOT / "configs" / "training").rglob("*")),
        *sorted((PROJECT_ROOT / "scripts").glob("*.py")),
        *sorted((PROJECT_ROOT / "scripts").glob("*.sh")),
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "kaggle_training_guide.md",
        PROJECT_ROOT / "docs" / "training_pipeline.md",
    ]
    included: list[Path] = []
    skipped: list[str] = []
    for path in candidates:
        if not path.is_file() or not include_file(path):
            continue
        if path.suffix == ".py":
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except SyntaxError as exc:
                skipped.append(f"{path.relative_to(PROJECT_ROOT).as_posix()}: {exc.msg}")
                continue
        included.append(path)
    return sorted(set(included)), skipped


def deterministic_zip(
    archive_path: Path,
    files: list[Path],
    generated_files: dict[str, bytes],
) -> None:
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in files:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            data = path.read_bytes()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
        for relative, data in sorted(generated_files.items()):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)


def build_wheel(bundle_dir: Path) -> Path:
    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        str(PROJECT_ROOT),
        "--wheel-dir",
        str(bundle_dir),
        "--no-deps",
        "--no-build-isolation",
        "--disable-pip-version-check",
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to build the importable Kaggle wheel.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    wheels = sorted(bundle_dir.glob("edge_ai_mass-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one edge-ai-mass wheel, found: {wheels}")
    return wheels[0]


def validate_training_notebook() -> dict[str, Any]:
    notebook_path = PROJECT_ROOT / "notebooks" / "yolo_training_pipeline.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = 0
    for index, cell in enumerate(payload.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        compile("".join(cell.get("source", [])), f"{notebook_path}:cell-{index}", "exec")
        code_cells += 1
    return {
        "path": str(notebook_path),
        "cells": len(payload.get("cells", [])),
        "code_cells": code_cells,
    }


def validate_kernel_metadata(module_dataset: str, kernel: str) -> dict[str, Any]:
    metadata_path = PROJECT_ROOT / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("id") != kernel:
        raise ValueError(f"kernel-metadata.json id must be {kernel!r}")
    if metadata.get("code_file") != "notebooks/yolo_training_pipeline.ipynb":
        raise ValueError("kernel metadata must target notebooks/yolo_training_pipeline.ipynb")
    if module_dataset not in metadata.get("dataset_sources", []):
        raise ValueError(f"kernel metadata must attach module dataset {module_dataset!r}")
    taco_sources = [
        source for source in metadata.get("dataset_sources", []) if "taco" in source.lower()
    ]
    if taco_sources:
        raise ValueError(
            "TACO must be downloaded from its Hugging Face annotations JSON, not attached "
            f"as a Kaggle dataset: {taco_sources}"
        )
    if not metadata.get("enable_gpu"):
        raise ValueError("Training kernel must enable GPU")
    if not metadata.get("enable_internet"):
        raise ValueError("Training kernel must enable internet for TACO acquisition")
    return metadata


def prepare_bundle(
    bundle_dir: Path,
    *,
    module_dataset: str,
    kernel: str,
    force: bool,
) -> dict[str, Any]:
    safe_reset_directory(bundle_dir, force=force)
    notebook = validate_training_notebook()
    metadata = validate_kernel_metadata(module_dataset, kernel)
    files, skipped = collect_module_files()
    wheel_path = build_wheel(bundle_dir)
    manifest = {
        "schema_version": 1,
        "module_dataset": module_dataset,
        "kernel": kernel,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "notebook": notebook,
        "kernel_sources": metadata.get("kernel_sources", []),
        "dataset_sources": metadata.get("dataset_sources", []),
        "wheel": {
            "name": wheel_path.name,
            "sha256": sha256_bytes(wheel_path.read_bytes()),
        },
        "files": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_bytes(path.read_bytes())
            for path in files
        },
        "skipped_invalid_scripts": skipped,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    archive_path = bundle_dir / ARCHIVE_NAME
    deterministic_zip(
        archive_path,
        files,
        {"module-manifest.json": manifest_bytes},
    )
    (bundle_dir / "module-manifest.json").write_bytes(manifest_bytes)
    owner, slug = module_dataset.split("/", 1)
    dataset_metadata = {
        "title": "Edge AI Mass Training Module",
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "MIT"}],
    }
    (bundle_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "bundle_dir": str(bundle_dir),
        "archive": str(archive_path),
        "archive_sha256": sha256_bytes(archive_path.read_bytes()),
        "archive_size_bytes": archive_path.stat().st_size,
        "wheel": str(wheel_path),
        "wheel_sha256": sha256_bytes(wheel_path.read_bytes()),
        "files": len(files),
        "skipped_invalid_scripts": skipped,
        "notebook": notebook,
    }
    print(json.dumps(summary, indent=2))
    return summary


def authenticate_kaggle():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit("Install Kaggle support with: pip install -e '.[orchestration]'") from exc
    api = KaggleApi()
    api.authenticate()
    return api


def find_owned_dataset(api: Any, reference: str) -> Any | None:
    """Find a private dataset through the endpoint Kaggle exposes to its owner."""
    slug = reference.split("/", 1)[-1]
    datasets = api.dataset_list(mine=True, search=slug) or []
    return next((item for item in datasets if item and item.ref == reference), None)


def is_private_status_error(exc: Exception) -> bool:
    """Identify Kaggle's current private-dataset status authorization failure."""
    message = str(exc).lower()
    return "401" in message or "403" in message or "unauthorized" in message


def dataset_exists(api: Any, reference: str) -> bool:
    try:
        api.dataset_status(reference)
        return True
    except Exception as exc:
        # Kaggle API 2.2 can return 401/403 from GetDatasetStatus for a private
        # dataset even when the same credentials can list and version it.
        if is_private_status_error(exc):
            return find_owned_dataset(api, reference) is not None
        if "404" in str(exc) or "not found" in str(exc).lower():
            return False
        raise


def wait_for_dataset(api: Any, reference: str, timeout_seconds: int = 300) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            status = str(api.dataset_status(reference)).lower()
            print(f"Dataset status: {status}")
            if "ready" in status:
                return
            if any(value in status for value in ("error", "failed")):
                raise RuntimeError(f"Kaggle dataset processing failed: {status}")
        except Exception as exc:
            if not is_private_status_error(exc):
                raise
            owned = find_owned_dataset(api, reference)
            if owned is not None:
                version = getattr(owned, "current_version_number", None)
                # Kaggle SDK 0.1.30 stores this response field privately and
                # does not expose the usual generated public property.
                frozen = bool(
                    getattr(owned, "is_frozen", getattr(owned, "_is_frozen", False))
                )
                print(f"Owned dataset fallback: version={version} ready={frozen}")
                if frozen:
                    return
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for Kaggle dataset: {reference}")


def publish(
    bundle_dir: Path,
    *,
    module_dataset: str,
    kernel: str,
    message: str,
    skip_module: bool,
    skip_kernel: bool,
) -> None:
    prepare_bundle(
        bundle_dir,
        module_dataset=module_dataset,
        kernel=kernel,
        force=True,
    )
    api = authenticate_kaggle()
    if not skip_module:
        if dataset_exists(api, module_dataset):
            response = api.dataset_create_version(
                str(bundle_dir),
                version_notes=message,
                quiet=False,
                convert_to_csv=False,
                delete_old_versions=False,
                dir_mode="skip",
            )
        else:
            response = api.dataset_create_new(
                str(bundle_dir),
                public=False,
                quiet=False,
                convert_to_csv=False,
                dir_mode="skip",
            )
        print(f"Module dataset response: {response}")
        wait_for_dataset(api, module_dataset)
    if not skip_kernel:
        response = api.kernels_push(str(PROJECT_ROOT))
        print(f"Kernel response: {response}")
        print(f"Kaggle kernel: https://www.kaggle.com/code/{kernel}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build and validate the module archive")
    prepare.add_argument("--force", action="store_true", help="Replace the generated bundle")

    upload = subparsers.add_parser(
        "publish",
        help="Prepare, version the module dataset, and push the training kernel",
    )
    upload.add_argument("--message", default="Update governed YOLO training pipeline")
    upload.add_argument("--skip-module", action="store_true")
    upload.add_argument("--skip-kernel", action="store_true")

    for subparser in (prepare, upload):
        subparser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
        subparser.add_argument("--module-dataset", default=DEFAULT_MODULE_DATASET)
        subparser.add_argument("--kernel", default=DEFAULT_KERNEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_bundle(
            args.bundle_dir,
            module_dataset=args.module_dataset,
            kernel=args.kernel,
            force=args.force,
        )
        return
    publish(
        args.bundle_dir,
        module_dataset=args.module_dataset,
        kernel=args.kernel,
        message=args.message,
        skip_module=args.skip_module,
        skip_kernel=args.skip_kernel,
    )


if __name__ == "__main__":
    main()
