"""Manifest-driven updater for models, configs, engines, and app packages."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import tarfile
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from edge_ai_mass.orchestration.config import InstallRule, OrchestratorConfig
from edge_ai_mass.orchestration.releases import (
    GitHubReleaseClient,
    GitHubReleaseNotFound,
    Release,
    ReleaseAsset,
)

logger = logging.getLogger(__name__)


class UpdaterError(RuntimeError):
    """Raised when an update cannot be applied safely."""


@dataclass(slots=True)
class InstalledFile:
    name: str
    target: str
    destination: str
    backup: str = ""
    restart_app: bool = True


@dataclass(slots=True)
class UpdateResult:
    checked: bool = True
    release: str = ""
    installed: list[InstalledFile] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    restarted: bool = False
    message: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.installed)


class Updater:
    """Apply release assets according to an update manifest or local install plan."""

    def __init__(
        self,
        config: OrchestratorConfig,
        release_client: GitHubReleaseClient | None = None,
        restart_callback: Callable[[], None] | None = None,
        kaggle_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.release_client = release_client or GitHubReleaseClient(config.github)
        self.restart_callback = restart_callback
        self.kaggle_client_factory = kaggle_client_factory
        self.state = UpdateState(config.state_path)

    def check_for_updates(
        self,
        *,
        target: str | None = None,
        force: bool = False,
        release_tag: str | None = None,
    ) -> UpdateResult:
        """Fetch the configured GitHub release and apply matching assets."""

        try:
            release = self.release_client.get_release(release_tag)
        except GitHubReleaseNotFound as exc:
            self.state.record_check(release_tag or self.config.github.release, changed=False)
            self.state.save()
            return UpdateResult(
                checked=True,
                release=release_tag or self.config.github.release,
                message=str(exc),
            )

        if release.draft:
            return UpdateResult(release=release.tag_name, message="Release is a draft; skipping")
        if release.prerelease and not self.config.github.include_prereleases:
            return UpdateResult(
                release=release.tag_name,
                message="Release is a prerelease and prereleases are disabled; skipping",
            )

        manifest = self._download_manifest(release)
        if manifest and not self._manifest_matches_channel(manifest):
            channel = manifest.get("channel", "")
            return UpdateResult(
                release=release.tag_name,
                message=f"Manifest channel {channel!r} does not match configured channel",
            )

        rules = self._rules_for_release(manifest)
        rules = [rule for rule in rules if _rule_matches_target(rule, target)]
        if not rules:
            return UpdateResult(
                release=release.tag_name,
                message=f"No install rules matched target {target or 'all'}",
            )

        result = UpdateResult(release=release.tag_name)
        backups_for_rollback: list[InstalledFile] = []

        try:
            for rule in rules:
                if self._asset_is_current(rule, release, force):
                    result.skipped.append(rule.name)
                    continue

                downloaded = self._download_rule_source(rule, release)
                if downloaded is None:
                    result.skipped.append(rule.name)
                    continue

                expected_sha = rule.sha256 or _manifest_sha_for_asset(manifest, rule.asset)
                if expected_sha:
                    _verify_sha256(downloaded, expected_sha)

                installed = self._install_rule(rule, downloaded, release)
                result.installed.extend(installed)
                backups_for_rollback.extend([item for item in installed if item.backup])
                self.state.record_asset(rule, release, expected_sha or _sha256(downloaded))

            if not result.changed:
                result.message = "No updates were installed"
                self.state.record_check(release.tag_name, changed=False)
                self.state.save()
                return result

            self._run_health_check()
        except Exception:
            logger.exception("Update failed; rolling back file assets from this attempt")
            self._rollback_files(backups_for_rollback)
            raise

        self.state.record_update(release.tag_name, result.installed)
        self.state.save()

        if self._should_restart(result.installed):
            result.restarted = self._restart_application()

        result.message = f"Installed {len(result.installed)} asset(s) from {release.tag_name}"
        return result

    def rollback(self, *, target: str | None = None) -> UpdateResult:
        """Restore the most recent backed-up files for the selected target."""

        last_update = self.state.data.get("last_update", {})
        backups = [
            InstalledFile(**item)
            for item in last_update.get("installed", [])
            if item.get("backup")
            and (target in {None, "all", item.get("target"), item.get("name")})
        ]
        if not backups:
            return UpdateResult(message=f"No backups found for target {target or 'all'}")

        self._rollback_files(backups)
        self.state.record_rollback(target or "all", backups)
        self.state.save()
        return UpdateResult(
            installed=backups,
            message=f"Rolled back {len(backups)} asset(s) for target {target or 'all'}",
        )

    def _download_manifest(self, release: Release) -> dict[str, Any]:
        manifest_asset = release.find_asset(self.config.github.manifest_asset)
        if manifest_asset is None:
            logger.info(
                "Release %s has no %s asset; using local install plan",
                release.tag_name,
                self.config.github.manifest_asset,
            )
            return {}

        manifest_path = self._download_asset(release, manifest_asset)
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise UpdaterError(f"Invalid update manifest JSON: {manifest_path}") from exc

    def _manifest_matches_channel(self, manifest: dict[str, Any]) -> bool:
        channel = str(manifest.get("channel") or self.config.github.channel)
        return channel == self.config.github.channel

    def _rules_for_release(self, manifest: dict[str, Any]) -> list[InstallRule]:
        manifest_rules = _manifest_install_rules(manifest)
        if manifest_rules:
            return manifest_rules
        return list(self.config.install_plan)

    def _download_asset(self, release: Release, asset: ReleaseAsset) -> Path:
        destination = self.config.cache_dir / release.tag_name / asset.name
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        return self.release_client.download_asset(asset, destination)

    def _download_rule_source(self, rule: InstallRule, release: Release) -> Path | None:
        source = rule.source.replace("-", "_")
        if source in {"github", "github_release", "release"}:
            release_asset = release.find_asset(rule.asset)
            if release_asset is None:
                if not rule.required:
                    return None
                raise UpdaterError(
                    f"Release {release.tag_name} does not contain asset {rule.asset!r}"
                )
            return self._download_asset(release, release_asset)

        if source in {"kaggle", "kaggle_model", "kaggle_models"}:
            return self._download_kaggle_model_version(rule, release)

        raise UpdaterError(f"Install rule {rule.name!r} has unsupported source {rule.source!r}")

    def _download_kaggle_model_version(
        self, rule: InstallRule, release: Release
    ) -> Path | None:
        model_version = _kaggle_model_version_ref(rule)
        if not model_version or not rule.kaggle_model_file:
            if not rule.required:
                return None
            raise UpdaterError(
                f"Kaggle model rule {rule.name!r} needs kaggle_model and kaggle_model_file"
            )

        destination = self.config.cache_dir / release.tag_name / rule.asset
        if destination.exists() and destination.stat().st_size > 0:
            return destination

        download_dir = self.config.cache_dir / release.tag_name / "kaggle-model" / rule.name
        if download_dir.exists():
            shutil.rmtree(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        api = self._kaggle_client()
        archive_path = Path(
            api.model_instance_version_download(
                model_version,
                path=str(download_dir),
                force=True,
                quiet=True,
                untar=False,
            )
        )

        extract_dir = download_dir / "extracted"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        with tarfile.open(archive_path) as archive:
            _safe_extract_tar(archive, extract_dir)

        source = _locate_downloaded_model_file(extract_dir, rule.kaggle_model_file, rule.asset)
        if source is None:
            source = _locate_downloaded_model_file(
                download_dir,
                rule.kaggle_model_file,
                rule.asset,
            )

        if source is None:
            if not rule.required:
                return None
            raise UpdaterError(
                "Kaggle model file was not found after download: "
                f"model={model_version!r}, file={rule.kaggle_model_file!r}"
            )

        _atomic_copy(source, destination)
        logger.info(
            "Downloaded Kaggle model file %s from %s to %s",
            rule.kaggle_model_file,
            model_version,
            destination,
        )
        return destination

    def _kaggle_client(self) -> Any:
        if self.kaggle_client_factory is not None:
            return self.kaggle_client_factory()
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:
            raise UpdaterError(
                "Kaggle model updates require the 'kaggle' package. "
                "Install it and configure KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json."
            ) from exc

        api = KaggleApi()
        api.authenticate()
        return api

    def _asset_is_current(self, rule: InstallRule, release: Release, force: bool) -> bool:
        if force:
            return False
        installed = self.state.data.get("installed_assets", {}).get(rule.name)
        if not installed:
            return False
        return installed.get("release") == release.tag_name

    def _install_rule(
        self, rule: InstallRule, downloaded: Path, release: Release
    ) -> list[InstalledFile]:
        if rule.install_command:
            self._run_install_command(rule, downloaded, release)
            return [
                InstalledFile(
                    name=rule.name,
                    target=rule.target,
                    destination=" ".join(rule.install_command),
                    restart_app=rule.restart_app,
                )
            ]

        if not rule.destination:
            raise UpdaterError(
                f"Install rule {rule.name!r} needs a destination or install_command"
            )

        destination = _resolve_destination(self.config.project_root, rule.destination)
        if rule.unpack in {"zip", "tar", "tgz", "tar.gz"}:
            return self._install_archive(rule, downloaded, destination)

        backup = _backup_existing(destination, self.config.backup_dir, self.config.project_root)
        _atomic_copy(downloaded, destination)
        if rule.executable:
            _make_executable(destination)
        logger.info("Installed %s to %s", rule.name, destination)
        return [
            InstalledFile(
                name=rule.name,
                target=rule.target,
                destination=str(destination),
                backup=str(backup) if backup else "",
                restart_app=rule.restart_app,
            )
        ]

    def _install_archive(
        self, rule: InstallRule, downloaded: Path, destination: Path
    ) -> list[InstalledFile]:
        staging = self.config.cache_dir / "staging" / rule.name
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        if rule.unpack == "zip":
            with zipfile.ZipFile(downloaded) as archive:
                _safe_extract_zip(archive, staging)
        else:
            with tarfile.open(downloaded) as archive:
                _safe_extract_tar(archive, staging)

        backup = _backup_existing(destination, self.config.backup_dir, self.config.project_root)
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(destination))
        logger.info("Installed archive %s to %s", rule.name, destination)
        return [
            InstalledFile(
                name=rule.name,
                target=rule.target,
                destination=str(destination),
                backup=str(backup) if backup else "",
                restart_app=rule.restart_app,
            )
        ]

    def _run_install_command(
        self, rule: InstallRule, downloaded: Path, release: Release
    ) -> None:
        command = [
            part.format(
                asset_path=str(downloaded),
                project_root=str(self.config.project_root),
                release=release.tag_name,
                asset=rule.asset,
            )
            for part in rule.install_command
        ]
        _run_command(command, self.config)

    def _run_health_check(self) -> None:
        if self.config.runtime.health_check_command:
            _run_command(self.config.runtime.health_check_command, self.config)

    def _should_restart(self, installed: list[InstalledFile]) -> bool:
        if not self.config.runtime.restart_app_on_update:
            return False
        restart_targets = {"program", "app", "config", "model", "engine"}
        return any(item.restart_app and item.target in restart_targets for item in installed)

    def _restart_application(self) -> bool:
        if self.restart_callback is not None:
            return bool(self.restart_callback())
        if self.config.runtime.restart_command:
            _run_command(self.config.runtime.restart_command, self.config)
            return True
        logger.info("Update installed, but no restart mechanism is configured")
        return False

    def _rollback_files(self, files: list[InstalledFile]) -> None:
        for item in reversed(files):
            if not item.backup:
                continue
            backup = Path(item.backup)
            destination = Path(item.destination)
            if backup.exists():
                _atomic_copy(backup, destination)
                logger.warning("Rolled back %s from %s", destination, backup)


class UpdateState:
    """JSON state file used to avoid reinstalling the same assets."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = self._load()

    def record_check(self, release: str, *, changed: bool) -> None:
        self.data["last_check"] = _now()
        self.data["last_release"] = release
        self.data["last_check_changed"] = changed

    def record_asset(self, rule: InstallRule, release: Release, sha256: str) -> None:
        assets = self.data.setdefault("installed_assets", {})
        assets[rule.name] = {
            "asset": rule.asset,
            "source": rule.source,
            "target": rule.target,
            "release": release.tag_name,
            "sha256": sha256,
            "kaggle_model": rule.kaggle_model,
            "kaggle_model_version": rule.kaggle_model_version,
            "kaggle_model_file": rule.kaggle_model_file,
            "installed_at": _now(),
        }

    def record_update(self, release: str, installed: list[InstalledFile]) -> None:
        self.record_check(release, changed=True)
        self.data["current_release"] = release
        self.data["last_update"] = {
            "release": release,
            "installed_at": _now(),
            "installed": [asdict(item) for item in installed],
        }

    def record_rollback(self, target: str, files: list[InstalledFile]) -> None:
        self.data["last_rollback"] = {
            "target": target,
            "rolled_back_at": _now(),
            "files": [asdict(item) for item in files],
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"package_version": _installed_package_version()}
        return json.loads(self.path.read_text(encoding="utf-8"))


def _manifest_install_rules(manifest: dict[str, Any]) -> list[InstallRule]:
    if not manifest:
        return []
    assets = manifest.get("assets")
    if isinstance(assets, list):
        return [InstallRule.from_mapping(item) for item in assets]

    targets = manifest.get("targets", {})
    rules: list[InstallRule] = []
    if isinstance(targets, dict):
        for target, value in targets.items():
            items = value if isinstance(value, list) else [value]
            for item in items:
                if not isinstance(item, dict):
                    continue
                item = {**item, "target": item.get("target", target)}
                rules.append(InstallRule.from_mapping(item))
    return rules


def _manifest_sha_for_asset(manifest: dict[str, Any], asset_name: str) -> str:
    for rule in _manifest_install_rules(manifest):
        if rule.asset == asset_name:
            return rule.sha256
    return ""


def _kaggle_model_version_ref(rule: InstallRule) -> str:
    model = rule.kaggle_model.strip().strip("/")
    version = rule.kaggle_model_version.strip().strip("/")
    if not model:
        return ""
    if len(model.split("/")) == 5:
        return model
    if version:
        return f"{model}/{version}"
    return ""


def _locate_downloaded_model_file(
    download_dir: Path,
    model_file: str,
    asset_name: str,
) -> Path | None:
    normalized = model_file.replace("\\", "/").strip("/")
    expected = download_dir / Path(normalized)
    if expected.is_file():
        return expected

    basename = Path(normalized or asset_name).name
    matches = [path for path in download_dir.rglob(basename) if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    return None


def _rule_matches_target(rule: InstallRule, target: str | None) -> bool:
    if target in {None, "", "all"}:
        return True
    return target in {rule.target, rule.name, rule.asset}


def _resolve_destination(project_root: Path, destination: str) -> Path:
    path = Path(destination)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, tmp)
    os.replace(tmp, destination)


def _backup_existing(destination: Path, backup_dir: Path, project_root: Path) -> Path | None:
    if not destination.exists():
        return None
    try:
        relative = destination.relative_to(project_root)
    except ValueError:
        relative = Path(destination.name)
    backup = backup_dir / _now_for_path() / relative
    backup.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_dir():
        shutil.copytree(destination, backup)
    else:
        shutil.copy2(destination, backup)
    return backup


def _verify_sha256(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise UpdaterError(
            f"Checksum mismatch for {path.name}: expected {expected}, got {actual}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.namelist():
        target = (destination / member).resolve()
        if root != target and root not in target.parents:
            raise UpdaterError(f"Unsafe path in zip archive: {member}")
    archive.extractall(destination)


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root != target and root not in target.parents:
            raise UpdaterError(f"Unsafe path in tar archive: {member.name}")
    archive.extractall(destination)


def _run_command(command: list[str], config: OrchestratorConfig) -> None:
    logger.info("Running command: %s", " ".join(command))
    result = subprocess.run(
        command,
        cwd=config.project_root,
        text=True,
        capture_output=True,
        timeout=config.runtime.command_timeout_seconds,
        check=False,
    )
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.warning(result.stderr.strip())
    if result.returncode != 0:
        raise UpdaterError(f"Command failed with exit code {result.returncode}: {command}")


def _installed_package_version() -> str:
    try:
        return version("edge-ai-mass")
    except PackageNotFoundError:
        return "0.0.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_for_path() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
