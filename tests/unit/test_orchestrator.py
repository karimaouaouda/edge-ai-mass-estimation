"""Unit tests for the update orchestrator."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from edge_ai_mass.orchestration.config import GitHubSettings, OrchestratorConfig
from edge_ai_mass.orchestration.releases import (
    GitHubReleaseClient,
    GitHubReleaseNotFound,
    Release,
    ReleaseAsset,
)
from edge_ai_mass.orchestration.updater import Updater


class FakeReleaseClient:
    def __init__(self, release: Release, files: dict[str, Path]) -> None:
        self.release = release
        self.files = files

    def get_release(self, tag=None):
        return self.release

    def download_asset(self, asset: ReleaseAsset, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[asset.name].read_bytes())
        return destination


class MissingReleaseClient:
    def get_release(self, tag=None):
        raise GitHubReleaseNotFound("release is missing")


class FakeGitHubReleaseClient(GitHubReleaseClient):
    def __init__(self, settings: GitHubSettings, payload):
        super().__init__(settings)
        self.payload = payload
        self.urls: list[str] = []

    def _get_json(self, url: str, *, release_selector: str):
        self.urls.append(url)
        return self.payload


class FakeKaggleClient:
    def __init__(self, model_file: str, payload: bytes) -> None:
        self.model_file = model_file
        self.payload = payload
        self.calls: list[dict] = []

    def model_instance_version_download(
        self,
        model_instance_version,
        path=None,
        force=False,
        quiet=True,
        untar=False,
    ):
        self.calls.append(
            {
                "model_instance_version": model_instance_version,
                "path": path,
                "force": force,
                "quiet": quiet,
                "untar": untar,
            }
        )
        download_dir = Path(path)
        output = download_dir / self.model_file
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self.payload)
        archive = download_dir / "model.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(output, arcname=self.model_file)
        return str(archive)


def test_orchestrator_config_parses_install_plan(tmp_path):
    cfg = OrchestratorConfig.from_mapping(
        {
            "project_root": str(tmp_path),
            "updates": {
                "mode": "hybrid",
                "install_plan": {
                    "assets": [
                        {
                            "name": "detector",
                            "asset": "detector.engine",
                            "destination": "models/weights/detector.engine",
                        }
                    ]
                },
            },
        },
        base_dir=tmp_path,
    )

    assert cfg.mode == "hybrid"
    assert cfg.install_plan[0].asset == "detector.engine"
    assert cfg.cache_dir == tmp_path / ".edge_ai_mass/orchestrator/cache"


def test_updater_installs_release_asset_atomically(tmp_path):
    asset_file = tmp_path / "detector.engine"
    asset_file.write_bytes(b"new-model")
    checksum = hashlib.sha256(b"new-model").hexdigest()
    manifest = {
        "channel": "stable",
        "assets": [
            {
                "name": "detector",
                "target": "model",
                "asset": "detector.engine",
                "destination": "models/weights/detector.engine",
                "sha256": checksum,
            }
        ],
    }
    manifest_file = tmp_path / "edge-ai-update-manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    release = Release(
        tag_name="v1.0.0",
        name="v1.0.0",
        body="",
        prerelease=False,
        draft=False,
        assets=[
            ReleaseAsset("edge-ai-update-manifest.json", "file://manifest"),
            ReleaseAsset("detector.engine", "file://detector"),
        ],
    )
    cfg = OrchestratorConfig.from_mapping(
        {
            "project_root": str(tmp_path),
            "updates": {"mode": "manual"},
            "runtime": {"restart_app_on_update": False},
        },
        base_dir=tmp_path,
    )
    updater = Updater(
        cfg,
        release_client=FakeReleaseClient(
            release,
            {
                "edge-ai-update-manifest.json": manifest_file,
                "detector.engine": asset_file,
            },
        ),
    )

    result = updater.check_for_updates()

    assert result.changed
    assert (tmp_path / "models/weights/detector.engine").read_bytes() == b"new-model"
    assert json.loads(cfg.state_path.read_text(encoding="utf-8"))["current_release"] == "v1.0.0"


def test_updater_rolls_back_last_file_update(tmp_path):
    destination = tmp_path / "models/weights/detector.engine"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old-model")

    new_asset = tmp_path / "detector.engine"
    new_asset.write_bytes(b"new-model")
    checksum = hashlib.sha256(b"new-model").hexdigest()
    release = Release(
        tag_name="v1.0.0",
        name="v1.0.0",
        body="",
        prerelease=False,
        draft=False,
        assets=[ReleaseAsset("detector.engine", "file://detector")],
    )
    cfg = OrchestratorConfig.from_mapping(
        {
            "project_root": str(tmp_path),
            "updates": {
                "mode": "manual",
                "install_plan": {
                    "assets": [
                        {
                            "name": "detector",
                            "target": "model",
                            "asset": "detector.engine",
                            "destination": "models/weights/detector.engine",
                            "sha256": checksum,
                        }
                    ]
                },
            },
            "runtime": {"restart_app_on_update": False},
        },
        base_dir=tmp_path,
    )
    updater = Updater(
        cfg,
        release_client=FakeReleaseClient(release, {"detector.engine": new_asset}),
    )

    updater.check_for_updates()
    rollback = updater.rollback(target="model")

    assert "Rolled back" in rollback.message
    assert destination.read_bytes() == b"old-model"


def test_updater_returns_message_when_release_is_missing(tmp_path):
    cfg = OrchestratorConfig.from_mapping(
        {
            "project_root": str(tmp_path),
            "updates": {"mode": "manual"},
        },
        base_dir=tmp_path,
    )
    updater = Updater(cfg, release_client=MissingReleaseClient())

    result = updater.check_for_updates()

    assert not result.changed
    assert result.release == "latest"
    assert result.message == "release is missing"


def test_latest_release_can_include_prereleases():
    client = FakeGitHubReleaseClient(
        GitHubSettings(
            owner="owner",
            repo="repo",
            include_prereleases=True,
        ),
        payload=[
            {
                "tag_name": "v1.1.0-dev",
                "name": "v1.1.0-dev",
                "draft": False,
                "prerelease": True,
                "assets": [],
            }
        ],
    )

    release = client.get_release("latest")

    assert release.tag_name == "v1.1.0-dev"
    assert release.prerelease is True
    assert client.urls[0].endswith("/repos/owner/repo/releases?per_page=20")


def test_updater_installs_model_from_kaggle_model_version(tmp_path):
    payload = b"kaggle-model"
    checksum = hashlib.sha256(payload).hexdigest()
    manifest = {
        "channel": "stable",
        "assets": [
            {
                "name": "kaggle-trained-detector",
                "source": "kaggle_model",
                "target": "model",
                "asset": "best.pt",
                "destination": "models/weights/yolo-seg-best.pt",
                "sha256": checksum,
                "kaggle_model": "owner/waste-detector/pyTorch/yolo-seg/3",
                "kaggle_model_file": "weights/best.pt",
            }
        ],
    }
    manifest_file = tmp_path / "edge-ai-update-manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    release = Release(
        tag_name="v1.1.0",
        name="v1.1.0",
        body="",
        prerelease=False,
        draft=False,
        assets=[ReleaseAsset("edge-ai-update-manifest.json", "file://manifest")],
    )
    cfg = OrchestratorConfig.from_mapping(
        {
            "project_root": str(tmp_path),
            "updates": {"mode": "manual"},
            "runtime": {"restart_app_on_update": False},
        },
        base_dir=tmp_path,
    )
    fake_kaggle = FakeKaggleClient("weights/best.pt", payload)
    updater = Updater(
        cfg,
        release_client=FakeReleaseClient(
            release,
            {"edge-ai-update-manifest.json": manifest_file},
        ),
        kaggle_client_factory=lambda: fake_kaggle,
    )

    result = updater.check_for_updates()

    assert result.changed
    assert (tmp_path / "models/weights/yolo-seg-best.pt").read_bytes() == payload
    assert fake_kaggle.calls[0]["model_instance_version"] == (
        "owner/waste-detector/pyTorch/yolo-seg/3"
    )
    assert fake_kaggle.calls[0]["untar"] is False
