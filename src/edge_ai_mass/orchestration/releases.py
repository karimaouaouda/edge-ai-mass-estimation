"""Small GitHub Releases client used by the orchestrator updater."""

from __future__ import annotations

import json
import logging
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edge_ai_mass.orchestration.config import GitHubSettings

logger = logging.getLogger(__name__)


class GitHubReleaseError(RuntimeError):
    """Base error for GitHub Release API failures."""


class GitHubReleaseNotFound(GitHubReleaseError):
    """Raised when GitHub cannot find the configured release."""


@dataclass(frozen=True, slots=True)
class ReleaseAsset:
    name: str
    download_url: str
    size: int = 0
    content_type: str = ""


@dataclass(frozen=True, slots=True)
class Release:
    tag_name: str
    name: str
    body: str
    prerelease: bool
    draft: bool
    assets: list[ReleaseAsset]

    def find_asset(self, name: str) -> ReleaseAsset | None:
        for asset in self.assets:
            if asset.name == name:
                return asset
        return None


class GitHubReleaseClient:
    """Fetch release metadata and assets from GitHub."""

    def __init__(self, settings: GitHubSettings) -> None:
        self.settings = settings

    def get_release(self, tag: str | None = None) -> Release:
        release_selector = tag or self.settings.release
        if release_selector == "latest" and self.settings.include_prereleases:
            release = self.get_latest_visible_release()
            logger.info(
                "Fetched release %s with %d assets",
                release.tag_name,
                len(release.assets),
            )
            return release

        if release_selector == "latest":
            path = (
                f"/repos/{self.settings.owner}/{self.settings.repo}/releases/latest"
            )
        else:
            path = (
                f"/repos/{self.settings.owner}/{self.settings.repo}/releases/tags/"
                f"{release_selector}"
            )
        data = self._get_json(
            self.settings.api_base_url.rstrip("/") + path,
            release_selector=release_selector,
        )
        release = _parse_release(data)
        logger.info("Fetched release %s with %d assets", release.tag_name, len(release.assets))
        return release

    def get_latest_visible_release(self) -> Release:
        """Return the newest visible non-draft release, including prereleases."""
        path = f"/repos/{self.settings.owner}/{self.settings.repo}/releases?per_page=20"
        data = self._get_json(
            self.settings.api_base_url.rstrip("/") + path,
            release_selector="latest",
        )
        releases = [_parse_release(item) for item in data]
        for release in releases:
            if not release.draft:
                return release
        raise GitHubReleaseNotFound(
            "No published GitHub release was found for "
            f"{self.settings.owner}/{self.settings.repo}. Draft releases are ignored."
        )

    def download_asset(self, asset: ReleaseAsset, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = self._request(asset.download_url)
        logger.info("Downloading release asset %s", asset.name)
        with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
            with destination.open("wb") as out:
                shutil.copyfileobj(response, out)
        return destination

    def _get_json(self, url: str, *, release_selector: str) -> dict[str, Any]:
        request = self._request(url)
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404:
                raise GitHubReleaseNotFound(
                    "GitHub release was not found for "
                    f"{self.settings.owner}/{self.settings.repo} "
                    f"(release={release_selector!r}). Check that the owner/repo are correct, "
                    "the release is published and not draft-only, and GITHUB_TOKEN is set "
                    "when the repository is private."
                ) from exc
            raise GitHubReleaseError(
                f"GitHub API request failed ({exc.code}) for "
                f"{self.settings.owner}/{self.settings.repo}: {body}"
            ) from exc

    def _request(self, url: str) -> urllib.request.Request:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "edge-ai-mass-orchestrator",
        }
        if self.settings.token:
            headers["Authorization"] = f"Bearer {self.settings.token}"
        return urllib.request.Request(url, headers=headers)


def _parse_release(data: dict[str, Any]) -> Release:
    assets = [
        ReleaseAsset(
            name=str(item.get("name") or ""),
            download_url=str(item.get("browser_download_url") or ""),
            size=int(item.get("size") or 0),
            content_type=str(item.get("content_type") or ""),
        )
        for item in data.get("assets", [])
    ]
    return Release(
        tag_name=str(data.get("tag_name") or data.get("name") or ""),
        name=str(data.get("name") or data.get("tag_name") or ""),
        body=str(data.get("body") or ""),
        prerelease=bool(data.get("prerelease", False)),
        draft=bool(data.get("draft", False)),
        assets=[asset for asset in assets if asset.name and asset.download_url],
    )
