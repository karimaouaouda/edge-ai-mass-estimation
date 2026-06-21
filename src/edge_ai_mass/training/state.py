"""Small durable state contract shared by independently invoked training stages."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class PipelineState:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "completed_stages": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def update(self, stage: str, **values: Any) -> None:
        self.data.update(values)
        completed = list(self.data.get("completed_stages", []))
        if stage not in completed:
            completed.append(stage)
        self.data["completed_stages"] = completed
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(self.path)

    def reset(self, **values: Any) -> None:
        self.data = {"schema_version": 1, "completed_stages": [], **values}
        self.save()

    def require(self, *keys: str) -> None:
        missing = [key for key in keys if not self.data.get(key)]
        if missing:
            raise RuntimeError(
                f"Pipeline state is missing {missing}. Run the required earlier stage(s), "
                "or use --stage all."
            )


def git_metadata(project_root: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=project_root, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(command("status", "--porcelain")),
    }
