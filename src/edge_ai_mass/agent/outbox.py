"""Durable local outbox for backend-bound telemetry, media, and MQTT events."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class OutboxRecord:
    """One durable delivery record."""

    id: str
    kind: str
    payload: dict[str, Any]
    created_at: float
    next_attempt_at: float
    attempts: int = 0
    workflow_id: str = ""
    last_error: str = ""
    dead_lettered: bool = False
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("path", None)
        return data


class FileOutbox:
    """Filesystem-backed outbox with bounded retry and dead-letter support."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_attempts: int = 8,
        retention_seconds: int = 7 * 24 * 60 * 60,
    ) -> None:
        self.root = Path(root)
        self.pending_dir = self.root / "pending"
        self.dead_dir = self.root / "dead"
        self.max_attempts = max_attempts
        self.retention_seconds = retention_seconds

    def start(self) -> None:
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.dead_dir.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        workflow_id: str = "",
        record_id: str | None = None,
        now: float | None = None,
    ) -> OutboxRecord:
        self.start()
        created_at = now if now is not None else time.time()
        record = OutboxRecord(
            id=record_id or str(uuid.uuid4()),
            kind=kind,
            payload=payload,
            created_at=created_at,
            next_attempt_at=created_at,
            workflow_id=workflow_id,
        )
        path = self.pending_dir / f"{int(created_at * 1000)}-{record.id}.json"
        record.path = str(path)
        self._write_record(path, record)
        return record

    def due_records(self, *, limit: int = 100, now: float | None = None) -> list[OutboxRecord]:
        self.start()
        current = now if now is not None else time.time()
        records: list[OutboxRecord] = []
        for path in sorted(self.pending_dir.glob("*.json")):
            record = self._read_record(path)
            if record.dead_lettered:
                continue
            if current - record.created_at > self.retention_seconds:
                self.mark_failed(record, "retention expired", permanent=True)
                continue
            if record.next_attempt_at <= current:
                records.append(record)
            if len(records) >= limit:
                break
        return records

    def mark_sent(self, record: OutboxRecord) -> None:
        path = Path(record.path)
        if path.exists():
            path.unlink()

    def mark_failed(
        self,
        record: OutboxRecord,
        error: str,
        *,
        permanent: bool = False,
        now: float | None = None,
    ) -> OutboxRecord:
        current = now if now is not None else time.time()
        record.attempts += 1
        record.last_error = error[:500]
        if permanent or record.attempts >= self.max_attempts:
            record.dead_lettered = True
            self._move_to_dead(record)
            return record

        delay = min(300.0, 2.0 ** max(record.attempts - 1, 0))
        record.next_attempt_at = current + delay
        self._write_record(Path(record.path), record)
        return record

    def _read_record(self, path: Path) -> OutboxRecord:
        data = json.loads(path.read_text(encoding="utf-8"))
        record = OutboxRecord(**data)
        record.path = str(path)
        return record

    def _write_record(self, path: Path, record: OutboxRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _move_to_dead(self, record: OutboxRecord) -> None:
        source = Path(record.path)
        destination = self.dead_dir / source.name
        record.path = str(destination)
        self._write_record(destination, record)
        if source.exists():
            source.unlink()
