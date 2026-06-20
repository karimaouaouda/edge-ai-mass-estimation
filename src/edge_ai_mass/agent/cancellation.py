"""Thread-safe cooperative cancellation for inference workflows."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

from edge_ai_mass.agent.envelopes import utc_now_iso


class InferenceCancelled(RuntimeError):
    """Raised at a safe checkpoint after cancellation was requested."""

    is_inference_cancellation = True

    def __init__(self, reason: str, checkpoint: str) -> None:
        super().__init__(f"Inference cancelled at {checkpoint}: {reason}")
        self.reason = reason
        self.checkpoint = checkpoint


class InferenceExecution:
    """Mutable state and cancellation token for one inference workflow."""

    def __init__(self, *, request_id: str, correlation_id: str) -> None:
        self.request_id = request_id
        self.correlation_id = correlation_id
        self._state = "active"
        self._reason = "operator_requested"
        self._cancelled_at: str | None = None
        self._event = threading.Event()
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def is_cancellation_requested(self) -> bool:
        return self._event.is_set()

    def request_cancel(self, reason: str) -> str:
        with self._lock:
            if self._state == "active":
                self._state = "cancelling"
                self._reason = _normalize_reason(reason)
                self._event.set()
            return self._state

    def checkpoint(self, checkpoint: str) -> None:
        if self._event.is_set():
            with self._lock:
                reason = self._reason
            raise InferenceCancelled(reason, checkpoint)

    def try_complete(self) -> bool:
        """Atomically commit success unless a stop won the race."""

        with self._lock:
            if self._state != "active" or self._event.is_set():
                return False
            self._state = "completed"
            return True

    def mark_failed(self) -> None:
        with self._lock:
            if self._state == "active":
                self._state = "failed"

    def confirm_cancelled(self) -> dict[str, str]:
        with self._lock:
            if self._state != "cancelled":
                self._state = "cancelled"
                self._event.set()
                self._cancelled_at = utc_now_iso()
            return self._cancellation_payload_locked()

    def cancellation_payload(self) -> dict[str, str]:
        with self._lock:
            if self._state != "cancelled":
                raise RuntimeError("Inference cancellation has not been confirmed")
            return self._cancellation_payload_locked()

    def _cancellation_payload_locked(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "reason": self._reason,
            "cancelled_at": self._cancelled_at or utc_now_iso(),
        }


@dataclass(frozen=True, slots=True)
class CancellationDecision:
    status: str
    execution: InferenceExecution


class InferenceCancellationRegistry:
    """Track active and recent inference executions by either workflow id."""

    def __init__(self, *, max_records: int = 1024) -> None:
        self.max_records = max_records
        self._by_identity: dict[str, InferenceExecution] = {}
        self._records: OrderedDict[int, InferenceExecution] = OrderedDict()
        self._lock = threading.Lock()

    def begin(
        self,
        *,
        request_id: str,
        correlation_id: str,
    ) -> tuple[InferenceExecution, bool]:
        identities = _identities(request_id, correlation_id)
        with self._lock:
            existing = self._find_locked(identities)
            if existing is not None:
                return existing, False

            execution = InferenceExecution(
                request_id=request_id,
                correlation_id=correlation_id,
            )
            self._remember_locked(execution, identities)
            return execution, True

    def request_stop(
        self,
        *,
        request_id: str,
        correlation_id: str,
        reason: str,
    ) -> CancellationDecision:
        identities = _identities(request_id, correlation_id)
        unmatched = False
        with self._lock:
            execution = self._find_locked(identities)
            if execution is None:
                unmatched = True
                execution = InferenceExecution(
                    request_id=request_id,
                    correlation_id=correlation_id,
                )
                self._remember_locked(execution, identities)

        if unmatched:
            execution.request_cancel(reason)
            execution.confirm_cancelled()
            return CancellationDecision("confirmed", execution)

        state = execution.state
        if state in {"active", "cancelling"}:
            state = execution.request_cancel(reason)
            return CancellationDecision(
                "signalled" if state == "cancelling" else state,
                execution,
            )
        if state == "cancelled":
            return CancellationDecision("confirmed", execution)
        return CancellationDecision(state, execution)

    def _find_locked(self, identities: tuple[str, ...]) -> InferenceExecution | None:
        for identity in identities:
            execution = self._by_identity.get(identity)
            if execution is not None:
                self._records.move_to_end(id(execution))
                return execution
        return None

    def _remember_locked(
        self,
        execution: InferenceExecution,
        identities: tuple[str, ...],
    ) -> None:
        for identity in identities:
            self._by_identity[identity] = execution
        self._records[id(execution)] = execution
        self._prune_locked()

    def _prune_locked(self) -> None:
        attempts = len(self._records)
        while len(self._records) > self.max_records and attempts > 0:
            record_id, execution = self._records.popitem(last=False)
            attempts -= 1
            if execution.state in {"active", "cancelling"}:
                self._records[record_id] = execution
                continue
            for identity, value in list(self._by_identity.items()):
                if value is execution:
                    self._by_identity.pop(identity, None)


def _identities(request_id: str, correlation_id: str) -> tuple[str, ...]:
    return tuple(value for value in dict.fromkeys((request_id, correlation_id)) if value)


def _normalize_reason(reason: str) -> str:
    value = str(reason or "operator_requested").strip()
    return (value or "operator_requested")[:100]
