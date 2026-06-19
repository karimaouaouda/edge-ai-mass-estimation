"""Preview session and WebRTC signaling state manager."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


DEFAULT_CODECS = ("H264", "VP8")


class PreviewError(RuntimeError):
    """Raised when a preview command or signal cannot be accepted."""

    def __init__(self, error_type: str, summary: str, *, retryable: bool = False) -> None:
        super().__init__(summary)
        self.error_type = error_type
        self.summary = summary
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class PreviewEvent:
    """One backend preview event emitted by the preview manager."""

    event_name: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class WebRTCAnswer:
    """Result produced by a WebRTC peer after accepting an offer."""

    sdp: str
    media: dict[str, Any]
    local_candidates: list[dict[str, Any]] = field(default_factory=list)
    media_flowing: bool = False


class PreviewPeer(Protocol):
    """WebRTC peer adapter used by the preview manager."""

    def create_answer(
        self,
        *,
        offer_sdp: str,
        session: "PreviewSession",
        peer_id: str,
        metadata: dict[str, Any],
    ) -> WebRTCAnswer:
        """Apply a remote offer and return a local answer."""

    def add_remote_ice_candidate(self, candidate: dict[str, Any]) -> list[dict[str, Any]]:
        """Apply a remote ICE candidate and return any local candidates to publish."""

    def close(self) -> None:
        """Close media and signaling resources."""


class SimulatedPreviewPeer:
    """Contract-compatible peer for local testing and non-WebRTC environments.

    It does not send camera frames. Production Jetson deployments should inject
    an aiortc or GStreamer-backed peer, while this adapter keeps MQTT signaling
    and backend workflow tests deterministic.
    """

    def create_answer(
        self,
        *,
        offer_sdp: str,
        session: "PreviewSession",
        peer_id: str,
        metadata: dict[str, Any],
    ) -> WebRTCAnswer:
        if not offer_sdp.strip().startswith("v=0"):
            raise PreviewError(
                "webrtc_offer_invalid",
                "WebRTC offer SDP is missing or invalid",
                retryable=False,
            )

        codec = session.supported_codecs[0] if session.supported_codecs else "H264"
        answer_sdp = "\r\n".join(
            [
                "v=0",
                "o=edge-ai-mass 0 0 IN IP4 127.0.0.1",
                "s=DrovenAI Preview",
                "t=0 0",
                "m=video 9 UDP/TLS/RTP/SAVPF 96",
                "a=rtpmap:96 " + codec + "/90000",
                "a=sendonly",
                "",
            ]
        )
        return WebRTCAnswer(
            sdp=answer_sdp,
            media={"video": True, "audio": False, "codec": codec},
            local_candidates=[
                {
                    "candidate": "candidate:edge-simulated 1 udp 1 127.0.0.1 9 typ host",
                    "sdpMid": "0",
                    "sdpMLineIndex": 0,
                }
            ],
            media_flowing=True,
        )

    def add_remote_ice_candidate(self, candidate: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(candidate, dict) or not candidate.get("candidate"):
            raise PreviewError(
                "ice_failed",
                "ICE candidate payload is missing candidate data",
                retryable=True,
            )
        return []

    def close(self) -> None:
        return


class SimulatedPreviewPeerFactory:
    """Factory for local preview peers."""

    def __call__(self, session: "PreviewSession") -> PreviewPeer:
        return SimulatedPreviewPeer()


@dataclass(slots=True)
class PreviewSession:
    """State for one backend-owned preview session."""

    request_id: str
    correlation_id: str
    mode: str
    camera_source: str
    created_at: float
    expires_at: float
    supported_codecs: list[str]
    max_width: int
    max_height: int
    max_fps: int
    audio_enabled: bool = False
    peer_id: str | None = None
    state: str = "ready"
    selected_codec: str | None = None
    peer: PreviewPeer | None = None

    def is_expired(self, now: float | None = None) -> bool:
        return (now if now is not None else time.time()) >= self.expires_at

    def ready_payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "mode": self.mode,
            "camera_source": self.camera_source,
            "expires_at": _iso_from_timestamp(self.expires_at),
            "webrtc": {
                "role": "answerer",
                "supported_codecs": self.supported_codecs,
                "max_width": self.max_width,
                "max_height": self.max_height,
                "max_fps": self.max_fps,
                "ice_candidate_policy": "all",
            },
        }


class PreviewManager:
    """Track preview sessions and bind WebRTC signaling to active sessions."""

    def __init__(
        self,
        *,
        peer_factory: Any | None = None,
        camera_available: Any | None = None,
        time_fn: Any = time.time,
    ) -> None:
        self.sessions: dict[str, PreviewSession] = {}
        self.peer_factory = peer_factory or SimulatedPreviewPeerFactory()
        self.camera_available = camera_available or (lambda _source: True)
        self.time_fn = time_fn

    def start(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        correlation_id: str,
        preview_enabled: bool = True,
    ) -> dict[str, Any]:
        if not preview_enabled:
            raise PreviewError(
                "permission_denied",
                "Preview capability is disabled for this device",
                retryable=False,
            )

        camera_source = str(payload.get("camera_source") or "camera:0")
        if not self.camera_available(camera_source):
            raise PreviewError(
                "camera_unavailable",
                f"Camera source {camera_source} could not be opened",
                retryable=True,
            )

        ttl = int(payload.get("ttl_seconds") or 600)
        now = self.time_fn()
        session = PreviewSession(
            request_id=request_id,
            correlation_id=correlation_id,
            mode=str(payload.get("mode") or "low_fps"),
            camera_source=camera_source,
            created_at=now,
            expires_at=now + max(ttl, 1),
            **_webrtc_session_options(payload.get("webrtc")),
        )
        self.sessions[request_id] = session
        return session.ready_payload()

    def handle_signal(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        correlation_id: str,
    ) -> list[PreviewEvent]:
        signal_type = str(payload.get("signal_type") or "").strip()
        session = self._session_for_signal(request_id, correlation_id)

        if signal_type in {"offer", "renegotiate"}:
            return self._handle_offer(session, payload)
        if signal_type == "ice_candidate":
            return self._handle_ice_candidate(session, payload)
        if signal_type == "close":
            stopped = self.stop(
                {"reason": "peer_closed"},
                request_id=request_id,
                correlation_id=correlation_id,
            )
            return [PreviewEvent("preview.stopped", stopped)]

        raise PreviewError(
            "preview_internal_error",
            f"Unsupported preview signal_type: {signal_type!r}",
            retryable=False,
        )

    def stop(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        correlation_id: str,
    ) -> dict[str, Any]:
        session = self.sessions.pop(request_id, None)
        if session is not None and session.peer is not None:
            session.peer.close()
        return {
            "request_id": request_id,
            "correlation_id": correlation_id,
            "peer_id": session.peer_id if session is not None else payload.get("peer_id"),
            "reason": str(payload.get("reason") or "operator_requested"),
            "stopped_at": _iso_from_timestamp(self.time_fn()),
            "was_active": session is not None and session.state == "active",
        }

    def expire_sessions(self) -> list[PreviewEvent]:
        events: list[PreviewEvent] = []
        now = self.time_fn()
        for request_id, session in list(self.sessions.items()):
            if not session.is_expired(now):
                continue
            payload = self.stop(
                {"reason": "ttl_expired"},
                request_id=request_id,
                correlation_id=session.correlation_id,
            )
            events.append(PreviewEvent("preview.stopped", payload))
        return events

    def stop_all(self) -> None:
        for session in self.sessions.values():
            if session.peer is not None:
                session.peer.close()
        self.sessions.clear()

    def _handle_offer(
        self,
        session: PreviewSession,
        payload: dict[str, Any],
    ) -> list[PreviewEvent]:
        print(f"Handling offer for session {session.request_id} with payload: {payload}")
        peer_id = str(payload.get("peer_id") or "")
        offer_sdp = str(payload.get("sdp") or "")
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        session.peer_id = peer_id
        session.state = "connecting"
        if session.peer is None:
            session.peer = self.peer_factory(session)

        answer = session.peer.create_answer(
            offer_sdp=offer_sdp,
            session=session,
            peer_id=peer_id,
            metadata=metadata,
        )
        session.selected_codec = str(answer.media.get("codec") or session.supported_codecs[0])

        events = [
            PreviewEvent(
                "preview.webrtc_answer",
                {
                    "request_id": session.request_id,
                    "correlation_id": session.correlation_id,
                    "peer_id": peer_id,
                    "sdp": answer.sdp,
                    "media": answer.media,
                },
            )
        ]
        events.extend(
            PreviewEvent(
                "preview.webrtc_ice_candidate",
                {
                    "request_id": session.request_id,
                    "correlation_id": session.correlation_id,
                    "peer_id": peer_id,
                    "candidate": candidate,
                },
            )
            for candidate in answer.local_candidates
        )
        if answer.media_flowing:
            session.state = "active"
            events.append(
                PreviewEvent(
                    "preview.started",
                    {
                        "request_id": session.request_id,
                        "correlation_id": session.correlation_id,
                        "peer_id": peer_id,
                        "started_at": _iso_from_timestamp(self.time_fn()),
                        "selected_codec": session.selected_codec,
                        "selected_resolution": {
                            "width": session.max_width,
                            "height": session.max_height,
                        },
                        "fps": session.max_fps,
                    },
                )
            )
        return events

    def _handle_ice_candidate(
        self,
        session: PreviewSession,
        payload: dict[str, Any],
    ) -> list[PreviewEvent]:
        if session.peer is None:
            raise PreviewError(
                "ice_failed",
                "ICE candidate received before WebRTC offer",
                retryable=True,
            )
        peer_id = str(payload.get("peer_id") or session.peer_id or "")
        candidate = payload.get("candidate")
        local_candidates = session.peer.add_remote_ice_candidate(candidate)
        return [
            PreviewEvent(
                "preview.webrtc_ice_candidate",
                {
                    "request_id": session.request_id,
                    "correlation_id": session.correlation_id,
                    "peer_id": peer_id,
                    "candidate": local_candidate,
                },
            )
            for local_candidate in local_candidates
        ]

    def _session_for_signal(self, request_id: str, correlation_id: str) -> PreviewSession:
        session = self.sessions.get(request_id)
        if session is None:
            raise PreviewError(
                "session_expired",
                f"No active preview session for request_id {request_id}",
                retryable=False,
            )
        if session.is_expired(self.time_fn()):
            self.stop(
                {"reason": "ttl_expired"},
                request_id=request_id,
                correlation_id=session.correlation_id,
            )
            raise PreviewError(
                "session_expired",
                f"Preview session {request_id} has expired",
                retryable=False,
            )
        if session.correlation_id != correlation_id:
            raise PreviewError(
                "permission_denied",
                "Preview signaling correlation_id does not match the active session",
                retryable=False,
            )
        return session


def _webrtc_session_options(value: Any) -> dict[str, Any]:
    webrtc = value if isinstance(value, dict) else {}
    video = webrtc.get("video") if isinstance(webrtc.get("video"), dict) else {}
    audio = webrtc.get("audio") if isinstance(webrtc.get("audio"), dict) else {}
    codec_preferences = video.get("codec_preferences") or DEFAULT_CODECS
    supported_codecs = [str(codec) for codec in codec_preferences if str(codec)]
    if not supported_codecs:
        supported_codecs = list(DEFAULT_CODECS)
    return {
        "supported_codecs": supported_codecs,
        "max_width": int(video.get("max_width") or 1280),
        "max_height": int(video.get("max_height") or 720),
        "max_fps": int(video.get("max_fps") or 10),
        "audio_enabled": bool(audio.get("enabled", False)),
    }


def _iso_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00",
        "Z",
    )
