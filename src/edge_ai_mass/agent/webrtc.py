"""aiortc-backed WebRTC preview media peer for Jetson devices."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from concurrent.futures import Future
from fractions import Fraction
from typing import Any

from edge_ai_mass.agent.camera import open_video_capture
from edge_ai_mass.agent.preview import (
    PreviewError,
    PreviewPeer,
    PreviewPeerEvent,
    PreviewSession,
    WebRTCAnswer,
)

logger = logging.getLogger(__name__)

try:
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCRtpSender,
        RTCSessionDescription,
        VideoStreamTrack,
    )
    from aiortc.mediastreams import MediaStreamError
    from aiortc.sdp import candidate_from_sdp
    from av import VideoFrame
except ImportError as exc:  # pragma: no cover - depends on optional runtime extra
    AIORTC_IMPORT_ERROR: ImportError | None = exc
    VideoStreamTrack = object  # type: ignore[assignment,misc]
    MediaStreamError = RuntimeError  # type: ignore[assignment,misc]
    VideoFrame = None  # type: ignore[assignment]
else:
    AIORTC_IMPORT_ERROR = None


class OpenCVCameraVideoTrack(VideoStreamTrack):  # type: ignore[misc]
    """Read the requested attached camera and expose frames to aiortc."""

    kind = "video"

    def __init__(
        self,
        *,
        camera_source: str,
        max_width: int,
        max_height: int,
        max_fps: int,
        on_first_frame: Any,
        on_error: Any,
    ) -> None:
        if AIORTC_IMPORT_ERROR is not None:
            raise _missing_aiortc_error()
        super().__init__()
        import cv2

        self._cv2 = cv2
        self._capture = open_video_capture(camera_source, cv2_module=cv2)
        if not self._capture.isOpened():
            self._capture.release()
            raise PreviewError(
                "camera_unavailable",
                f"Camera source {camera_source} could not be opened for WebRTC",
                retryable=True,
            )

        self.camera_source = camera_source
        self.max_width = max(1, int(max_width))
        self.max_height = max(1, int(max_height))
        self.max_fps = max(1, int(max_fps))
        self._on_first_frame = on_first_frame
        self._on_error = on_error
        self._first_frame_sent = False
        self._stopped = False
        self._clock_rate = 90_000
        self._frame_step = max(1, int(self._clock_rate / self.max_fps))
        self._pts = 0
        self._next_frame_time: float | None = None

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.max_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.max_height)
        self._capture.set(cv2.CAP_PROP_FPS, self.max_fps)
        logger.info(
            "Opened WebRTC camera source=%s max_resolution=%sx%s max_fps=%s",
            camera_source,
            self.max_width,
            self.max_height,
            self.max_fps,
        )

    async def recv(self) -> Any:
        if self._stopped:
            raise MediaStreamError

        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._next_frame_time is None:
            self._next_frame_time = now
        delay = self._next_frame_time - now
        if delay > 0:
            await asyncio.sleep(delay)
        self._next_frame_time = max(self._next_frame_time, now) + 1.0 / self.max_fps

        ok, frame = await asyncio.to_thread(self._capture.read)
        if not ok or frame is None:
            summary = f"Camera source {self.camera_source} stopped returning frames"
            self._on_error(summary)
            raise MediaStreamError

        frame = _resize_to_bounds(
            frame,
            max_width=self.max_width,
            max_height=self.max_height,
            cv2_module=self._cv2,
        )
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        self._pts += self._frame_step
        video_frame.pts = self._pts
        video_frame.time_base = Fraction(1, self._clock_rate)

        if not self._first_frame_sent:
            self._first_frame_sent = True
            height, width = frame.shape[:2]
            self._on_first_frame(
                {
                    "width": int(width),
                    "height": int(height),
                    "fps": self.max_fps,
                }
            )
        return video_frame

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._capture.release()
        super().stop()
        logger.info("Released WebRTC camera source=%s", self.camera_source)


class AiortcPreviewPeerFactory:
    """Create a real aiortc media peer for each preview session."""

    def __init__(self, *, operation_timeout_seconds: float = 20.0) -> None:
        self.operation_timeout_seconds = operation_timeout_seconds

    def __call__(self, session: PreviewSession) -> PreviewPeer:
        if AIORTC_IMPORT_ERROR is not None:
            raise _missing_aiortc_error()
        return AiortcPreviewPeer(
            session,
            operation_timeout_seconds=self.operation_timeout_seconds,
        )


class AiortcPreviewPeer:
    """Synchronous adapter around an aiortc peer running on its own loop."""

    def __init__(
        self,
        session: PreviewSession,
        *,
        operation_timeout_seconds: float = 20.0,
    ) -> None:
        if AIORTC_IMPORT_ERROR is not None:
            raise _missing_aiortc_error()
        self.session = session
        self.operation_timeout_seconds = operation_timeout_seconds
        self._events: queue.Queue[PreviewPeerEvent] = queue.Queue()
        self._loop_thread = _AsyncioLoopThread(
            name=f"preview-{session.request_id[:12]}"
        )
        self._peer_connection: Any = None
        self._video_track: OpenCVCameraVideoTrack | None = None
        self._connected = False
        self._first_frame: dict[str, Any] | None = None
        self._media_started_emitted = False
        self._closing = False
        self._selected_codec = _preferred_codec(session.supported_codecs)

    def create_answer(
        self,
        *,
        offer_sdp: str,
        session: PreviewSession,
        peer_id: str,
        metadata: dict[str, Any],
    ) -> WebRTCAnswer:
        if not offer_sdp.strip().startswith("v=0"):
            raise PreviewError(
                "webrtc_offer_invalid",
                "WebRTC offer SDP is missing or invalid",
                retryable=False,
            )
        try:
            answer_sdp = self._submit(
                self._create_answer(offer_sdp),
                timeout=self.operation_timeout_seconds,
            )
        except PreviewError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise PreviewError(
                "webrtc_offer_invalid",
                f"Could not apply WebRTC offer: {exc}",
                retryable=False,
            ) from exc

        return WebRTCAnswer(
            sdp=answer_sdp,
            media={
                "video": True,
                "audio": False,
                "codec": self._selected_codec,
            },
            local_candidates=[],
            media_flowing=False,
        )

    def add_remote_ice_candidate(self, candidate: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(candidate, dict) or not candidate.get("candidate"):
            raise PreviewError(
                "ice_failed",
                "ICE candidate payload is missing candidate data",
                retryable=True,
            )
        if self._peer_connection is None:
            raise PreviewError(
                "ice_failed",
                "ICE candidate received before peer connection creation",
                retryable=True,
            )
        try:
            self._submit(
                self._add_remote_candidate(candidate),
                timeout=self.operation_timeout_seconds,
            )
        except Exception as exc:
            raise PreviewError(
                "ice_failed",
                f"Could not apply remote ICE candidate: {exc}",
                retryable=True,
            ) from exc
        return []

    def poll_events(self) -> list[PreviewPeerEvent]:
        events: list[PreviewPeerEvent] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            if self._peer_connection is not None:
                self._submit(self._close_peer(), timeout=5.0)
            elif self._video_track is not None:
                self._video_track.stop()
        except Exception:
            logger.exception("Failed to close WebRTC peer cleanly")
        finally:
            self._loop_thread.stop()

    async def _create_answer(self, offer_sdp: str) -> str:
        if self._peer_connection is None:
            configuration = RTCConfiguration(
                iceServers=[_ice_server(value) for value in self.session.ice_servers]
            )
            self._peer_connection = RTCPeerConnection(configuration=configuration)
            self._register_peer_callbacks()
            self._video_track = OpenCVCameraVideoTrack(
                camera_source=self.session.camera_source,
                max_width=self.session.max_width,
                max_height=self.session.max_height,
                max_fps=self.session.max_fps,
                on_first_frame=self._on_first_frame,
                on_error=self._on_track_error,
            )
            sender = self._peer_connection.addTrack(self._video_track)
            _apply_codec_preferences(
                self._peer_connection,
                sender,
                self.session.supported_codecs,
            )
        await self._peer_connection.setRemoteDescription(
            RTCSessionDescription(sdp=offer_sdp, type="offer")
        )
        answer = await self._peer_connection.createAnswer()
        await self._peer_connection.setLocalDescription(answer)
        return str(self._peer_connection.localDescription.sdp)

    async def _add_remote_candidate(self, value: dict[str, Any]) -> None:
        candidate_sdp = str(value["candidate"])
        if candidate_sdp.startswith("candidate:"):
            candidate_sdp = candidate_sdp.split(":", 1)[1]
        candidate = candidate_from_sdp(candidate_sdp)
        candidate.sdpMid = value.get("sdpMid")
        candidate.sdpMLineIndex = value.get("sdpMLineIndex")
        await self._peer_connection.addIceCandidate(candidate)

    async def _close_peer(self) -> None:
        if self._video_track is not None:
            self._video_track.stop()
        await self._peer_connection.close()

    def _register_peer_callbacks(self) -> None:
        @self._peer_connection.on("connectionstatechange")
        async def on_connection_state_change() -> None:
            state = str(self._peer_connection.connectionState)
            logger.info(
                "WebRTC connection state changed: request_id=%s state=%s",
                self.session.request_id,
                state,
            )
            if state == "connected":
                self._connected = True
                self._maybe_emit_media_started()
            elif state == "failed":
                self._emit(
                    "failed",
                    {
                        "error_type": "ice_failed",
                        "summary": "WebRTC peer connection failed",
                        "retryable": True,
                    },
                )
            elif state in {"disconnected", "closed"} and not self._closing:
                self._emit("disconnected", {"reason": "peer_disconnected"})

        @self._peer_connection.on("iceconnectionstatechange")
        async def on_ice_state_change() -> None:
            logger.debug(
                "WebRTC ICE state changed: request_id=%s state=%s",
                self.session.request_id,
                self._peer_connection.iceConnectionState,
            )

    def _on_first_frame(self, details: dict[str, Any]) -> None:
        self._first_frame = details
        self._maybe_emit_media_started()

    def _on_track_error(self, summary: str) -> None:
        self._emit(
            "failed",
            {
                "error_type": "camera_unavailable",
                "summary": summary,
                "retryable": True,
            },
        )

    def _maybe_emit_media_started(self) -> None:
        if not self._connected or self._first_frame is None or self._media_started_emitted:
            return
        self._media_started_emitted = True
        self._emit(
            "media_started",
            {
                **self._first_frame,
                "selected_codec": self._selected_codec,
            },
        )

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        self._events.put(PreviewPeerEvent(kind=kind, payload=payload))

    def _submit(self, coroutine: Any, *, timeout: float) -> Any:
        return self._loop_thread.submit(coroutine, timeout=timeout)


class _AsyncioLoopThread:
    def __init__(self, *, name: str) -> None:
        self.loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=5.0):
            raise PreviewError(
                "preview_internal_error",
                "WebRTC asyncio loop did not start",
                retryable=True,
            )

    def submit(self, coroutine: Any, *, timeout: float) -> Any:
        future: Future[Any] = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        return future.result(timeout=timeout)

    def stop(self) -> None:
        if not self.loop.is_running():
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._started.set()
        self.loop.run_forever()
        self.loop.close()


def _ice_server(value: dict[str, Any]) -> Any:
    urls = value.get("urls") or []
    return RTCIceServer(
        urls=urls,
        username=value.get("username"),
        credential=value.get("credential"),
    )


def _apply_codec_preferences(
    peer_connection: Any,
    sender: Any,
    codec_preferences: list[str],
) -> None:
    transceiver = next(
        (
            value
            for value in peer_connection.getTransceivers()
            if value.sender is sender
        ),
        None,
    )
    if transceiver is None:
        return
    available = RTCRtpSender.getCapabilities("video").codecs
    preferred = []
    for codec_name in codec_preferences:
        preferred.extend(
            codec
            for codec in available
            if codec.mimeType.rsplit("/", 1)[-1].upper() == codec_name.upper()
            and codec not in preferred
        )
    if not preferred:
        raise PreviewError(
            "codec_unsupported",
            f"None of the requested codecs are available: {codec_preferences}",
            retryable=False,
        )
    transceiver.setCodecPreferences(preferred)


def _preferred_codec(codec_preferences: list[str]) -> str:
    return str(codec_preferences[0] if codec_preferences else "H264")


def _resize_to_bounds(
    frame: Any,
    *,
    max_width: int,
    max_height: int,
    cv2_module: Any,
) -> Any:
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    target = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2_module.resize(frame, target, interpolation=cv2_module.INTER_AREA)


def _missing_aiortc_error() -> PreviewError:
    return PreviewError(
        "preview_internal_error",
        "Real WebRTC preview requires aiortc and av. Install with: "
        "pip install -e '.[webrtc]'",
        retryable=False,
    )
