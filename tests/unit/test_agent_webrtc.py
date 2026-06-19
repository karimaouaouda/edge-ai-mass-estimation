"""Tests for real-preview integration boundaries without camera hardware."""

from __future__ import annotations

import numpy as np
import pytest

from edge_ai_mass.agent.camera import video_capture_spec
from edge_ai_mass.agent.config import AgentConfig
from edge_ai_mass.agent.preview import (
    PreviewError,
    PreviewManager,
    PreviewPeerEvent,
    WebRTCAnswer,
)
from edge_ai_mass.agent.runner import EdgeDeviceAgent
from edge_ai_mass.agent.webrtc import AiortcPreviewPeerFactory, _resize_to_bounds


class DeferredPeer:
    def __init__(self) -> None:
        self.events: list[PreviewPeerEvent] = []
        self.closed = False

    def create_answer(self, *, offer_sdp, session, peer_id, metadata):
        return WebRTCAnswer(
            sdp="v=0\r\nanswer",
            media={"video": True, "audio": False, "codec": "H264"},
            media_flowing=False,
        )

    def add_remote_ice_candidate(self, candidate):
        return []

    def poll_events(self):
        events = list(self.events)
        self.events.clear()
        return events

    def close(self):
        self.closed = True


def _start_manager_with_peer(peer: DeferredPeer) -> PreviewManager:
    manager = PreviewManager(
        peer_factory=lambda _session: peer,
        camera_available=lambda _source: True,
        time_fn=lambda: 100.0,
    )
    manager.start(
        {
            "camera_source": "camera:0",
            "ttl_seconds": 60,
            "webrtc": {
                "video": {
                    "codec_preferences": ["H264", "VP8"],
                    "max_width": 1280,
                    "max_height": 720,
                    "max_fps": 10,
                }
            },
        },
        request_id="request-1",
        correlation_id="correlation-1",
    )
    return manager


def test_camera_source_resolver_supports_index_url_and_gstreamer():
    class FakeCv2:
        CAP_GSTREAMER = 1800

    assert video_capture_spec("camera:2").source == 2
    assert video_capture_spec("3").source == 3
    assert video_capture_spec("rtsp://camera/live").source == "rtsp://camera/live"
    gstreamer = video_capture_spec(
        "gstreamer:nvarguscamerasrc ! appsink",
        cv2_module=FakeCv2,
    )
    assert gstreamer.source == "nvarguscamerasrc ! appsink"
    assert gstreamer.backend == FakeCv2.CAP_GSTREAMER


def test_started_event_waits_for_peer_connection_and_first_frame():
    peer = DeferredPeer()
    manager = _start_manager_with_peer(peer)

    offer_events = manager.handle_signal(
        {
            "signal_type": "offer",
            "peer_id": "browser-1",
            "sdp": "v=0\r\n",
        },
        request_id="request-1",
        correlation_id="correlation-1",
    )

    assert [event.event_name for event in offer_events] == ["preview.webrtc_answer"]
    assert manager.sessions["request-1"].state == "connecting"

    peer.events.append(
        PreviewPeerEvent(
            "media_started",
            {
                "width": 640,
                "height": 480,
                "fps": 10,
                "selected_codec": "H264",
            },
        )
    )
    runtime_events = manager.poll_events()

    assert [event.event_name for event in runtime_events] == ["preview.started"]
    assert runtime_events[0].payload["selected_resolution"] == {
        "width": 640,
        "height": 480,
    }
    assert manager.sessions["request-1"].state == "active"


def test_async_peer_failure_closes_and_removes_session():
    peer = DeferredPeer()
    manager = _start_manager_with_peer(peer)
    manager.handle_signal(
        {
            "signal_type": "offer",
            "peer_id": "browser-1",
            "sdp": "v=0\r\n",
        },
        request_id="request-1",
        correlation_id="correlation-1",
    )
    peer.events.append(
        PreviewPeerEvent(
            "failed",
            {
                "error_type": "camera_unavailable",
                "summary": "Camera stopped returning frames",
                "retryable": True,
            },
        )
    )

    events = manager.poll_events()

    assert events[0].event_name == "preview.failed"
    assert events[0].payload["error_type"] == "camera_unavailable"
    assert "request-1" not in manager.sessions
    assert peer.closed is True


def test_failed_offer_cleans_up_session():
    class FailingPeer(DeferredPeer):
        def create_answer(self, *, offer_sdp, session, peer_id, metadata):
            raise PreviewError("webrtc_offer_invalid", "bad offer")

    peer = FailingPeer()
    manager = _start_manager_with_peer(peer)

    with pytest.raises(PreviewError, match="bad offer"):
        manager.handle_signal(
            {
                "signal_type": "offer",
                "peer_id": "browser-1",
                "sdp": "v=0\r\n",
            },
            request_id="request-1",
            correlation_id="correlation-1",
        )

    assert "request-1" not in manager.sessions
    assert peer.closed is True


def test_device_agent_uses_aiortc_peer_factory_by_default():
    config = AgentConfig.from_mapping({"device": {"id": "jetson-01"}})

    agent = EdgeDeviceAgent(
        config,
        preview_camera_available=lambda _source: True,
    )

    assert isinstance(agent.preview_manager.peer_factory, AiortcPreviewPeerFactory)


def test_frame_resize_preserves_aspect_ratio():
    import cv2

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    resized = _resize_to_bounds(
        frame,
        max_width=1280,
        max_height=720,
        cv2_module=cv2,
    )

    assert resized.shape == (720, 1280, 3)
