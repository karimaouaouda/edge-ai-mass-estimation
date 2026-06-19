# WebRTC Preview Signaling

Preview is an operator-controlled WebRTC session between the Laravel/browser side and the edge device.

MQTT carries control and signaling messages. WebRTC carries media. Do not send preview frames through MQTT.

## Required sequence

The required flow is:

1. Operator asks Laravel to open a preview session.
2. Laravel creates a `preview_session` with `request_id`, `correlation_id`, `ttl_seconds`, and `camera_source`.
3. Laravel publishes `preview.start_requested` to the edge command topic.
4. Edge device validates the request and reserves camera resources.
5. Edge publishes `preview.ready`.
6. Browser/Laravel creates a WebRTC offer.
7. Laravel forwards the offer to the edge with `preview.webrtc_signal`.
8. Edge creates a peer connection and publishes `preview.webrtc_answer`.
9. Browser/Laravel and edge exchange ICE candidates.
10. When ICE connects and media is flowing, edge publishes `preview.started`.
11. Laravel marks the preview session active and shows the stream.
12. Stop happens by TTL expiry, operator stop, edge failure, or peer disconnect.

## Roles

Default role split:

- Browser/operator side: WebRTC offerer
- Edge device: WebRTC answerer and media sender
- Laravel: session owner, authorization gate, MQTT signaling relay, and audit/message trace
- MQTT: signaling and control transport
- WebRTC: encrypted media transport

The edge device MUST NOT accept a WebRTC offer unless there is an active `preview.start_requested` session for the same `request_id`.

## Topics

Laravel to edge:

```text
drovenai/devices/{device_id}/commands/preview/start
drovenai/devices/{device_id}/commands/preview/signal
drovenai/devices/{device_id}/commands/preview/stop
```

Edge to Laravel:

```text
drovenai/devices/{device_id}/events/preview.ready
drovenai/devices/{device_id}/events/preview.webrtc_answer
drovenai/devices/{device_id}/events/preview.webrtc_ice_candidate
drovenai/devices/{device_id}/events/preview.started
drovenai/devices/{device_id}/events/preview.stopped
drovenai/devices/{device_id}/events/preview.failed
```

All messages use the standard MQTT envelope from `03-mqtt-topics-and-events.md`.

## Start command

Topic:

```text
drovenai/devices/{device_id}/commands/preview/start
```

Envelope `event_name`:

```text
preview.start_requested
```

Payload:

```json
{
  "request_id": "uuid",
  "correlation_id": "uuid",
  "mode": "low_fps",
  "camera_source": "camera:0",
  "ttl_seconds": 600,
  "webrtc": {
    "signaling": "mqtt",
    "role": "answerer",
    "video": {
      "enabled": true,
      "codec_preferences": ["H264", "VP8"],
      "max_width": 1280,
      "max_height": 720,
      "max_fps": 10
    },
    "audio": {
      "enabled": false
    },
    "ice_servers": [
      {
        "urls": ["stun:stun.l.google.com:19302"]
      }
    ]
  }
}
```

The Laravel preview start action sends `request_id`, `correlation_id`, `mode`, `camera_source`, `ttl_seconds`, and the `webrtc` negotiation preferences. `PublishPreviewSignalAction` publishes offers and Laravel's MQTT consumer applies edge answers, ICE candidates, and lifecycle events to the preview session.

Edge behavior:

1. Check `device_id`.
2. Check preview capability is enabled.
3. Check camera is available.
4. Reserve the camera for this `request_id`.
5. Create local preview session state with expiry.
6. Publish `preview.ready`.

## Ready event

Topic:

```text
drovenai/devices/{device_id}/events/preview.ready
```

Envelope `event_name`:

```text
preview.ready
```

Payload:

```json
{
  "request_id": "uuid",
  "correlation_id": "uuid",
  "mode": "low_fps",
  "camera_source": "camera:0",
  "expires_at": "2026-06-18T12:10:00Z",
  "webrtc": {
    "role": "answerer",
    "supported_codecs": ["H264", "VP8"],
    "max_width": 1280,
    "max_height": 720,
    "max_fps": 10,
    "ice_candidate_policy": "all"
  }
}
```

Laravel/browser MUST wait for this event before sending the WebRTC offer.

## Offer signal

Topic:

```text
drovenai/devices/{device_id}/commands/preview/signal
```

Envelope `event_name`:

```text
preview.webrtc_signal
```

Payload:

```json
{
  "request_id": "uuid",
  "correlation_id": "uuid",
  "signal_type": "offer",
  "peer_id": "operator-browser-uuid",
  "sdp": "v=0...",
  "candidate": null,
  "metadata": {
    "ice_restart": false
  }
}
```

Edge behavior:

1. Verify the session exists and is ready.
2. Verify `correlation_id` matches the active preview session.
3. Create an `RTCPeerConnection`.
4. Apply the remote SDP offer.
5. Attach the preview video track.
6. Create an SDP answer.
7. Publish `preview.webrtc_answer`.

## Answer event

Topic:

```text
drovenai/devices/{device_id}/events/preview.webrtc_answer
```

Envelope `event_name`:

```text
preview.webrtc_answer
```

Payload:

```json
{
  "request_id": "uuid",
  "correlation_id": "uuid",
  "peer_id": "operator-browser-uuid",
  "sdp": "v=0...",
  "media": {
    "video": true,
    "audio": false,
    "codec": "H264"
  }
}
```

Laravel/browser applies this answer to the browser peer connection.

## ICE candidate exchange

Browser/Laravel to edge:

```json
{
  "request_id": "uuid",
  "correlation_id": "uuid",
  "signal_type": "ice_candidate",
  "peer_id": "operator-browser-uuid",
  "sdp": null,
  "candidate": {
    "candidate": "candidate:...",
    "sdpMid": "0",
    "sdpMLineIndex": 0
  }
}
```

Edge to Laravel:

Topic:

```text
drovenai/devices/{device_id}/events/preview.webrtc_ice_candidate
```

Envelope `event_name`:

```text
preview.webrtc_ice_candidate
```

Payload:

```json
{
  "request_id": "uuid",
  "correlation_id": "uuid",
  "peer_id": "operator-browser-uuid",
  "candidate": {
    "candidate": "candidate:...",
    "sdpMid": "0",
    "sdpMLineIndex": 0
  }
}
```

Both sides SHOULD support trickle ICE. If trickle ICE is disabled, candidates may be included in SDP, but the payload shape should remain compatible.

## Started event

Topic:

```text
drovenai/devices/{device_id}/events/preview.started
```

Envelope `event_name`:

```text
preview.started
```

Payload:

```json
{
  "request_id": "uuid",
  "correlation_id": "uuid",
  "peer_id": "operator-browser-uuid",
  "started_at": "2026-06-18T12:00:05Z",
  "selected_codec": "H264",
  "selected_resolution": {
    "width": 1280,
    "height": 720
  },
  "fps": 10
}
```

The edge MUST publish this only after the peer connection reaches a connected state and media frames are being sent.

## Stop and close

Operator stop:

1. Laravel publishes `preview.stop_requested` on `commands/preview/stop`.
2. Edge closes the peer connection.
3. Edge releases camera resources.
4. Edge publishes `preview.stopped`.

Browser-side close or disconnect:

1. Laravel MAY publish `preview.webrtc_signal` with `signal_type=close`.
2. Edge closes the peer connection.
3. Edge publishes `preview.stopped`.

TTL expiry:

1. Edge closes the peer connection when `ttl_seconds` expires.
2. Edge publishes `preview.stopped` with `reason=ttl_expired`.

Stopped payload:

```json
{
  "request_id": "uuid",
  "correlation_id": "uuid",
  "peer_id": "operator-browser-uuid",
  "reason": "operator_requested",
  "stopped_at": "2026-06-18T12:05:00Z"
}
```

## Failure event

Topic:

```text
drovenai/devices/{device_id}/events/preview.failed
```

Envelope `event_name`:

```text
preview.failed
```

Payload:

```json
{
  "request_id": "uuid",
  "correlation_id": "uuid",
  "error_type": "camera_unavailable",
  "summary": "Camera source camera:0 could not be opened",
  "retryable": true,
  "failed_at": "2026-06-18T12:00:03Z"
}
```

Recommended `error_type` values:

- `camera_unavailable`
- `webrtc_offer_invalid`
- `ice_failed`
- `codec_unsupported`
- `permission_denied`
- `session_expired`
- `preview_internal_error`

## Security requirements

The edge device MUST:

- Accept signaling only after `preview.start_requested`.
- Reject offers for unknown or expired `request_id`.
- Reject messages whose `device_id` does not match the local device.
- Reject signaling whose `correlation_id` does not match the active session.
- Close the peer connection when `preview.stop_requested` is received.
- Stop media when `ttl_seconds` expires.
- Never log full SDP if it may contain sensitive IP addresses in production logs.

Laravel/browser MUST:

- Authorize the operator before creating a preview session.
- Bind all signaling messages to the active `preview_session`.
- Forward signaling only to the device that owns the session.
- Stop the session when the browser disconnects.
- Prefer TURN for production networks where direct ICE is unreliable.

## Python implementation note

Python agents can implement this with `aiortc` for MVPs or a GStreamer-backed WebRTC pipeline for Jetson hardware acceleration.

The preview manager SHOULD expose:

```python
async def handle_preview_start(envelope: dict) -> None: ...
async def handle_webrtc_signal(envelope: dict) -> None: ...
async def handle_preview_stop(envelope: dict) -> None: ...
```

The manager SHOULD keep session state keyed by `request_id`:

```python
{
    "request_id": "uuid",
    "correlation_id": "uuid",
    "camera_source": "camera:0",
    "peer_id": "operator-browser-uuid",
    "expires_at": "...",
    "state": "ready|connecting|active|stopped|failed",
}
```
