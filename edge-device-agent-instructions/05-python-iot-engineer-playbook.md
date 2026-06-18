# Python And IoT Engineer Playbook

This file translates the backend contract into implementation tasks for a Python edge agent.

## Recommended Python packages

Core:

```text
paho-mqtt
requests or httpx
pydantic
tenacity
python-dotenv
psutil
PyYAML
```

WebRTC preview, choose one stack:

```text
aiortc
av
```

or a GStreamer-based stack when Jetson hardware acceleration is required.

Computer vision and inference, chosen by hardware/runtime:

```text
opencv-python
numpy
ultralytics
torch
onnxruntime
```

Optional:

```text
orjson
structlog
cryptography
```

## Suggested package layout

```text
drovenai_agent/
  __init__.py
  main.py
  config.py
  secrets.py
  identity.py
  envelopes.py
  mqtt_client.py
  http_client.py
  telemetry.py
  outbox.py
  camera.py
  inference/
    runner.py
    schemas.py
  media.py
  model_manager.py
  preview.py
  errors.py
```

## Implementation tasks for AI agents

### Task 1: Build config and identity

MUST implement:

- Load `device.id`, backend URL, MQTT host/port, topic prefix, and intervals from YAML/env.
- Load secrets from environment or OS secret store.
- Fail startup if `device.id` or HTTP token is missing.
- Expose `agent_version`, `firmware_version`, and capabilities.

### Task 2: Build HTTP client

MUST implement:

- `submit_telemetry(payload: dict) -> dict`
- `upload_media(path, media_type, request_id=None, correlation_id=None, metadata=None) -> dict`
- Add `X-Device-Token` to every request.
- Add `Accept: application/json`.
- Use timeouts.
- Retry network and 5xx failures.
- Do not retry 403, 404, or 422 without changing the input.

### Task 3: Build MQTT client

MUST implement:

- Connect to configured broker.
- Use MQTT username/password when provided.
- Subscribe to `drovenai/devices/{device_id}/commands/#`.
- Dispatch known commands by `event_name`.
- Publish events to `drovenai/devices/{device_id}/events/{event_name}`.
- Reconnect automatically.
- Record recently processed `message_id` values for dedupe.

### Task 4: Build envelope helper

MUST implement:

- `new_envelope(event_name, payload, correlation_id=None, request_id=None)`
- `parse_envelope(bytes) -> Envelope`
- `validate_command(envelope, expected_device_id)`
- Preserve `correlation_id` and `request_id` from commands in events and uploads.

### Task 5: Build telemetry loop

MUST implement:

- Periodic telemetry every configured interval.
- CPU percent, memory percent, temperature, disk percent.
- Camera connected flag.
- Inference busy flag.
- Active model versions.
- `status=online` during healthy operation.
- `health_status=warning` or `critical` when thresholds are crossed.

### Task 6: Build inference command handler

Input command:

- topic: `drovenai/devices/{device_id}/commands/inference`
- event: `inference.requested`

MUST implement:

- Parse `source_type`, `source_reference`, and `options`.
- Capture/load source.
- Run models.
- Build normalized result objects.
- Publish `inference.result`.
- Upload media when `return_annotated_image` or `return_depth_preview` is true.
- Report `inference.failed` when the pipeline fails.

### Task 7: Build media uploader

MUST implement:

- Compute SHA-256 before upload.
- Upload images as multipart form data.
- Use `request_id` and `correlation_id`.
- Store upload result in local workflow state.
- Retry safely on network failure.

### Task 8: Build model deployment handler

Input command:

- topic: `drovenai/devices/{device_id}/commands/model-deployment`
- event: `model_deployment.requested`

MUST implement:

- Validate `target_device_id` equals local device id.
- Download or resolve each component.
- Verify checksum when present.
- Stage model files before activation.
- Keep previous active model until the new model is verified.
- Publish progress events.
- Update telemetry `active_models`.

### Task 9: Build preview handler

Input commands:

- `preview.start_requested`
- `preview.stop_requested`

MUST implement:

- Start a low-FPS preview stream or a simulated preview.
- Publish `preview.ready` before WebRTC signaling starts.
- Accept WebRTC offers on `preview.webrtc_signal`.
- Return SDP answers with `preview.webrtc_answer`.
- Exchange ICE candidates with `preview.webrtc_ice_candidate`.
- Send preview media over WebRTC only.
- Enforce `ttl_seconds`.
- Publish `preview.started`, `preview.stopped`, or `preview.failed`.
- Release camera resources on stop.

The full handshake is defined in `07-webrtc-preview-signaling.md`.

### Task 10: Build local outbox

MUST implement:

- Durable storage for unsent HTTP telemetry, media upload jobs, and MQTT events.
- Backoff scheduling.
- Maximum retention policy.
- Dead-letter records after repeated permanent failures.
- Redacted error logs.

## Result object normalization

Each detected object SHOULD use this shape:

```json
{
  "class_label": "bottle",
  "material_label": "plastic",
  "confidence": 0.94,
  "bbox": {
    "x": 10,
    "y": 20,
    "w": 100,
    "h": 140
  },
  "estimated_geometry": {
    "area_px": 14000
  },
  "estimated_mass_grams": 120.5,
  "mass_method": "depth_estimate",
  "features": {}
}
```

Backend requirements:

- `class_label` is required.
- Other fields are optional but should be included when available.
- `bbox`, `estimated_geometry`, and `features` are JSON objects.

## Error handling requirements

The agent SHOULD classify errors:

- `invalid_command`
- `camera_unavailable`
- `inference_failed`
- `media_upload_failed`
- `model_download_failed`
- `checksum_mismatch`
- `preview_failed`
- `auth_failed`
- `network_unavailable`

Error event payload:

```json
{
  "error_type": "inference_failed",
  "summary": "Detector model failed to run",
  "request_id": "uuid",
  "correlation_id": "uuid",
  "retryable": false,
  "details": {
    "stage": "detector"
  }
}
```

Never include secrets, full stack traces with tokens, or raw image bytes in error events.
