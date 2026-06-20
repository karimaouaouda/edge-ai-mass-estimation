# MQTT Topics And Events

Default topic prefix:

```text
drovenai
```

The prefix comes from `MQTT_TOPIC_PREFIX`.

Topic patterns:

```text
{prefix}/devices/{device_id}/commands/{command}
{prefix}/devices/{device_id}/events/{event_name}
{prefix}/devices/{device_id}/telemetry
```

The current Laravel `TopicBuilder` generates these patterns. Device-specific ACLs should prevent cross-device access.

## Envelope

MQTT command and event payloads MUST be JSON objects.

Envelope shape:

```json
{
  "message_id": "uuid",
  "correlation_id": "uuid",
  "request_id": "uuid-or-null",
  "event_name": "inference.requested",
  "device_id": "jetson-nano-line-a-01",
  "issued_at": "2026-06-18T12:00:00.000000Z",
  "payload": {}
}
```

Rules:

- `message_id` MUST be unique for each message.
- `correlation_id` MUST be reused across the full workflow.
- `request_id` SHOULD be reused for one inference job, preview session, or deployment request.
- `device_id` MUST match the topic device id.
- `event_name` MUST match the command or event semantic name.
- `issued_at` SHOULD be ISO 8601 UTC.
- `payload` MUST be a JSON object.

## Laravel to edge command topics

The edge device MUST subscribe to:

```text
drovenai/devices/{device_id}/commands/#
```

Implemented command topics:

| Topic | `event_name` | Meaning |
| --- | --- | --- |
| `drovenai/devices/{device_id}/commands/inference` | `inference.requested` | Run one inference job |
| `drovenai/devices/{device_id}/commands/firmware/update` | `firmware.update_requested` | Install the latest or a specific firmware version |
| `drovenai/devices/{device_id}/commands/preview/start` | `preview.start_requested` | Start a preview session |
| `drovenai/devices/{device_id}/commands/preview/signal` | `preview.webrtc_signal` | Deliver WebRTC offer, ICE candidate, close, or renegotiation data |
| `drovenai/devices/{device_id}/commands/preview/stop` | `preview.stop_requested` | Stop a preview session |
| `drovenai/devices/{device_id}/commands/model-deployment` | `model_deployment.requested` | Install or activate model components |

### `inference.requested`

Payload:

```json
{
  "request_id": "uuid",
  "correlation_id": "uuid",
  "source_type": "camera",
  "source_reference": "camera:0",
  "options": {
    "return_annotated_image": true,
    "return_depth_preview": false,
    "confidence_threshold": 0.5,
    "max_objects": 50
  }
}
```

Edge behavior:

1. Mark local inference worker busy.
2. Capture or load the requested source.
3. Run detector/material/depth/mass pipeline as available.
4. Publish an `inference.result` event.
5. Upload requested media with the same `request_id` and `correlation_id`.
6. Send telemetry with `inference_busy=false`.

### `firmware.update_requested`

Payload:

```json
{
  "request_id": "uuid",
  "correlation_id": "uuid",
  "target": "latest",
  "version": null,
  "current_version": "1.7.0",
  "requested_at": "2026-06-20T08:30:00Z"
}
```

When `target=specific`, `version` contains the exact requested version. The edge MUST validate package signatures and compatibility before installation, publish progress or failure events using the same correlation identifiers, and MUST NOT interpret the version as a shell command or package URL.

### `preview.start_requested`

Payload:

```json
{
  "request_id": "uuid",
  "correlation_id": "uuid",
  "mode": "low_fps",
  "camera_source": "camera:0",
  "ttl_seconds": 600
}
```

Edge behavior:

1. Reserve the requested camera source.
2. Publish `preview.ready` when it can accept WebRTC signaling, or `preview.failed` when it cannot.
3. Wait for WebRTC signaling on `commands/preview/signal`.
4. Publish WebRTC answers and ICE candidates through preview event topics.
5. Publish `preview.started` only after the peer connection is established and media is flowing.
6. Stop automatically when `ttl_seconds` expires unless renewed.

The complete WebRTC preview handshake is documented in `07-webrtc-preview-signaling.md`.

### `preview.webrtc_signal`

Topic:

```text
drovenai/devices/{device_id}/commands/preview/signal
```

This command carries WebRTC signaling data from Laravel/browser control plane to the edge device. MQTT is only the signaling channel. It MUST NOT carry preview video frames.

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

Allowed `signal_type` values:

- `offer`
- `ice_candidate`
- `renegotiate`
- `close`

Edge behavior:

1. Validate that `request_id` matches an active preview session.
2. Validate that the session has not expired.
3. Apply `offer` data to the WebRTC peer connection.
4. Publish `preview.webrtc_answer`.
5. Exchange ICE candidates through `preview.webrtc_ice_candidate`.
6. Close the peer connection when `signal_type=close`.

### `preview.stop_requested`

Payload:

```json
{
  "request_id": "uuid",
  "correlation_id": "uuid",
  "reason": "operator_requested"
}
```

Edge behavior:

1. Stop the preview stream for the matching `request_id`.
2. Release camera resources if no other workflow needs them.
3. Publish `preview.stopped`.

### `model_deployment.requested`

Payload:

```json
{
  "deployment_id": 1,
  "correlation_id": "uuid",
  "request_id": "model-deployment-1",
  "target_device_id": "jetson-nano-line-a-01",
  "target_type": "device",
  "strategy": "rolling",
  "restart_agent": true,
  "components": [
    {
      "task": "detector",
      "version": "v2.1.0",
      "source_uri": "github://drovenai/releases/waste-detector-v2.1.0",
      "artifact": {
        "id": 10,
        "source_type": "github_release",
        "checksum": "sha256"
      },
      "config": {
        "confidence_threshold": 0.55
      }
    }
  ]
}
```

Allowed component `task` values:

- `detector`
- `depth`
- `material`
- `mass`

Known artifact source values:

- `kaggle_model`
- `mlflow`
- `github_release`
- `local`
- `manual`

Edge behavior:

1. Validate the message target matches local `device_id`.
2. Download or locate each component.
3. Verify checksum when present.
4. Stage the new artifacts.
5. Activate according to strategy.
6. Restart the agent only when `restart_agent=true`.
7. Publish deployment progress and final status events.
8. Include active model versions in telemetry.

## Edge to Laravel event topics

The edge device SHOULD publish events to:

```text
drovenai/devices/{device_id}/events/{event_name}
```

The Laravel app consumes these topics through `php artisan mqtt:consume`. It validates the topic and envelope identity, records an MQTT trace, and dispatches supported events to domain actions.

Recommended edge event names:

| Topic suffix | `event_name` | Purpose | Backend target |
| --- | --- | --- | --- |
| `events/inference.stage` | `inference.stage` | Create or update one live execution stage | `ApplyInferenceStageAction` |
| `events/inference.result` | `inference.result` | Inference completed successfully | `StoreInferenceResultAction` |
| `events/inference.failed` | `inference.failed` | Inference failed | `FailInferenceJobAction` |
| `events/preview.ready` | `preview.ready` | Edge reserved camera and can accept WebRTC signaling | `ApplyPreviewEventAction` |
| `events/preview.webrtc_answer` | `preview.webrtc_answer` | Edge WebRTC SDP answer | `ApplyPreviewEventAction` |
| `events/preview.webrtc_ice_candidate` | `preview.webrtc_ice_candidate` | Edge ICE candidate | `ApplyPreviewEventAction` |
| `events/preview.started` | `preview.started` | Preview stream active | `ApplyPreviewEventAction` |
| `events/preview.stopped` | `preview.stopped` | Preview stream stopped | `ApplyPreviewEventAction` |
| `events/preview.failed` | `preview.failed` | Preview failed to start or continue | `ApplyPreviewEventAction` |
| `events/model_deployment.started` | `model_deployment.started` | Deployment started | `ApplyModelDeploymentEventAction` |
| `events/model_deployment.succeeded` | `model_deployment.succeeded` | Deployment succeeded | `ApplyModelDeploymentEventAction` |
| `events/model_deployment.failed` | `model_deployment.failed` | Deployment failed | `ApplyModelDeploymentEventAction` |
| `events/device.error` | `device.error` | Device-level error | Future error/audit handler |

Preview signaling events are detailed in `07-webrtc-preview-signaling.md`.

### `inference.stage`

Publish this event whenever the visible execution stage starts, advances, completes, or fails. Reuse the same `stage_key` to update an existing stage instead of creating duplicates.

Envelope payload:

```json
{
  "request_id": "uuid-from-inference-request",
  "correlation_id": "uuid-from-inference-request",
  "stage_key": "object-detection",
  "sequence": 2,
  "status": "running",
  "title": "Detecting objects",
  "description": "YOLO is detecting waste objects in the captured frame.",
  "progress_percent": 45,
  "started_at": "2026-06-20T10:30:12Z",
  "completed_at": null,
  "metadata": {
    "model": "yolo-waste-v1"
  }
}
```

Requirements:

- `request_id` or `correlation_id` MUST identify the matching inference request.
- `stage_key` SHOULD be stable for updates to the same stage.
- `sequence` controls display order and SHOULD start at `1`.
- `status` MUST be `pending`, `running`, `completed`, or `failed`.
- `title` is required and `description` SHOULD explain the current Jetson operation in operator-friendly language.
- `progress_percent`, when present, MUST be an integer from `0` through `100`.
- The edge SHOULD publish the final `inference.result` or `inference.failed` only after its last stage update.

### `inference.result`

Envelope payload:

```json
{
  "request_id": "uuid",
  "correlation_id": "uuid",
  "latency_ms": 142,
  "fps": 8.75,
  "object_count": 2,
  "total_estimated_mass_grams": 328.25,
  "dominant_material": "plastic",
  "model_versions": {
    "detector": "yolo-waste-v1",
    "mass": "mass-v1"
  },
  "objects": [
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
      "features": {
        "notes": "optional"
      }
    }
  ]
}
```

Object field requirements:

- `class_label` is required for each object.
- `material_label`, `confidence`, `bbox`, `estimated_geometry`, `estimated_mass_grams`, `mass_method`, and `features` are optional.

The backend computes `object_count` from `objects` when it is omitted.

## MQTT telemetry topic

Topic:

```text
drovenai/devices/{device_id}/telemetry
```

MQTT telemetry uses the standard envelope with `event_name=telemetry.reported`. Its inner `payload` mirrors the HTTP telemetry body. The MQTT consumer dispatches it through `UpdateDeviceTelemetryAction`, so HTTP and MQTT update the same device state.

## Python MQTT example

```python
import json
import time
import uuid
import paho.mqtt.client as mqtt

prefix = "drovenai"
device_id = "jetson-nano-line-a-01"
broker_host = "127.0.0.1"
broker_port = 1883

def topic(name: str) -> str:
    return f"{prefix}/devices/{device_id}/{name}"

def envelope(event_name: str, payload: dict, correlation_id: str | None = None, request_id: str | None = None) -> dict:
    return {
        "message_id": str(uuid.uuid4()),
        "correlation_id": correlation_id or str(uuid.uuid4()),
        "request_id": request_id,
        "event_name": event_name,
        "device_id": device_id,
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "payload": payload,
    }

def on_message(client, userdata, message):
    command = json.loads(message.payload.decode("utf-8"))
    event_name = command["event_name"]
    payload = command["payload"]
    print("received", message.topic, event_name, payload)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{device_id}-agent")
client.on_message = on_message
client.connect(broker_host, broker_port, keepalive=30)
client.subscribe(topic("commands/#"))

result = envelope(
    "inference.result",
    {
        "request_id": "request-uuid",
        "correlation_id": "correlation-uuid",
        "latency_ms": 142,
        "objects": [],
    },
    correlation_id="correlation-uuid",
    request_id="request-uuid",
)
client.publish(topic("events/inference.result"), json.dumps(result), qos=0)
client.loop_forever()
```
