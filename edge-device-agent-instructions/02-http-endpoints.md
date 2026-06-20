# HTTP Endpoints

Base URL for local development:

```text
http://localhost:8000
```

API prefix:

```text
/api/edge
```

All endpoints require `X-Device-Token`.

## Endpoint summary

| Method | Path | Purpose | Auth |
| --- | --- | --- | --- |
| POST | `/api/edge/devices/{device_id}/telemetry` | Submit device heartbeat and resource state | `X-Device-Token` |
| POST | `/api/edge/devices/{device_id}/media` | Upload original, annotated, depth, mask, thumbnail, or input images | `X-Device-Token` |

There is no current edge self-registration endpoint. Device records and tokens are created by an authenticated operator in the Laravel dashboard.

There is no HTTP endpoint for inference results. Devices publish results over MQTT, and the backend's long-running consumer validates the envelope, stores the trace, and dispatches it to `StoreInferenceResultAction`.

## POST telemetry

Path:

```text
POST /api/edge/devices/{device_id}/telemetry
```

Headers:

```http
Accept: application/json
Content-Type: application/json
X-Device-Token: <raw provisioning token>
```

Request JSON:

```json
{
  "status": "online",
  "health_status": "healthy",
  "cpu_percent": 35.7,
  "memory_percent": 62.4,
  "temperature_celsius": 54.3,
  "disk_percent": 70.1,
  "camera_connected": true,
  "inference_busy": false,
  "active_models": {
    "detector": "yolo-waste-v1",
    "depth": "depth-v1",
    "material": "material-v1",
    "mass": "mass-v1"
  },
  "reported_at": "2026-06-18T12:00:00Z"
}
```

Allowed `status` values:

- `provisioning`
- `online`
- `offline`
- `maintenance`
- `disabled`

Allowed `health_status` values:

- `unknown`
- `healthy`
- `warning`
- `critical`

Validation rules:

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| `status` | No | enum | Defaults to `online` if omitted by backend action |
| `health_status` | No | enum | Defaults to `healthy` if omitted by backend action |
| `cpu_percent` | No | number | 0 to 100 |
| `memory_percent` | No | number | 0 to 100 |
| `temperature_celsius` | No | number | -40 to 125 |
| `disk_percent` | No | number | 0 to 100 |
| `camera_connected` | No | boolean | Defaults to false |
| `inference_busy` | No | boolean | Defaults to false |
| `active_models` | No | object | Model versions by task |
| `reported_at` | No | date | ISO 8601 recommended |

Success response:

```json
{
  "telemetry": {
    "id": 1,
    "device_id": "jetson-telemetry-01",
    "status": "online",
    "reported_at": "2026-06-18T12:00:00.000000Z"
  }
}
```

Side effects:

- Creates `device_telemetry`.
- Updates or creates `device_status_snapshots`.
- Updates `edge_devices.status`.
- Updates `edge_devices.health_status`.
- Updates `edge_devices.last_seen_at`.
- Updates `edge_devices.active_model_versions` when `active_models` is provided.

## POST media upload

Path:

```text
POST /api/edge/devices/{device_id}/media
```

Headers:

```http
Accept: application/json
X-Device-Token: <raw provisioning token>
```

Content type:

```http
multipart/form-data
```

Multipart fields:

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| `file` | Yes | image file | Max 10 MB |
| `media_type` | Yes | enum | See values below |
| `request_id` | No | string | Max 100 chars |
| `correlation_id` | No | string | Max 100 chars |
| `sha256` | No | string | 64 hex characters; backend verifies file hash |
| `metadata` | No | object | Extra JSON-style multipart fields |

Allowed `media_type` values:

- `original`
- `annotated`
- `depth_preview`
- `mask`
- `thumbnail`
- `uploaded_input`

Success response:

```json
{
  "media_file": {
    "id": 1,
    "media_type": "annotated",
    "disk": "public",
    "path": "edge-media/jetson-upload-01/example.jpg",
    "sha256": "64-character-sha256"
  }
}
```

Side effects:

- Stores the file under `edge-media/{device_id}` on the configured disk.
- Creates `media_files`.
- Links to the newest matching inference job/result when `request_id` or `correlation_id` matches.
- Verifies `sha256` when provided.

## Python HTTP examples

Telemetry:

```python
import requests

base_url = "http://localhost:8000"
device_id = "jetson-telemetry-01"
device_token = "<provisioning-token>"

response = requests.post(
    f"{base_url}/api/edge/devices/{device_id}/telemetry",
    headers={
        "Accept": "application/json",
        "X-Device-Token": device_token,
    },
    json={
        "status": "online",
        "health_status": "healthy",
        "cpu_percent": 35.7,
        "memory_percent": 62.4,
        "temperature_celsius": 54.3,
        "disk_percent": 70.1,
        "camera_connected": True,
        "inference_busy": False,
        "active_models": {"detector": "yolo-waste-v1"},
    },
    timeout=10,
)
response.raise_for_status()
print(response.json())
```

Media upload:

```python
from pathlib import Path
import hashlib
import requests

base_url = "http://localhost:8000"
device_id = "jetson-upload-01"
device_token = "<provisioning-token>"
path = Path("annotated.jpg")
sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

with path.open("rb") as file_handle:
    response = requests.post(
        f"{base_url}/api/edge/devices/{device_id}/media",
        headers={
            "Accept": "application/json",
            "X-Device-Token": device_token,
        },
        data={
            "request_id": "request-uuid",
            "correlation_id": "correlation-uuid",
            "media_type": "annotated",
            "sha256": sha256,
        },
        files={
            "file": ("annotated.jpg", file_handle, "image/jpeg"),
        },
        timeout=60,
    )
response.raise_for_status()
print(response.json())
```
