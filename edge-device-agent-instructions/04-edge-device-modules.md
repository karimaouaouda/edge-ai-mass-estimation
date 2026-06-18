# Required Edge Device Modules

An edge device agent should be modular. The Laravel backend is a supervision layer; the Jetson owns capture, inference, model lifecycle, preview generation, and local reliability.

## Required modules

| Module | Required | Responsibility |
| --- | --- | --- |
| Configuration | Yes | Load device id, backend URL, MQTT broker, topic prefix, TLS settings, retry policy, and feature flags |
| Secret store | Yes | Store `X-Device-Token`, MQTT username/password, certificates, and signing keys |
| Device identity | Yes | Expose immutable `device_id`, hardware metadata, firmware version, agent version, and capabilities |
| HTTP API client | Yes | Submit telemetry and upload media with `X-Device-Token` |
| MQTT client | Yes | Subscribe to commands, publish events, handle reconnects, and apply per-device topic ACL assumptions |
| Envelope/correlation | Yes | Parse, validate, build, and persist message envelopes |
| Local queue/outbox | Yes | Buffer telemetry, media upload jobs, and MQTT events during network loss |
| Telemetry sampler | Yes | Collect CPU, memory, temperature, disk, camera status, inference busy flag, and active models |
| Camera/capture adapter | Yes | Capture frames from configured `camera_source` values such as `camera:0` |
| Inference runner | Yes | Run detector/material/depth/mass models and return normalized result objects |
| Media renderer/uploader | Yes | Produce original, annotated, depth, mask, thumbnail, and uploaded input files as needed |
| Model manager | Yes | Download, verify, stage, activate, retire, and report model artifacts |
| Preview manager | Yes | Start and stop low-FPS or diagnostic preview streams |
| WebRTC signaling | Yes | Exchange preview SDP and ICE data over MQTT and bind it to an active preview session |
| WebRTC media peer | Yes | Create the peer connection and send preview video over WebRTC, not MQTT |
| Error reporter | Yes | Publish `device.error` events and include failure summaries in logs |
| Observability logger | Yes | Structured logs with redacted secrets and workflow ids |
| Time sync checker | Yes | Verify system clock drift is acceptable for issued timestamps |

## Configuration keys

The agent SHOULD support these config keys:

```yaml
device:
  id: "jetson-nano-line-a-01"
  name: "Line A Jetson 01"
  type: "jetson_nano"
  firmware_version: "1.0.0"
  agent_version: "0.1.0"
  capabilities:
    inference: true
    preview: true
    model_update: true

backend:
  base_url: "http://localhost:8000"
  device_token_env: "DROVENAI_DEVICE_TOKEN"
  request_timeout_seconds: 10

mqtt:
  host: "127.0.0.1"
  port: 1883
  topic_prefix: "drovenai"
  username_env: "DROVENAI_MQTT_USERNAME"
  password_env: "DROVENAI_MQTT_PASSWORD"
  tls: false
  keepalive_seconds: 30

runtime:
  telemetry_interval_seconds: 30
  media_upload_retries: 5
  outbox_path: "/var/lib/drovenai-agent/outbox"
  model_dir: "/var/lib/drovenai-agent/models"
```

## Envelope validation rules

For inbound command envelopes, the agent MUST validate:

- Payload is valid JSON.
- Envelope is a JSON object.
- `device_id` equals local device id.
- `event_name` is expected for the subscribed topic.
- `message_id` has not already been processed recently.
- `payload` is a JSON object.
- Required payload fields exist for the command.

On validation failure, the agent SHOULD publish a `device.error` event with:

```json
{
  "error_type": "invalid_command",
  "summary": "Command envelope failed validation",
  "source_message_id": "uuid",
  "details": {
    "field": "device_id"
  }
}
```

## Reliability rules

The agent MUST:

- Retry HTTP requests with bounded exponential backoff.
- Persist unsent telemetry/media/event records to a local outbox.
- Avoid duplicate media uploads for the same local file hash and workflow id.
- Treat MQTT QoS 0 commands as at-most-once; use `message_id` dedupe to avoid double execution after reconnects.
- Report failures with `request_id` and `correlation_id` when available.
- Continue telemetry even if inference is failing.
- Refuse model deployment commands targeted at a different `device_id`.

## Startup sequence

1. Load config.
2. Load secrets.
3. Validate `device_id` exists and is stable.
4. Start structured logging.
5. Check clock synchronization.
6. Start local outbox.
7. Connect MQTT.
8. Subscribe to `drovenai/devices/{device_id}/commands/#`.
9. Send initial HTTP telemetry.
10. Start periodic telemetry.
11. Start command loop.

## Shutdown sequence

1. Stop accepting new command work.
2. Publish or enqueue a final telemetry payload with `status=offline` when possible.
3. Stop preview streams.
4. Flush outbox within a bounded timeout.
5. Disconnect MQTT.
6. Close camera/model resources.
