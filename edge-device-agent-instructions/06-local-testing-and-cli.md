# Local Testing And CLI

Use this file to test the connection and message exchange during development.

## Start the Laravel stack

From the Laravel project root:

```bash
./vendor/bin/sail up -d
```

On Windows PowerShell, if Sail shell aliases are not configured, use:

```powershell
docker compose up -d
```

Local services:

| Service | Host from host machine | Host from Laravel container |
| --- | --- | --- |
| Laravel HTTP | `http://localhost:8000` | `http://laravel.test` |
| Mosquitto MQTT | `127.0.0.1:1883` | `mosquitto:1883` |

## Create a test device

The normal path is the authenticated dashboard:

```text
http://localhost:8000/devices/create
```

Store the generated provisioning token securely. It is shown once.

## Test HTTP telemetry with curl

```bash
curl -i \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -H "X-Device-Token: <provisioning-token>" \
  -d '{"status":"online","health_status":"healthy","cpu_percent":35.7,"memory_percent":62.4,"temperature_celsius":54.3,"disk_percent":70.1,"camera_connected":true,"inference_busy":false,"active_models":{"detector":"yolo-waste-v1"}}' \
  http://localhost:8000/api/edge/devices/jetson-telemetry-01/telemetry
```

Expected success:

```text
HTTP/1.1 201 Created
```

Expected invalid token:

```text
HTTP/1.1 403 Forbidden
```

## Test HTTP media upload with curl

```bash
SHA256="$(sha256sum annotated.jpg | awk '{print $1}')"

curl -i \
  -H "Accept: application/json" \
  -H "X-Device-Token: <provisioning-token>" \
  -F "request_id=request-uuid" \
  -F "correlation_id=correlation-uuid" \
  -F "media_type=annotated" \
  -F "sha256=${SHA256}" \
  -F "file=@annotated.jpg" \
  http://localhost:8000/api/edge/devices/jetson-upload-01/media
```

Expected success:

```text
HTTP/1.1 201 Created
```

## Laravel MQTT CLI commands

The Laravel app includes CLI commands for testing MQTT traces and broker exchange.

List commands:

```bash
php artisan list mqtt --no-interaction
```

Record a fake outbound event in `mqtt_messages`:

```bash
php artisan mqtt:send-event jetson-cli-01 diagnostic.ping --connection=fake --payload='{"ok":true}'
```

Publish an event to Mosquitto:

```bash
php artisan mqtt:send-event jetson-cli-01 diagnostic.ping --connection=mosquitto --host=127.0.0.1 --port=1883 --payload='{"ok":true}'
```

Subscribe, publish, and wait for the same event through Mosquitto:

```bash
php artisan mqtt:test-exchange jetson-cli-01 diagnostic.ping --host=127.0.0.1 --port=1883 --payload='{"roundtrip":true}'
```

Record a simulated inbound event:

```bash
php artisan mqtt:record-event jetson-cli-01 diagnostic.ack --processed --payload='{"ack":true}'
```

## Mosquitto CLI smoke test

Terminal 1:

```bash
mosquitto_sub -h 127.0.0.1 -p 1883 -t 'drovenai/devices/jetson-cli-01/events/#' -v
```

Terminal 2:

```bash
mosquitto_pub -h 127.0.0.1 -p 1883 \
  -t 'drovenai/devices/jetson-cli-01/events/diagnostic.ping' \
  -m '{"message_id":"manual-1","correlation_id":"manual-correlation","request_id":null,"event_name":"diagnostic.ping","device_id":"jetson-cli-01","issued_at":"2026-06-18T12:00:00Z","payload":{"ok":true}}'
```

## Backend verification commands

Run focused edge ingestion tests:

```bash
php artisan test --compact tests/Feature/EdgeAiIngestionEndpointsTest.php
```

Run workflow action tests:

```bash
php artisan test --compact tests/Feature/EdgeAiWorkflowActionsTest.php
php artisan test --compact tests/Feature/EdgeAiDeploymentPreviewActionsTest.php
```

Run MQTT CLI tests:

```bash
php artisan test --compact tests/Feature/MqttCliCommandsTest.php
```

## Troubleshooting

### `403 Forbidden` from HTTP API

Check:

- `X-Device-Token` is present.
- Token belongs to the same `{device_id}` in the URL.
- Credential is active.
- You are using the raw token, not the `secret_hint`.

### `404 Not Found` from HTTP API

Check:

- `{device_id}` exists in `edge_devices`.
- You used the device route key, not the numeric id.

### `422 Unprocessable Entity` from media upload

Check:

- `file` is a valid image.
- File size is under 10 MB.
- `media_type` is one of the allowed values.
- `sha256` matches the uploaded file.

### MQTT command not received

Check:

- Broker host and port.
- The edge agent subscribed to `drovenai/devices/{device_id}/commands/#`.
- The backend and device use the same topic prefix.
- The topic uses the exact `device_id`.
- Production ACLs allow subscribe/publish for the device.

### Message cannot be correlated

Check:

- Edge copied `correlation_id` from the command envelope.
- Edge copied `request_id` from the command payload or envelope.
- Media uploads include the same `request_id` or `correlation_id`.
