# Laravel MQTT Message Flow

This document is the backend operator and integration-agent guide for how this Laravel application sends and receives MQTT messages.

For complete payload schemas, also read `03-mqtt-topics-and-events.md`. For preview signaling, read `07-webrtc-preview-signaling.md`.

## Runtime architecture

The MQTT module separates domain workflows from broker transport:

```text
Laravel action
  -> MqttPublisher interface
      -> FakeMqttPublisher       (database trace only)
      -> BrokerMqttPublisher     (Mosquitto publish + database trace)

Mosquitto subscription
  -> php artisan mqtt:consume
  -> MqttBrokerClient
  -> ProcessInboundMqttMessageAction
  -> domain action
  -> mqtt_messages trace
```

`MQTT_CONNECTION` selects the outbound publisher:

```dotenv
MQTT_CONNECTION=fake
```

Use `fake` for automated tests and local UI work without a broker. It writes the outgoing message to `mqtt_messages` but does not open a network connection.

```dotenv
MQTT_CONNECTION=mosquitto
```

Use `mosquitto` when Laravel must publish command envelopes to the broker. The Docker Compose Laravel service uses this mode by default.

## Broker configuration

```dotenv
MQTT_HOST=mosquitto
MQTT_PORT=1883
MQTT_CLIENT_ID=drovenai-master-backend
MQTT_CONSUMER_CLIENT_ID=drovenai-master-backend-consumer
MQTT_TOPIC_PREFIX=drovenai
MQTT_USERNAME=null
MQTT_PASSWORD=null
MQTT_USE_TLS=false
MQTT_TIMEOUT=5
```

From the host machine, use `127.0.0.1` when port `1883` is forwarded. From a Sail container, use the Docker service name `mosquitto`.

The current Docker Mosquitto configuration allows anonymous clients for local development. The Laravel broker client already supports username/password and TLS through the variables above, but production broker users and ACLs must be provisioned in Mosquitto infrastructure.

Production ACL intent:

```text
Edge device subscribe: {prefix}/devices/{device_id}/commands/#
Edge device publish:   {prefix}/devices/{device_id}/events/#
Edge device publish:   {prefix}/devices/{device_id}/telemetry
Laravel publish:       {prefix}/devices/+/commands/#
Laravel subscribe:     {prefix}/devices/+/events/#
Laravel subscribe:     {prefix}/devices/+/telemetry
```

The HTTP `X-Device-Token` is not an MQTT credential. MQTT authentication is enforced by the broker, while the Laravel consumer additionally verifies that topic `device_id` equals envelope `device_id`.

## Standard envelope

Every command, event, and telemetry message must be a JSON object using this envelope:

```json
{
  "message_id": "f62a9cf6-54fd-42e8-8675-7409f7e5dc70",
  "correlation_id": "da8261a8-3567-4cae-9f76-f728f660b920",
  "request_id": "job-or-session-id-or-null",
  "event_name": "telemetry.reported",
  "device_id": "jetson-nano-line-a-01",
  "issued_at": "2026-06-19T12:00:00Z",
  "payload": {}
}
```

Backend rules:

- `message_id`, `correlation_id`, `event_name`, and `device_id` are required non-empty strings.
- `payload` must be a JSON object.
- The topic device and envelope device must match.
- Event-topic suffix and `event_name` must match.
- Telemetry must use `event_name=telemetry.reported`.
- Duplicate `message_id` values return the existing trace and do not repeat domain writes.
- `request_id` and `correlation_id` are copied into the application payload before domain dispatch.

## Laravel to edge topics

Laravel publishes command envelopes to these topics:

| Topic | Envelope `event_name` | Publisher |
| --- | --- | --- |
| `{prefix}/devices/{device_id}/commands/inference` | `inference.requested` | `PublishInferenceCommandAction` |
| `{prefix}/devices/{device_id}/commands/firmware/update` | `firmware.update_requested` | `PublishFirmwareUpdateAction` |
| `{prefix}/devices/{device_id}/commands/preview/start` | `preview.start_requested` | `StartPreviewSessionAction` |
| `{prefix}/devices/{device_id}/commands/preview/signal` | `preview.webrtc_signal` | `PublishPreviewSignalAction` |
| `{prefix}/devices/{device_id}/commands/preview/stop` | `preview.stop_requested` | `StopPreviewSessionAction` |
| `{prefix}/devices/{device_id}/commands/model-deployment` | `model_deployment.requested` | `PublishModelDeploymentCommandAction` |

The domain action passes only the application payload to `MqttPublisher`. `MessageEnvelopeBuilder` adds message identity, device identity, correlation, and timestamp exactly once.

For preview, `preview.ready` is the trigger for the browser on the device detail page to create a receive-only WebRTC offer. The browser sends that offer and its trickle ICE candidates through the Livewire `publishPreviewSignal` method, which publishes them on `commands/preview/signal`.

When using the Mosquitto transport, the backend first records a `pending` outgoing trace, publishes at QoS 0, and then marks that trace `published`. A transport error marks the same trace `failed` and rethrows the error to the caller.

## Edge to Laravel topics

The consumer subscribes to:

```text
{prefix}/devices/+/events/#
{prefix}/devices/+/telemetry
```

Supported events are dispatched as follows:

| Topic/event | Backend behavior |
| --- | --- |
| `telemetry` / `telemetry.reported` | Creates telemetry, updates the status snapshot, and updates device health/last seen |
| `events/inference.stage` | Upserts an ordered live stage and marks the matching inference job running |
| `events/inference.result` | Stores the result and detected objects, completes the matching inference job |
| `events/inference.failed` | Marks the matching inference job failed |
| `events/preview.ready` | Marks the session ready for a browser offer |
| `events/preview.webrtc_answer` | Stores the edge SDP answer and advances signaling state |
| `events/preview.webrtc_ice_candidate` | Stores the latest edge ICE candidate |
| `events/preview.started` | Marks the preview session active |
| `events/preview.stopped` | Marks the preview session stopped |
| `events/preview.failed` | Marks the preview session failed and stores failure metadata |
| `events/model_deployment.started` | Records per-device progress and marks the deployment running |
| `events/model_deployment.succeeded` | Records success; completes a device-target deployment |
| `events/model_deployment.failed` | Records failure; fails a device-target deployment |

Other valid event topics, such as `device.error`, are stored with status `received` but are not marked `processed` until a domain handler exists.

Invalid JSON, unsupported topic shapes, identity mismatches, and envelope mismatches are rejected before domain processing. If a domain handler fails after the trace is created, that trace is marked `failed` with the error message.

## Start the inbound consumer

The consumer is a long-running process and must run alongside the web server and queue worker. Docker Compose starts it as the dedicated `mqtt.consumer` service with a client ID separate from the web publisher.

Inside Sail:

```shell
./vendor/bin/sail artisan mqtt:consume
```

From the host:

```shell
php artisan mqtt:consume --host=127.0.0.1 --port=1883
```

Useful controlled runs:

```shell
php artisan mqtt:consume --once
php artisan mqtt:consume --max-messages=10
php artisan mqtt:consume --run-seconds=30
```

Authenticated/TLS example:

```shell
php artisan mqtt:consume \
  --host=mqtt.example.com \
  --port=8883 \
  --username=drovenai-backend \
  --password=secret \
  --tls
```

Run this command under Supervisor, systemd, Kubernetes, or another process manager in production. For local development, `docker compose up -d` starts the dedicated consumer automatically; use the manual commands above only for debugging or when running outside Compose.

## Send and exchange diagnostics

Record an outbound event without broker I/O:

```shell
php artisan mqtt:send-event jetson-nano-line-a-01 diagnostic.ping \
  --connection=fake \
  --payload='{"ok":true}'
```

Publish a diagnostic event to Mosquitto:

```shell
php artisan mqtt:send-event jetson-nano-line-a-01 diagnostic.ping \
  --connection=mosquitto \
  --host=127.0.0.1 \
  --payload='{"ok":true}'
```

Test broker publish/subscribe round trip on one event topic:

```shell
php artisan mqtt:test-exchange jetson-nano-line-a-01 diagnostic.ping \
  --host=127.0.0.1 \
  --payload='{"roundtrip":true}'
```

`mqtt:test-exchange` verifies broker transport and records one outbound and one inbound trace. It is not a replacement for the long-running consumer.

`mqtt:record-event` creates a simulated inbound trace for database/UI testing. It does not run the production domain dispatcher.

## Edge telemetry example

Topic:

```text
drovenai/devices/jetson-nano-line-a-01/telemetry
```

Message:

```json
{
  "message_id": "434424e4-29d1-4a9f-88c9-d63c52f6a48f",
  "correlation_id": "e317f93c-b60d-4ea8-8c8e-805ba7cbfd35",
  "request_id": null,
  "event_name": "telemetry.reported",
  "device_id": "jetson-nano-line-a-01",
  "issued_at": "2026-06-19T12:00:00Z",
  "payload": {
    "status": "online",
    "health_status": "healthy",
    "cpu_percent": 42.5,
    "memory_percent": 61.2,
    "temperature_celsius": 51.2,
    "disk_percent": 37.8,
    "camera_connected": true,
    "inference_busy": false,
    "active_models": {
      "detector": "yolo-waste-v1"
    },
    "reported_at": "2026-06-19T12:00:00Z"
  }
}
```

## Message trace statuses

`mqtt_messages.status` communicates transport/processing state:

| Status | Meaning |
| --- | --- |
| `published` | Outbound message was recorded or published successfully |
| `received` | Inbound envelope was accepted but has no domain handler |
| `processed` | Inbound envelope completed its domain handler |
| `failed` | Transport publishing or inbound domain processing failed |

The trace stores topic, direction, message type, QoS, envelope, application payload, timestamps, and errors. Passwords and device tokens are never added to message headers.

## Source files

- `config/mqtt.php`
- `Modules/Mqtt/app/Interfaces/MqttPublisher.php`
- `Modules/Mqtt/app/Classes/MqttBrokerClient.php`
- `Modules/Mqtt/app/Classes/BrokerMqttPublisher.php`
- `Modules/Mqtt/app/Classes/FakeMqttPublisher.php`
- `Modules/Mqtt/app/Classes/TopicBuilder.php`
- `Modules/Mqtt/app/Classes/TopicParser.php`
- `Modules/Mqtt/app/Actions/ProcessInboundMqttMessageAction.php`
- `app/Console/Commands/MqttConsume.php`
- `tests/Feature/MqttInboundProcessingTest.php`
