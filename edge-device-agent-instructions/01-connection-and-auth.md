# Connection And Auth

## Identity

Each edge device is identified by `device_id`.

Examples:

- `jetson-nano-line-a-01`
- `north-jetson-01`
- `simulated-agent-01`

The backend route model key is `device_id`, not the numeric database id. All edge-facing paths and topics use this value.

## Provisioning flow

1. An admin registers a device in the Laravel dashboard at `/devices/create`.
2. The backend creates an `edge_devices` row.
3. If credential generation is enabled, the backend creates a `device_credentials` row with a hashed provisioning token.
4. The raw provisioning token is shown once to the operator.
5. The IoT engineer installs that raw token into the device secret store.
6. The device uses that token in HTTP requests as `X-Device-Token`.

The raw token is not recoverable from the backend after registration. If it is lost, rotate or recreate credentials.

## HTTP authentication

Every current edge HTTP endpoint requires this header:

```http
X-Device-Token: <raw provisioning token>
```

Recommended headers:

```http
Accept: application/json
X-Device-Token: <raw provisioning token>
```

Behavior:

- Missing token returns `403 Forbidden`.
- Invalid token returns `403 Forbidden`.
- Unknown `device_id` returns `404 Not Found`.
- Invalid payload returns `422 Unprocessable Entity`.
- API throttling is `60` requests per minute per throttle key on the route group.

The backend checks the token against active hashed credentials for the route device.

## MQTT authentication

Local development Mosquitto currently allows anonymous clients:

```conf
listener 1883 0.0.0.0
allow_anonymous true
```

Production MUST NOT use anonymous MQTT.

Production MQTT credentials should be per-device:

- `mqtt_username`
- `mqtt_password`
- optional TLS client certificate later

The database already has credential fields for this:

- `device_credentials.mqtt_username`
- `device_credentials.mqtt_password_secret`

The current admin registration flow generates the HTTP provisioning token only. Until MQTT credential rotation is implemented in the UI, the IoT engineer should provision broker credentials out-of-band for production tests.

## MQTT ACL target

For device `{device_id}` and topic prefix `{prefix}`, usually `drovenai`:

The edge device SHOULD be allowed to subscribe to:

```text
{prefix}/devices/{device_id}/commands/#
```

The edge device SHOULD be allowed to publish to:

```text
{prefix}/devices/{device_id}/events/#
{prefix}/devices/{device_id}/telemetry
```

The edge device MUST NOT publish to another device's topic.

The Laravel server SHOULD be allowed to publish to all command topics and subscribe to all event and telemetry topics.

## Transport security

Local development may use plain HTTP and plain MQTT on trusted machines.

Production requirements:

- HTTP MUST use HTTPS.
- MQTT SHOULD use TLS.
- Secrets MUST be stored in a device secret store, not in source code.
- Logs MUST redact `X-Device-Token`, MQTT passwords, bearer tokens, and signed URLs.
- The device clock SHOULD be synchronized with NTP because message envelopes include timestamps.

## Message correlation

Every command and response flow MUST preserve:

- `message_id`: unique id for the current message envelope
- `correlation_id`: shared id across a workflow
- `request_id`: id for the specific request or job when available
- `device_id`: route/topic device id
- `event_name`: semantic event name

When a device receives a command, it MUST copy the command `correlation_id` and `request_id` into related events, telemetry metadata, and media uploads.

