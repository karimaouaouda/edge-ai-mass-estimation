# Jetson Orchestrator

The orchestrator is a long-running update process for Jetson Nano deployments. It can poll GitHub Releases on an interval, react to Mosquitto/MQTT update messages, or run both modes at the same time.

## Run

```bash
pip install -e ".[orchestration]"
edge-ai-orchestrator --config configs/orchestration/jetson_nano.yaml
```

For a one-shot update check:

```bash
edge-ai-orchestrator --config configs/orchestration/jetson_nano.yaml --once
```

The backend-facing agent can also run the same updater automatically. In
`configs/agent/jetson_nano.yaml`, the `updates` block is enabled by default on
Jetson:

```yaml
updates:
  enabled: true
  config_path: configs/orchestration/jetson_nano.yaml
  run_on_start: true
  poll_interval_seconds: 21600
```

When enabled, the agent checks GitHub Releases before inference preloading and
then repeats the check on the configured interval. Installed model or config
updates clear the loaded inference runner so the next inference uses the new
artifacts.

## Update Sources

GitHub Releases should publish an `edge-ai-update-manifest.json` asset. If that manifest is missing, the orchestrator falls back to the `install_plan` in `configs/orchestration/jetson_nano.yaml`.

For the full release structure, packaging steps, manifest schema, and updater flow, see `docs/github_releases_deployment.md`.

Example manifest:

```json
{
  "version": "v0.2.0",
  "channel": "stable",
  "assets": [
    {
      "name": "detection-tensorrt-engine",
      "target": "model",
      "asset": "yolov8n-seg.engine",
      "destination": "models/weights/yolov8n-seg.engine",
      "sha256": "PUT_SHA256_HERE",
      "restart_app": true
    },
    {
      "name": "application-wheel",
      "target": "program",
      "asset": "edge_ai_mass.whl",
      "install_command": ["python3", "-m", "pip", "install", "--no-deps", "--upgrade", "{asset_path}"],
      "restart_app": true
    }
  ]
}
```

Supported targets include `model`, `engine`, `config`, and `program`. File assets are installed atomically with backups under `.edge_ai_mass/orchestrator/backups`.

## MQTT Commands

Publish JSON to `edge-ai-mass/<device-id>/updates` or `edge-ai-mass/all/updates`.

```json
{"action": "check"}
{"action": "update", "target": "model", "force": true}
{"action": "update", "release": "v0.2.0", "target": "program"}
{"action": "rollback", "target": "model"}
{"action": "restart_app"}
```

Status is published to `edge-ai-mass/<device-id>/status`.

## Recommended Additions

For production releases, add SHA-256 values to every manifest asset, publish TensorRT engines separately from PyTorch weights, and keep a small health check command that can validate imports or run a quick dry inference after updates.
