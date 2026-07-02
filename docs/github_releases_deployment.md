# GitHub Releases Deployment Guide

This document explains how to package GitHub Releases for the Jetson updater, what each release should contain, and how the orchestrator applies a release on the device.

## Release Contract

Each GitHub Release is treated as an update bundle. The updater does not inspect the release body for install instructions; it only uses release metadata and uploaded assets.

Required production structure:

```text
GitHub Release: v0.2.0
Assets:
  edge-ai-update-manifest.json
  yolov8n-seg.engine
  yolo-seg-best.pt
  mass_residual_best_model.joblib
  jetson_nano.yaml
  agent_jetson_nano.yaml
  orchestration_jetson_nano.yaml
  edge_ai_mass-0.2.0-py3-none-any.whl
```

The asset names must match the names in `edge-ai-update-manifest.json` exactly. GitHub stores release assets as a flat list, so the structure above is conceptual; do not rely on folders inside the GitHub Release page unless the asset itself is an archive.

When a model rule uses `source: "kaggle_model"`, the model file is not a GitHub Release asset. In that case, the GitHub Release only needs the manifest plus any other GitHub-hosted assets such as configs or wheels.

## Manifest

The default manifest asset name is configured in `configs/orchestration/jetson_nano.yaml`:

```yaml
updates:
  github:
    manifest_asset: edge-ai-update-manifest.json
    channel: stable
```

The manifest should be uploaded as a release asset named `edge-ai-update-manifest.json`.

Example:

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
      "restart_app": true,
      "required": true
    },
    {
      "name": "mass-residual-model",
      "target": "model",
      "asset": "mass_residual_best_model.joblib",
      "destination": "models/weights/mass_residual_best_model.joblib",
      "sha256": "PUT_SHA256_HERE",
      "restart_app": true,
      "required": false
    },
    {
      "name": "jetson-pipeline-config",
      "target": "config",
      "asset": "jetson_nano.yaml",
      "destination": "configs/pipeline/jetson_nano.yaml",
      "sha256": "PUT_SHA256_HERE",
      "restart_app": true,
      "required": true
    },
    {
      "name": "jetson-agent-config",
      "target": "config",
      "asset": "agent_jetson_nano.yaml",
      "destination": "configs/agent/jetson_nano.yaml",
      "sha256": "PUT_SHA256_HERE",
      "restart_app": true,
      "required": true
    },
    {
      "name": "jetson-orchestrator-config",
      "target": "config",
      "asset": "orchestration_jetson_nano.yaml",
      "destination": "configs/orchestration/jetson_nano.yaml",
      "sha256": "PUT_SHA256_HERE",
      "restart_app": true,
      "required": true
    },
    {
      "name": "kaggle-trained-detector",
      "source": "kaggle_model",
      "target": "model",
      "asset": "best.pt",
      "destination": "models/weights/yolo-seg-best.pt",
      "sha256": "PUT_SHA256_HERE_IF_AVAILABLE",
      "kaggle_model": "owner/waste-detector/pyTorch/yolo-seg/3",
      "kaggle_model_file": "weights/best.pt",
      "restart_app": true,
      "required": true
    },
    {
      "name": "application-wheel",
      "target": "program",
      "asset": "edge_ai_mass-0.2.0-py3-none-any.whl",
      "sha256": "PUT_SHA256_HERE",
      "install_command": [
        "python3",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--upgrade",
        "{asset_path}"
      ],
      "restart_app": true,
      "required": false
    }
  ]
}
```

## Manifest Fields

`version`: Human-readable release version. The updater records the actual GitHub tag as the installed release.

`channel`: Must match `updates.github.channel` on the Jetson. If the device is configured for `stable`, a manifest marked `beta` is skipped.

`assets`: List of install rules. Each rule maps one GitHub release asset to one local install action.

`name`: Stable local name for this update item. The updater uses this key in the state file to decide whether the same release asset was already installed.

`source`: Optional source for the asset. Omit it or use `github` for a normal GitHub Release asset. Use `kaggle_model` when the updater should download the model from Kaggle Models.

`kaggle_model`: Kaggle Model version handle, for example `owner/waste-detector/pyTorch/yolo-seg/3`. You can also provide the four-part model instance handle, such as `owner/waste-detector/pyTorch/yolo-seg`, plus `kaggle_model_version`.

`kaggle_model_version`: Specific Kaggle Model version number. Required only when `kaggle_model` has four path parts instead of five.

`kaggle_model_file`: File path inside the downloaded Kaggle Model version archive, for example `weights/best.pt` or `best.pt`.

Kaggle Model rules require Kaggle credentials on the Jetson, through `KAGGLE_USERNAME` / `KAGGLE_KEY` or `~/.kaggle/kaggle.json`.

`target`: Logical update category. Common values are `model`, `engine`, `config`, `program`, and `app`. CLI and MQTT commands can filter by this value.

`asset`: Exact GitHub Release asset filename to download for `github` rules. For `kaggle_model` rules, this is the local cache filename used before install.

`destination`: Local path where the asset should be installed. Relative paths resolve from `project_root`.

`sha256`: Expected SHA-256 checksum. Use this for every production asset.

`unpack`: Optional archive mode. Supported values are `zip`, `tar`, `tgz`, and `tar.gz`. When set, the archive is extracted into `destination`.

`install_command`: Optional command used instead of copying to `destination`. Use `{asset_path}` for the downloaded file path, `{project_root}` for the project root, `{release}` for the tag, and `{asset}` for the asset filename.

`restart_app`: Whether this asset should request an application restart after a successful update.

`executable`: Marks a copied file executable after installation.

`required`: When `true`, the update fails if the named GitHub asset is missing. When `false`, the asset is skipped if absent.

## Build And Upload

The fastest path is to use the release preparation script:

```bash
scripts/prepare_github_release.sh --version v0.2.0
```

This creates `release/v0.2.0/`, stages the configured artifacts, writes `edge-ai-update-manifest.json`, and writes `SHA256SUMS`.

If the detector is trained on Kaggle and published to Kaggle Models, include the Kaggle Model handle, its specific version, and the file path inside that model version:

```bash
scripts/prepare_github_release.sh \
  --version v0.2.0 \
  --kaggle-model owner/waste-detector/pyTorch/yolo-seg \
  --kaggle-model-version 3 \
  --kaggle-model-file weights/best.pt \
  --kaggle-model-dest models/weights/yolo-seg-best.pt \
  --no-detector-weights
```

That creates a manifest rule with `source: "kaggle_model"`. The model itself does not need to be uploaded as a GitHub Release asset, but the Jetson must have Kaggle credentials configured.

To publish with the GitHub CLI:

```bash
scripts/prepare_github_release.sh \
  --version v0.2.0 \
  --publish \
  --repo drovenai/edge-ai-mass-estimation
```

Manual flow:

1. Build Jetson-compatible artifacts.

```bash
python scripts/export_tensorrt.py --model models/weights/yolov8n-seg.pt --format engine --half
python -m build --wheel
```

2. Prepare a release folder.

```bash
mkdir -p release/v0.2.0
cp models/weights/yolov8n-seg.engine release/v0.2.0/
cp models/weights/mass_residual_best_model.joblib release/v0.2.0/
cp configs/pipeline/jetson_nano.yaml release/v0.2.0/
cp configs/agent/jetson_nano.yaml release/v0.2.0/agent_jetson_nano.yaml
cp configs/orchestration/jetson_nano.yaml release/v0.2.0/orchestration_jetson_nano.yaml
cp dist/edge_ai_mass-0.2.0-py3-none-any.whl release/v0.2.0/
```

3. Compute checksums.

Linux or Jetson:

```bash
sha256sum release/v0.2.0/*
```

Windows PowerShell:

```powershell
Get-FileHash release\v0.2.0\* -Algorithm SHA256
```

4. Create `release/v0.2.0/edge-ai-update-manifest.json` and paste the computed checksums into the matching `sha256` fields.

5. Publish the GitHub Release.

```bash
gh release create v0.2.0 release/v0.2.0/* --title "v0.2.0" --notes "Jetson model and app update"
```

For private repositories, set `GITHUB_TOKEN` on the Jetson so the updater can read release metadata and download assets.

## 404 From GitHub Latest Release

If update checks return a message like this:

```text
GitHub API request failed (404): {"message":"Not Found", ...}
```

GitHub could not resolve `GET /repos/<owner>/<repo>/releases/latest`. Check these points:

1. `EDGE_AI_GITHUB_OWNER` and `EDGE_AI_GITHUB_REPO` match the actual GitHub repository.
2. The repository has at least one published release. Draft releases do not count.
3. If the release is marked prerelease, either publish a normal release or set `include_prereleases: true`.
4. For private repositories, set `GITHUB_TOKEN` in the Jetson environment. Do not put the token directly in `configs/orchestration/jetson_nano.yaml`.
5. To test a specific release instead of `latest`, run:

```bash
edge-ai-orchestrator --config configs/orchestration/jetson_nano.yaml --once --release-tag v0.2.0
```

## How The Updater Applies A Release

1. The orchestrator starts from `edge-ai-orchestrator --config configs/orchestration/jetson_nano.yaml`, or through `edge-ai-mass orchestrator`.

2. It chooses an update source:
   - `poll`: check GitHub Releases on an interval.
   - `mqtt`: wait for a broker command.
   - `hybrid`: do both.
   - `manual`: run only when called with `--once`.

3. It asks GitHub for either:
   - the latest release when `updates.github.release: latest`, or
   - a specific tag when configured or passed with `--release-tag`.

4. It skips unsafe release states:
   - draft releases are skipped.
   - prereleases are skipped unless `include_prereleases: true`.
   - manifests with a different `channel` are skipped.

5. It downloads `edge-ai-update-manifest.json`. If the release has no manifest, it uses the fallback `updates.install_plan` in `configs/orchestration/jetson_nano.yaml`.

6. It filters install rules by requested target. Examples:

```bash
edge-ai-orchestrator --config configs/orchestration/jetson_nano.yaml --once --target model
edge-ai-orchestrator --config configs/orchestration/jetson_nano.yaml --once --target program
```

MQTT examples:

```json
{"action": "update", "target": "model"}
{"action": "update", "release": "v0.2.0", "target": "program", "force": true}
```

7. For each selected rule, it checks `.edge_ai_mass/orchestrator/state.json`. If the same rule name is already installed from the same GitHub tag, it skips that asset unless `force` is set.

8. It downloads each asset into `.edge_ai_mass/orchestrator/cache/<release-tag>/`. GitHub assets come from the release asset list. `source: "kaggle_model"` rules download the configured Kaggle Model version, extract it, select `kaggle_model_file`, and cache that file under the release tag.

9. It verifies `sha256` when provided. A mismatch fails the update.

10. It installs the asset:
    - Plain files are copied atomically to `destination`.
    - Existing files or directories are backed up under `.edge_ai_mass/orchestrator/backups/`.
    - Archives are safely extracted into a staging directory, then moved into `destination`.
    - Rules with `install_command` run that command instead of copying a file.

11. It runs `runtime.health_check_command` if configured. In the Jetson config this currently compiles the CLI module:

```yaml
health_check_command:
  - python3
  - -m
  - py_compile
  - src/edge_ai_mass/cli.py
```

12. If the health check fails, the updater rolls back file assets from this attempt and raises an error.

13. If the update succeeds, it records the installed release and assets in `.edge_ai_mass/orchestrator/state.json`.

14. If any installed rule has `restart_app: true`, and `runtime.restart_app_on_update` is true, the orchestrator restarts the managed app process or runs `runtime.restart_command`.

## Fallback Install Plan

The Jetson config includes a fallback plan for releases that do not contain `edge-ai-update-manifest.json`. This is useful during early development, but production releases should always include a manifest because the release then carries its own checksums and exact install rules.

Fallback plan location:

```text
configs/orchestration/jetson_nano.yaml
```

## Rollback

Rollback restores the latest backed-up file assets for a target.

```bash
edge-ai-orchestrator --config configs/orchestration/jetson_nano.yaml --rollback --target model
```

Rollback only applies to file or archive installs that created backups. It does not automatically undo arbitrary `install_command` side effects, such as a `pip install` command.

## Recommended Release Types

Model-only release:

```text
edge-ai-update-manifest.json
yolov8n-seg.engine
mass_residual_best_model.joblib
```

Kaggle Model-backed detector release:

```text
edge-ai-update-manifest.json
jetson_nano.yaml
agent_jetson_nano.yaml
orchestration_jetson_nano.yaml
```

In this release type, the manifest points to the exact Kaggle Model version and `kaggle_model_file`; the detector weight itself stays in Kaggle Models.

Config-only release:

```text
edge-ai-update-manifest.json
jetson_nano.yaml
```

Full application release:

```text
edge-ai-update-manifest.json
edge_ai_mass-0.2.0-py3-none-any.whl
yolov8n-seg.engine
mass_residual_best_model.joblib
jetson_nano.yaml
agent_jetson_nano.yaml
orchestration_jetson_nano.yaml
```

## Operational Notes

Use immutable tags such as `v0.2.0` and avoid editing assets after devices have started installing them.

Keep TensorRT engines separate from PyTorch weights so Jetson devices can update only the fast runtime artifact when needed.

Mark experimental releases as prereleases and keep `include_prereleases: false` on production devices.

Use `required: false` for optional artifacts during staged rollouts, and `required: true` for assets that must be present for the release to be valid.

For private GitHub repositories, use a token with the minimum read permissions needed for releases.
