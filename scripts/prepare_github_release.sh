#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Prepare a GitHub Release folder for the Jetson updater.

Usage:
  scripts/prepare_github_release.sh --version v0.2.0 [options]

Options:
  --version TAG             Release tag/version, for example v0.2.0. Required.
  --channel NAME            Manifest channel. Default: dev.
  --out-dir DIR             Parent release directory. Default: release.
  --engine PATH             TensorRT detector engine path.
                            Default: models/weights/yolov8n-seg.engine
  --detector-weights PATH   PyTorch detector weights path.
                            Default: models/weights/yolo-seg-best.pt
  --mass-model PATH         Residual mass model artifact path.
                            Default: artifacts/mass_estimation/mass-model-feature-residuals/models/best_model.joblib
  --mass-model-asset NAME   Release asset filename for the mass model.
                            Default: mass_residual_best_model.joblib
  --mass-weights PATH       Deprecated alias for --mass-model.
  --kaggle-model HANDLE     Kaggle Model instance or version handle, for example
                            owner/model/framework/variation or
                            owner/model/framework/variation/1.
  --kaggle-model-version N  Kaggle Model version number. Required when
                            --kaggle-model has only four path parts.
  --kaggle-model-file PATH  File path inside the Kaggle Model version archive,
                            for example weights/best.pt or best.pt.
  --kaggle-model-name NAME  Local release/cache filename for the Kaggle model.
                            Default: basename of --kaggle-model-file.
  --kaggle-model-dest PATH  Jetson install destination for the Kaggle model.
                            Default: models/weights/yolo-seg-best.pt
  --kaggle-model-sha256 SHA Optional SHA-256 for the Kaggle model file.
  --pipeline-config PATH    Jetson pipeline config path.
                            Default: configs/pipeline/jetson_nano.yaml
  --agent-config PATH       Jetson agent config path.
                            Default: configs/agent/jetson_nano.yaml
  --orchestrator-config PATH
                            Jetson updater/orchestrator config path.
                            Default: configs/orchestration/jetson_nano.yaml
  --wheel PATH              Wheel file path. Default: newest dist/edge_ai_mass*.whl.
  --no-engine               Do not include TensorRT engine.
  --no-detector-weights     Do not include PyTorch detector weights.
  --no-mass-model           Do not include the residual mass model.
  --no-mass-weights         Deprecated alias for --no-mass-model.
  --no-kaggle-model         Do not include a Kaggle Model version rule.
  --no-pipeline-config      Do not include Jetson pipeline config.
  --no-agent-config         Do not include Jetson agent config.
  --no-orchestrator-config  Do not include Jetson updater/orchestrator config.
  --no-wheel                Do not include application wheel.
  --move                    Move artifacts into release folder instead of copying.
  --publish                 Create the GitHub Release with gh release create.
  --repo OWNER/REPO         Repository for gh release create.
  --notes TEXT              Release notes used with --publish.
  --force                   Recreate output folder if it already exists.
  --help                    Show this help.

Environment:
  GH_BIN                    Explicit path to the GitHub CLI executable.

Examples:
  scripts/prepare_github_release.sh --version v0.2.0
  scripts/prepare_github_release.sh --version v0.2.0 --kaggle-model owner/model/pyTorch/default --kaggle-model-version 1 --kaggle-model-file best.pt
  scripts/prepare_github_release.sh --version v0.2.0 --publish --repo drovenai/edge-ai-mass-estimation
  scripts/prepare_github_release.sh --version v0.2.0 --no-wheel --no-detector-weights
EOF
}

version=""
channel="dev"
out_dir="release"
engine_path="models/weights/yolov8n-seg.engine"
detector_weights_path="models/weights/yolo-seg-best.pt"
mass_model_path="artifacts/mass_estimation/mass-model-feature-residuals/models/best_model.joblib"
mass_model_asset="mass_residual_best_model.joblib"
kaggle_model=""
kaggle_model_version=""
kaggle_model_file=""
kaggle_model_name=""
kaggle_model_dest="models/weights/yolo-seg-best.pt"
kaggle_model_sha256=""
pipeline_config_path="configs/pipeline/jetson_nano.yaml"
agent_config_path="configs/agent/jetson_nano.yaml"
orchestrator_config_path="configs/orchestration/jetson_nano.yaml"
wheel_path=""
include_engine=1
include_detector_weights=1
include_mass_model=1
include_kaggle_model=1
include_pipeline_config=1
include_agent_config=1
include_orchestrator_config=1
include_wheel=1
move_files=0
publish=0
repo=""
notes="Jetson model and app update"
force=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      version="${2:-}"
      shift 2
      ;;
    --channel)
      channel="${2:-}"
      shift 2
      ;;
    --out-dir)
      out_dir="${2:-}"
      shift 2
      ;;
    --engine)
      engine_path="${2:-}"
      include_engine=1
      shift 2
      ;;
    --detector-weights)
      detector_weights_path="${2:-}"
      include_detector_weights=1
      shift 2
      ;;
    --mass-weights)
      mass_model_path="${2:-}"
      include_mass_model=1
      shift 2
      ;;
    --mass-model)
      mass_model_path="${2:-}"
      include_mass_model=1
      shift 2
      ;;
    --mass-model-asset)
      mass_model_asset="${2:-}"
      include_mass_model=1
      shift 2
      ;;
    --kaggle-model)
      kaggle_model="${2:-}"
      include_kaggle_model=1
      shift 2
      ;;
    --kaggle-model-version)
      kaggle_model_version="${2:-}"
      include_kaggle_model=1
      shift 2
      ;;
    --kaggle-model-file)
      kaggle_model_file="${2:-}"
      include_kaggle_model=1
      shift 2
      ;;
    --kaggle-model-name)
      kaggle_model_name="${2:-}"
      include_kaggle_model=1
      shift 2
      ;;
    --kaggle-model-dest)
      kaggle_model_dest="${2:-}"
      include_kaggle_model=1
      shift 2
      ;;
    --kaggle-model-sha256)
      kaggle_model_sha256="${2:-}"
      include_kaggle_model=1
      shift 2
      ;;
    --pipeline-config)
      pipeline_config_path="${2:-}"
      include_pipeline_config=1
      shift 2
      ;;
    --agent-config)
      agent_config_path="${2:-}"
      include_agent_config=1
      shift 2
      ;;
    --orchestrator-config)
      orchestrator_config_path="${2:-}"
      include_orchestrator_config=1
      shift 2
      ;;
    --wheel)
      wheel_path="${2:-}"
      include_wheel=1
      shift 2
      ;;
    --no-engine)
      include_engine=0
      shift
      ;;
    --no-detector-weights)
      include_detector_weights=0
      shift
      ;;
    --no-mass-weights)
      include_mass_model=0
      shift
      ;;
    --no-mass-model)
      include_mass_model=0
      shift
      ;;
    --no-kaggle-model)
      include_kaggle_model=0
      shift
      ;;
    --no-pipeline-config)
      include_pipeline_config=0
      shift
      ;;
    --no-agent-config)
      include_agent_config=0
      shift
      ;;
    --no-orchestrator-config)
      include_orchestrator_config=0
      shift
      ;;
    --no-wheel)
      include_wheel=0
      shift
      ;;
    --move)
      move_files=1
      shift
      ;;
    --publish)
      publish=1
      shift
      ;;
    --repo)
      repo="${2:-}"
      shift 2
      ;;
    --notes)
      notes="${2:-}"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$version" ]]; then
  echo "--version is required, for example --version v0.2.0" >&2
  exit 2
fi

release_dir="${out_dir%/}/${version}"
manifest_path="${release_dir}/edge-ai-update-manifest.json"

sha256_file() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
  else
    echo "sha256sum or shasum is required" >&2
    exit 1
  fi
}

resolve_gh_command() {
  if [[ -n "${GH_BIN:-}" ]]; then
    if [[ -x "$GH_BIN" ]]; then
      printf '%s' "$GH_BIN"
      return 0
    fi
    echo "GH_BIN is set but not executable: $GH_BIN" >&2
    return 1
  fi

  if command -v gh >/dev/null 2>&1; then
    command -v gh
    return 0
  fi

  if command -v gh.exe >/dev/null 2>&1; then
    command -v gh.exe
    return 0
  fi

  return 1
}

append_sha256_sum() {
  local file="$1"
  local checksum
  checksum="$(sha256_file "$file")"
  printf '%s  %s\n' "$checksum" "$(basename "$file")" >>"${release_dir}/SHA256SUMS"
}

json_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  printf '%s' "$value"
}

find_newest_wheel() {
  local newest=""
  shopt -s nullglob
  local wheels=(dist/edge_ai_mass*.whl dist/edge-ai-mass*.whl)
  shopt -u nullglob
  if [[ ${#wheels[@]} -eq 0 ]]; then
    return 1
  fi
  newest="$(ls -t "${wheels[@]}" | head -n 1)"
  printf '%s' "$newest"
}

kaggle_model_basename() {
  local path="$1"
  path="${path//\\//}"
  path="${path%/}"
  printf '%s' "${path##*/}"
}

kaggle_model_version_ref() {
  local model="$1"
  local version="$2"
  model="${model#/}"
  model="${model%/}"
  version="${version#/}"
  version="${version%/}"

  local parts
  IFS='/' read -r -a parts <<<"$model"
  if [[ "${#parts[@]}" -eq 5 ]]; then
    printf '%s' "$model"
    return 0
  fi
  if [[ "${#parts[@]}" -eq 4 && -n "$version" ]]; then
    printf '%s/%s' "$model" "$version"
    return 0
  fi
  return 1
}

stage_asset() {
  local source="$1"
  local missing_policy="$2"
  local asset_name="${3:-$(basename "$source")}"
  local destination="${release_dir}/${asset_name}"

  if [[ ! -f "$source" ]]; then
    if [[ "$missing_policy" == "optional" ]]; then
      echo "Skipping missing optional asset: $source"
      return 1
    fi
    echo "Required asset is missing: $source" >&2
    exit 1
  fi

  if [[ "$move_files" -eq 1 ]]; then
    mv "$source" "$destination"
    echo "Moved ${asset_name}"
  else
    cp "$source" "$destination"
    echo "Copied ${asset_name}"
  fi
  return 0
}

append_file_rule() {
  local name="$1"
  local target="$2"
  local asset="$3"
  local destination="$4"
  local checksum="$5"
  local required="$6"
  local prefix="$7"

  cat >>"$manifest_path" <<EOF
${prefix}{
${prefix}  "name": "$(json_escape "$name")",
${prefix}  "target": "$(json_escape "$target")",
${prefix}  "asset": "$(json_escape "$asset")",
${prefix}  "destination": "$(json_escape "$destination")",
${prefix}  "sha256": "$checksum",
${prefix}  "restart_app": true,
${prefix}  "required": $required
${prefix}}
EOF
}

append_kaggle_model_rule() {
  local name="$1"
  local asset="$2"
  local destination="$3"
  local model="$4"
  local model_file="$5"
  local checksum="$6"
  local required="$7"
  local prefix="$8"

  cat >>"$manifest_path" <<EOF
${prefix}{
${prefix}  "name": "$(json_escape "$name")",
${prefix}  "source": "kaggle_model",
${prefix}  "target": "model",
${prefix}  "asset": "$(json_escape "$asset")",
${prefix}  "destination": "$(json_escape "$destination")",
${prefix}  "sha256": "$(json_escape "$checksum")",
${prefix}  "kaggle_model": "$(json_escape "$model")",
${prefix}  "kaggle_model_file": "$(json_escape "$model_file")",
${prefix}  "restart_app": true,
${prefix}  "required": $required
${prefix}}
EOF
}

append_wheel_rule() {
  local asset="$1"
  local checksum="$2"
  local required="$3"
  local prefix="$4"

  cat >>"$manifest_path" <<EOF
${prefix}{
${prefix}  "name": "application-wheel",
${prefix}  "target": "program",
${prefix}  "asset": "$(json_escape "$asset")",
${prefix}  "sha256": "$checksum",
${prefix}  "install_command": [
${prefix}    "python3",
${prefix}    "-m",
${prefix}    "pip",
${prefix}    "install",
${prefix}    "--no-deps",
${prefix}    "--upgrade",
${prefix}    "{asset_path}"
${prefix}  ],
${prefix}  "restart_app": true,
${prefix}  "required": $required
${prefix}}
EOF
}

if [[ -e "$release_dir" ]]; then
  if [[ "$force" -eq 1 ]]; then
    rm -rf "$release_dir"
  else
    echo "Release folder already exists: $release_dir" >&2
    echo "Use --force to recreate it." >&2
    exit 1
  fi
fi

mkdir -p "$release_dir"

if [[ "$include_wheel" -eq 1 && -z "$wheel_path" ]]; then
  if ! wheel_path="$(find_newest_wheel)"; then
    echo "No wheel found in dist/. Use --wheel PATH or --no-wheel." >&2
    exit 1
  fi
fi

declare -a rule_lines=()

if [[ "$include_engine" -eq 1 ]] && stage_asset "$engine_path" "optional"; then
  asset="$(basename "$engine_path")"
  checksum="$(sha256_file "${release_dir}/${asset}")"
  rule_lines+=("file|detection-tensorrt-engine|model|$asset|models/weights/$asset|$checksum|false")
fi

if [[ "$include_detector_weights" -eq 1 ]] && stage_asset "$detector_weights_path" "optional"; then
  asset="$(basename "$detector_weights_path")"
  checksum="$(sha256_file "${release_dir}/${asset}")"
  rule_lines+=("file|detection-yolo-weights|model|$asset|models/weights/$asset|$checksum|false")
fi

if [[ "$include_mass_model" -eq 1 ]] && stage_asset "$mass_model_path" "optional" "$mass_model_asset"; then
  asset="$mass_model_asset"
  checksum="$(sha256_file "${release_dir}/${asset}")"
  rule_lines+=("file|mass-residual-model|model|$asset|models/weights/$asset|$checksum|false")
fi

if [[ "$include_kaggle_model" -eq 1 && -n "$kaggle_model" && -n "$kaggle_model_file" ]]; then
  kaggle_model_ref="$(kaggle_model_version_ref "$kaggle_model" "$kaggle_model_version" || true)"
  if [[ -z "$kaggle_model_ref" ]]; then
    echo "--kaggle-model must include a version, or --kaggle-model-version must be provided" >&2
    exit 1
  fi
  if [[ -z "$kaggle_model_name" ]]; then
    kaggle_model_name="$(kaggle_model_basename "$kaggle_model_file")"
  fi
  if [[ -z "$kaggle_model_name" ]]; then
    echo "Could not infer --kaggle-model-name from --kaggle-model-file" >&2
    exit 1
  fi
  rule_lines+=("kaggle|kaggle-trained-detector|model|$kaggle_model_name|$kaggle_model_dest|$kaggle_model_sha256|true|$kaggle_model_ref|$kaggle_model_file")
elif [[ "$include_kaggle_model" -eq 1 ]] && { [[ -n "$kaggle_model" ]] || [[ -n "$kaggle_model_file" ]] || [[ -n "$kaggle_model_version" ]]; }; then
  echo "--kaggle-model and --kaggle-model-file must be provided together" >&2
  exit 1
fi

if [[ "$include_pipeline_config" -eq 1 ]] && stage_asset "$pipeline_config_path" "required"; then
  asset="$(basename "$pipeline_config_path")"
  checksum="$(sha256_file "${release_dir}/${asset}")"
  rule_lines+=("file|jetson-pipeline-config|config|$asset|configs/pipeline/$asset|$checksum|true")
fi

if [[ "$include_agent_config" -eq 1 ]] && stage_asset "$agent_config_path" "required" "agent_jetson_nano.yaml"; then
  asset="agent_jetson_nano.yaml"
  checksum="$(sha256_file "${release_dir}/${asset}")"
  rule_lines+=("file|jetson-agent-config|config|$asset|configs/agent/jetson_nano.yaml|$checksum|true")
fi

if [[ "$include_orchestrator_config" -eq 1 ]] && stage_asset "$orchestrator_config_path" "required" "orchestration_jetson_nano.yaml"; then
  asset="orchestration_jetson_nano.yaml"
  checksum="$(sha256_file "${release_dir}/${asset}")"
  rule_lines+=("file|jetson-orchestrator-config|config|$asset|configs/orchestration/jetson_nano.yaml|$checksum|true")
fi

if [[ "$include_wheel" -eq 1 ]] && stage_asset "$wheel_path" "optional"; then
  asset="$(basename "$wheel_path")"
  checksum="$(sha256_file "${release_dir}/${asset}")"
  rule_lines+=("wheel|application-wheel|program|$asset||$checksum|false")
fi

if [[ ${#rule_lines[@]} -eq 0 ]]; then
  echo "No assets were staged; not writing a manifest." >&2
  exit 1
fi

cat >"$manifest_path" <<EOF
{
  "version": "$(json_escape "$version")",
  "channel": "$(json_escape "$channel")",
  "assets": [
EOF

for index in "${!rule_lines[@]}"; do
  line="${rule_lines[$index]}"
  IFS='|' read -r kind name target asset destination checksum required kaggle_model_rule kaggle_model_file_rule <<<"$line"
  if [[ "$index" -gt 0 ]]; then
    printf ',\n' >>"$manifest_path"
  fi
  if [[ "$kind" == "wheel" ]]; then
    append_wheel_rule "$asset" "$checksum" "$required" "    "
  elif [[ "$kind" == "kaggle" ]]; then
    append_kaggle_model_rule "$name" "$asset" "$destination" "$kaggle_model_rule" "$kaggle_model_file_rule" "$checksum" "$required" "    "
  else
    append_file_rule "$name" "$target" "$asset" "$destination" "$checksum" "$required" "    "
  fi
done

cat >>"$manifest_path" <<'EOF'

  ]
}
EOF

: >"${release_dir}/SHA256SUMS"
for file in "$release_dir"/*; do
  if [[ "$(basename "$file")" == "SHA256SUMS" ]]; then
    continue
  fi
  append_sha256_sum "$file"
done

echo
echo "Release folder ready: $release_dir"
echo "Manifest: $manifest_path"
echo "Assets:"
find "$release_dir" -maxdepth 1 -type f -printf '  %f\n' 2>/dev/null || ls -1 "$release_dir"

if [[ "$publish" -eq 1 ]]; then
  gh_bin="$(resolve_gh_command || true)"
  if [[ -z "$gh_bin" ]]; then
    echo "gh CLI is required for --publish. Set GH_BIN if gh is installed but not on PATH." >&2
    exit 1
  fi
  gh_args=(release create "$version" "$release_dir"/* --title "$version" --notes "$notes")
  if [[ -n "$repo" ]]; then
    gh_args+=(--repo "$repo")
  fi
  echo
  echo "Publishing GitHub Release $version"
  "$gh_bin" "${gh_args[@]}"
fi
