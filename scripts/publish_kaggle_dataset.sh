#!/usr/bin/env bash
set -Eeuo pipefail

# Publish a new version of the importable edge-ai-mass Kaggle dataset.
# The Python publisher performs notebook compilation, wheel/source packaging,
# Kaggle authentication, upload, and readiness polling.

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

find_python() {
    local candidate
    for candidate in \
        "${PYTHON:-}" \
        "$PROJECT_ROOT/.venv/Scripts/python.exe" \
        "$PROJECT_ROOT/.venv/bin/python"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done

    printf 'Error: Python was not found. Set PYTHON=/path/to/python.\n' >&2
    return 1
}

PYTHON_BIN="$(find_python)"
if [[ $# -eq 0 ]]; then
    set -- --message "Publish latest edge-ai-mass source and training diagnostics"
fi

printf 'Project: %s\n' "$PROJECT_ROOT"
printf 'Python:  %s\n' "$PYTHON_BIN"

"$PYTHON_BIN" scripts/publish_kaggle_training.py publish \
    --skip-kernel \
    "$@"
