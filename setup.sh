#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required to install uv." >&2
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

UV_ENV_PATH="${UV_ENV_PATH:-$REPO_ROOT/.venv}"
UV_PYTHON_VERSION="${UV_PYTHON_VERSION:-3.12}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

uv venv --python "$UV_PYTHON_VERSION" "$UV_ENV_PATH"
uv pip install --python "$UV_ENV_PATH/bin/python" -r "$REPO_ROOT/requirements.txt"
uv pip install --python "$UV_ENV_PATH/bin/python" -e "$REPO_ROOT"

printf 'Environment ready at %s\n' "$UV_ENV_PATH"
printf 'Activate it with: source %s/bin/activate\n' "$UV_ENV_PATH"
