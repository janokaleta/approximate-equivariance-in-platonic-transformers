#!/usr/bin/env bash
set -euo pipefail

JOB_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${JOB_COMMON_DIR}/../.." && pwd)"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Required command '$cmd' was not found in PATH."
}

require_env_var() {
  local var_name="$1"
  [[ -n "${!var_name:-}" ]] || die "Required environment variable '$var_name' is not set."
}

require_file() {
  local file_path="$1"
  [[ -f "$file_path" ]] || die "Required file was not found: $file_path"
}

load_dotenv_if_present() {
  local dotenv_path="${1:-${REPO_ROOT:-$DEFAULT_REPO_ROOT}/.env}"
  if [[ -f "$dotenv_path" ]]; then
    log "Loading environment variables from $dotenv_path"
    set -a
    # shellcheck disable=SC1090
    source "$dotenv_path"
    set +a
  fi
}

resolve_repo_root() {
  local candidate_root

  if [[ -n "${REPO_ROOT_OVERRIDE:-}" ]]; then
    candidate_root="$REPO_ROOT_OVERRIDE"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    candidate_root="$SLURM_SUBMIT_DIR"
  else
    candidate_root="$DEFAULT_REPO_ROOT"
  fi

  cd "$candidate_root"
  [[ -f "pyproject.toml" ]] || die "pyproject.toml not found in $(pwd). Submit from the repo root or set REPO_ROOT_OVERRIDE."

  REPO_ROOT="$(pwd -P)"
  export REPO_ROOT

  log "Repo root: $REPO_ROOT"
  log "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}"
}

sanitize_slug() {
  local value="${1:-run}"
  value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
  value="$(printf '%s' "$value" | tr -cs 'a-z0-9._-' '-')"
  value="${value#-}"
  value="${value%-}"
  printf '%s' "${value:-run}"
}

hash_files() {
  require_cmd python3
  python3 - "$@" <<'PY'
import hashlib
import pathlib
import sys

hasher = hashlib.sha256()
for raw_path in sys.argv[1:]:
    path = pathlib.Path(raw_path)
    hasher.update(path.name.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(path.read_bytes())
    hasher.update(b"\0")
print(hasher.hexdigest())
PY
}

resolve_path() {
  require_cmd python3
  python3 - "${REPO_ROOT:-$DEFAULT_REPO_ROOT}" "$1" <<'PY'
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1]).resolve()
raw_value = pathlib.Path(sys.argv[2]).expanduser()
if raw_value.is_absolute():
    print(raw_value.resolve())
else:
    print((repo_root / raw_value).resolve())
PY
}

setup_cluster_layout() {
  local base_root="${PT_CLUSTER_ROOT:-/scratch-shared/${USER}/platonic-transformers}"

  export PT_CLUSTER_ROOT="$base_root"
  export PT_ENV_ROOT="${PT_ENV_ROOT:-${PT_CLUSTER_ROOT}/envs}"
  export PT_RUN_ROOT="${PT_RUN_ROOT:-${PT_CLUSTER_ROOT}/runs}"
  export PT_CACHE_ROOT="${PT_CACHE_ROOT:-${PT_CLUSTER_ROOT}/cache}"

  mkdir -p "$PT_ENV_ROOT" "$PT_RUN_ROOT"
  mkdir -p "$PT_CACHE_ROOT/huggingface" "$PT_CACHE_ROOT/torch" "$PT_CACHE_ROOT/wandb" "$PT_CACHE_ROOT/xdg"

  log "PT_CLUSTER_ROOT: $PT_CLUSTER_ROOT"
}

ensure_uv() {
  export PATH="$HOME/.local/bin:$PATH"

  if ! command -v uv >/dev/null 2>&1; then
    require_cmd curl
    log "Installing uv into $HOME/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  require_cmd uv
  export UV_BIN
  UV_BIN="$(command -v uv)"
  export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

  log "uv: $("$UV_BIN" --version)"
}

with_lock() {
  local lock_dir="$1"
  shift

  local timeout_seconds="${LOCK_WAIT_SECONDS:-1800}"
  local poll_seconds="${LOCK_POLL_SECONDS:-5}"
  local waited_seconds=0

  while ! mkdir "$lock_dir" 2>/dev/null; do
    if (( waited_seconds >= timeout_seconds )); then
      die "Timed out waiting for lock: $lock_dir"
    fi
    sleep "$poll_seconds"
    waited_seconds=$((waited_seconds + poll_seconds))
  done

  (
    trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT
    "$@"
  )
}

compute_env_key() {
  local python_version="${UV_PYTHON_VERSION:-3.12}"
  local fingerprint
  local extras_slug

  require_file "$REPO_ROOT/pyproject.toml"
  require_file "$REPO_ROOT/uv.lock"
  require_file "$REPO_ROOT/setup.sh"

  fingerprint="$(hash_files "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/uv.lock" "$REPO_ROOT/setup.sh")"
  extras_slug="$(sanitize_slug "${UV_SYNC_EXTRAS:-base}")"
  ENV_KEY="py${python_version//./}-${fingerprint:0:12}-${extras_slug}"
  export ENV_KEY
}

_build_shared_env_locked() {
  local build_dir
  local sync_args
  local raw_extra

  if [[ -x "$SHARED_ENV_PATH/bin/python" && -f "$SHARED_ENV_PATH/.ready" ]]; then
    log "Shared environment already exists: $SHARED_ENV_PATH"
    return 0
  fi

  build_dir="${PT_ENV_ROOT}/.${ENV_KEY}.tmp.${SLURM_JOB_ID:-manual}.$$"
  rm -rf "$build_dir"

  trap "rm -rf '$build_dir'" EXIT

  log "Creating immutable shared environment: $SHARED_ENV_PATH"
  "$UV_BIN" venv --python "${UV_PYTHON_VERSION:-3.12}" "$build_dir"

  sync_args=(--frozen --no-install-project)
  if [[ -n "${UV_SYNC_EXTRAS:-}" ]]; then
    IFS=',' read -r -a requested_extras <<< "$UV_SYNC_EXTRAS"
    for raw_extra in "${requested_extras[@]}"; do
      raw_extra="$(printf '%s' "$raw_extra" | xargs)"
      [[ -n "$raw_extra" ]] || continue
      sync_args+=(--extra "$raw_extra")
    done
  fi

  (
    cd "$REPO_ROOT"
    UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}" \
      UV_PROJECT_ENVIRONMENT="$build_dir" \
      "$UV_BIN" sync "${sync_args[@]}"
  )
  "$UV_BIN" pip check --python "$build_dir/bin/python"

  {
    printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repo_root=%s\n' "$REPO_ROOT"
    printf 'env_key=%s\n' "$ENV_KEY"
    printf 'extras=%s\n' "${UV_SYNC_EXTRAS:-}"
    printf 'python=%s\n' "$("$build_dir/bin/python" --version 2>&1)"
  } > "$build_dir/.metadata"

  touch "$build_dir/.ready"
  mv "$build_dir" "$SHARED_ENV_PATH"
  trap - EXIT
  log "Shared environment ready: $SHARED_ENV_PATH"
}

ensure_shared_env() {
  [[ -n "${PT_CLUSTER_ROOT:-}" ]] || die "PT_CLUSTER_ROOT is not set. Call setup_cluster_layout before ensure_shared_env."
  [[ -n "${PT_ENV_ROOT:-}" ]] || die "PT_ENV_ROOT is not set. Call setup_cluster_layout before ensure_shared_env."

  compute_env_key
  SHARED_ENV_PATH="${PT_ENV_ROOT}/${ENV_KEY}"
  export SHARED_ENV_PATH

  if [[ -x "$SHARED_ENV_PATH/bin/python" && -f "$SHARED_ENV_PATH/.ready" ]]; then
    log "Reusing shared environment: $SHARED_ENV_PATH"
    return 0
  fi

  with_lock "${SHARED_ENV_PATH}.lock" _build_shared_env_locked
  [[ -x "$SHARED_ENV_PATH/bin/python" && -f "$SHARED_ENV_PATH/.ready" ]] || die "Shared environment was not created successfully."
}

initialize_run_context() {
  local default_name="${1:-job}"
  local requested_name="${RUN_NAME:-$default_name}"

  RUN_NAME="$(sanitize_slug "$requested_name")"
  RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
  RUN_KEY="${RUN_NAME}_${RUN_TIMESTAMP}_job${SLURM_JOB_ID:-manual}"
  RUN_ROOT="${PT_RUN_ROOT}/${RUN_KEY}"
  STAGED_REPO_ROOT="${RUN_ROOT}/repo"
  RUN_ARTIFACT_ROOT="${RUN_ROOT}/artifacts"

  mkdir -p "$RUN_ROOT" "$RUN_ARTIFACT_ROOT" "$RUN_ROOT/tmp" "$RUN_ROOT/slurm"
  mkdir -p "$RUN_ARTIFACT_ROOT/lightning" "$RUN_ARTIFACT_ROOT/lightning_logs" "$RUN_ARTIFACT_ROOT/wandb"

  export RUN_NAME RUN_TIMESTAMP RUN_KEY RUN_ROOT STAGED_REPO_ROOT RUN_ARTIFACT_ROOT
  export TMPDIR="${RUN_ROOT}/tmp"
  export TMP="$TMPDIR"
  export TEMP="$TMPDIR"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PT_CACHE_ROOT}/xdg}"
  export HF_HOME="${HF_HOME:-${PT_CACHE_ROOT}/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-${PT_CACHE_ROOT}/torch}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${PT_CACHE_ROOT}/wandb}"
  export WANDB_DIR="${WANDB_DIR:-${RUN_ARTIFACT_ROOT}/wandb}"
  export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${RUN_ROOT}/pycache}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_ROOT}/matplotlib}"
  export WANDB_ENTITY="${WANDB_ENTITY:-platonic-transformers}"
  export WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
  export PYTHONUNBUFFERED=1

  mkdir -p "$TMPDIR" "$WANDB_DIR" "$PYTHONPYCACHEPREFIX" "$MPLCONFIGDIR"

  log "RUN_KEY: $RUN_KEY"
  log "RUN_ROOT: $RUN_ROOT"
}

stage_repo_snapshot() {
  require_cmd rsync
  mkdir -p "$STAGED_REPO_ROOT"

  log "Staging repo snapshot into $STAGED_REPO_ROOT"
  rsync -a \
    --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.mypy_cache/' \
    --exclude '.pytest_cache/' \
    --exclude '__pycache__/' \
    --exclude '.cache/' \
    --exclude 'data/' \
    --exclude 'wandb/' \
    --exclude 'mains/logs/' \
    --exclude 'lightning_logs/' \
    --exclude 'scripts/jobs/out/' \
    "$REPO_ROOT/" "$STAGED_REPO_ROOT/"

  rm -rf "$STAGED_REPO_ROOT/.venv" "$STAGED_REPO_ROOT/mains/logs" "$STAGED_REPO_ROOT/lightning_logs"
  ln -s "$RUN_ARTIFACT_ROOT/lightning" "$STAGED_REPO_ROOT/mains/logs"
  ln -s "$RUN_ARTIFACT_ROOT/lightning_logs" "$STAGED_REPO_ROOT/lightning_logs"
}

link_shared_env_into_stage() {
  [[ -n "${SHARED_ENV_PATH:-}" ]] || die "SHARED_ENV_PATH is not set."
  [[ -d "$STAGED_REPO_ROOT" ]] || die "Staged repo does not exist: $STAGED_REPO_ROOT"

  rm -rf "$STAGED_REPO_ROOT/.venv"
  ln -s "$SHARED_ENV_PATH" "$STAGED_REPO_ROOT/.venv"
  log "Linked staged repo .venv -> $SHARED_ENV_PATH"
}

write_run_metadata() {
  local metadata_path="${RUN_ROOT}/run_metadata.env"
  local git_commit="unknown"
  local git_branch="unknown"

  if command -v git >/dev/null 2>&1 && git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    git_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    git_branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
  fi

  {
    printf 'RUN_KEY=%s\n' "$RUN_KEY"
    printf 'RUN_NAME=%s\n' "$RUN_NAME"
    printf 'RUN_TIMESTAMP=%s\n' "$RUN_TIMESTAMP"
    printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
    printf 'STAGED_REPO_ROOT=%s\n' "$STAGED_REPO_ROOT"
    printf 'SHARED_ENV_PATH=%s\n' "${SHARED_ENV_PATH:-}"
    printf 'REPO_ROOT=%s\n' "$REPO_ROOT"
    printf 'GIT_BRANCH=%s\n' "$git_branch"
    printf 'GIT_COMMIT=%s\n' "$git_commit"
    printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-}"
  } > "$metadata_path"

  log "Run metadata written to $metadata_path"
}

print_runtime_diagnostics() {
  log "Python cache: $PYTHONPYCACHEPREFIX"
  log "TMPDIR: $TMPDIR"
  log "HF_HOME: $HF_HOME"
  log "TORCH_HOME: $TORCH_HOME"
  log "WANDB_DIR: $WANDB_DIR"
}

validate_cuda_runtime() {
  (
    cd "$STAGED_REPO_ROOT"
    srun uv run --no-sync python -c 'import torch; print(f"CUDA available: {torch.cuda.is_available()}"); print(f"GPU count: {torch.cuda.device_count()}"); print(f"Device 0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu-only\"}")'
  )
}

run_qm9_prep() {
  (
    cd "$STAGED_REPO_ROOT"
    srun uv run --no-sync python scripts/jobs/prepare_qm9.py "$@"
  )
}

run_qm9_approx_sym_prep() {
  (
    cd "$STAGED_REPO_ROOT"
    srun uv run --no-sync python scripts/jobs/prepare_qm9_approx_sym.py "$@"
  )
}

_prepare_qm9_dataset_locked() {
  run_qm9_prep --data-dir "$QM9_DATA_DIR"
}

ensure_qm9_dataset_ready() {
  require_env_var "QM9_DATA_DIR"
  mkdir -p "$QM9_DATA_DIR"

  if [[ -f "${QM9_DATA_DIR}/processed/data_v3.pt" ]]; then
    log "QM9 dataset already prepared at $QM9_DATA_DIR"
    return 0
  fi

  with_lock "${QM9_DATA_DIR}/.prepare.lock" _prepare_qm9_dataset_locked
}

_prepare_qm9_stats_locked() {
  local target="$1"
  run_qm9_prep --data-dir "$QM9_DATA_DIR" --target "$target"
}

ensure_qm9_stats_ready() {
  local target="$1"
  local stats_file="${QM9_DATA_DIR}/stats_${target}.npz"

  require_env_var "QM9_DATA_DIR"
  mkdir -p "$QM9_DATA_DIR"

  if [[ -f "$stats_file" ]]; then
    log "QM9 stats already present: $stats_file"
    return 0
  fi

  with_lock "${stats_file}.lock" _prepare_qm9_stats_locked "$target"
}

set_default_qm9_approx_sym_env() {
  : "${QM9_APPROX_CACHE_DIR:=${PT_CLUSTER_ROOT}/datasets/qm9_approx_sym}"
  : "${QM9_APPROX_TARGET:=${QM9_TARGET:-mu}}"
  : "${QM9_APPROX_BREAK_STRENGTH:=0.10}"
  : "${QM9_APPROX_VIEWS_PER_MOLECULE:=2}"
  : "${QM9_APPROX_SPLIT_SEED:=42}"
  : "${QM9_APPROX_ROTATION_SEED:=1729}"
  : "${QM9_APPROX_TRAIN_SIZE:=110000}"
  : "${QM9_APPROX_VAL_SIZE:=10000}"

  export QM9_APPROX_CACHE_DIR QM9_APPROX_TARGET QM9_APPROX_BREAK_STRENGTH
  export QM9_APPROX_VIEWS_PER_MOLECULE QM9_APPROX_SPLIT_SEED QM9_APPROX_ROTATION_SEED
  export QM9_APPROX_TRAIN_SIZE QM9_APPROX_VAL_SIZE
}

_prepare_qm9_approx_sym_locked() {
  run_qm9_approx_sym_prep \
    --data-dir "$QM9_DATA_DIR" \
    --cache-dir "$QM9_APPROX_CACHE_DIR" \
    --target "$QM9_APPROX_TARGET" \
    --break-strength "$QM9_APPROX_BREAK_STRENGTH" \
    --views-per-molecule "$QM9_APPROX_VIEWS_PER_MOLECULE" \
    --split-seed "$QM9_APPROX_SPLIT_SEED" \
    --rotation-seed "$QM9_APPROX_ROTATION_SEED" \
    --train-size "$QM9_APPROX_TRAIN_SIZE" \
    --val-size "$QM9_APPROX_VAL_SIZE"
}

ensure_qm9_approx_sym_ready() {
  local lock_slug

  require_env_var "QM9_DATA_DIR"
  set_default_qm9_approx_sym_env
  mkdir -p "$QM9_APPROX_CACHE_DIR"

  lock_slug="$(sanitize_slug "${QM9_APPROX_TARGET}-${QM9_APPROX_BREAK_STRENGTH}-${QM9_APPROX_VIEWS_PER_MOLECULE}-${QM9_APPROX_SPLIT_SEED}-${QM9_APPROX_ROTATION_SEED}")"
  with_lock "${QM9_APPROX_CACHE_DIR}/.prepare-${lock_slug}.lock" _prepare_qm9_approx_sym_locked
}
