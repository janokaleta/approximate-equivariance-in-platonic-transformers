# Snellius Workflow

This repo now uses a cluster-safe Slurm workflow under `scripts/jobs/` for QM9 work on Snellius.

## Design

- Jobs never run from the shared checkout directly. Each submission stages a fresh snapshot into a unique run directory keyed by `RUN_NAME`, timestamp, and Slurm job id.
- Jobs never mutate a shared repo-local `.venv`. The helper builds an immutable environment under `/scratch-shared/$USER/platonic-transformers/envs/<env-key>` and symlinks the staged repo `.venv` to that location.
- All writable paths are unique per run:
  - staged source: `${PT_CLUSTER_ROOT}/runs/<run-key>/repo`
  - Lightning logs and checkpoints: `${PT_CLUSTER_ROOT}/runs/<run-key>/artifacts/lightning`
  - `lightning_logs/`: `${PT_CLUSTER_ROOT}/runs/<run-key>/artifacts/lightning_logs`
  - temp, pycache, matplotlib, and wandb dirs: under `${PT_CLUSTER_ROOT}/runs/<run-key>/...`
- Shared assets that are safe to reuse stay under `${PT_CLUSTER_ROOT}/cache` or under the user-provided `QM9_DATA_DIR`.
- QM9 dataset download and `stats_<target>.npz` creation are protected by filesystem locks so concurrent jobs do not trample each other.

## Why No `uv.lock`

This repo is not ready for a reliable `uv sync --locked` flow yet.

- `pyproject.toml` and `requirements.txt` disagree on the actual install surface.
- The current install relies on CUDA-specific PyTorch/PyG wheel sources declared in `requirements.txt`.
- `nvidia-dali-cuda120` and the PyG wheel link strategy make a single committed lock file brittle across machines and cluster images.

For now the safer path is:

- build an immutable environment from `requirements.txt`
- run jobs with `uv run --no-sync`
- revisit `uv.lock` only after the dependency model is consolidated into `pyproject.toml`

## Local Setup

From the repo root:

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

`setup.sh` installs `uv` if needed, creates `.venv`, installs `requirements.txt`, and installs the repo in editable mode. This is intended for Linux/CUDA environments matching the project dependencies.

## Required Environment

Put cluster-specific values in `.env` or export them before `sbatch`:

```bash
QM9_DATA_DIR=/scratch-shared/$USER/datasets/qm9
WANDB_API_KEY=...
```

Optional overrides:

```bash
PT_CLUSTER_ROOT=/scratch-shared/$USER/platonic-transformers
UV_PYTHON_VERSION=3.12
RUN_NAME=qm9-mu-baseline
QM9_TARGET=mu
QM9_TARGETS=mu,alpha
QM9_BATCH_SIZE=96
QM9_EPOCHS=1000
QM9_NUM_WORKERS=8
QM9_ENABLE_WANDB=1
QM9_EXTRA_ARGS='--hidden_dim 768 --num_layers 12'
```

## Job Templates

### 1. Setup / staging

Build or reuse the immutable environment, stage a repo snapshot, verify CUDA, warm the QM9 dataset, and precompute stats:

```bash
sbatch scripts/jobs/setup_qm9_environment.job
```

To precompute multiple targets:

```bash
sbatch --export=ALL,QM9_TARGETS=mu,alpha,gap scripts/jobs/setup_qm9_environment.job
```

### 2. QM9 training

```bash
sbatch --export=ALL,RUN_NAME=qm9-mu,QM9_TARGET=mu scripts/jobs/train_qm9_regr.job
```

Example with overrides:

```bash
sbatch --export=ALL,RUN_NAME=qm9-gap-fast,QM9_TARGET=gap,QM9_EPOCHS=200,QM9_BATCH_SIZE=128,QM9_EXTRA_ARGS='--hidden_dim 768 --num_layers 12' scripts/jobs/train_qm9_regr.job
```

### 3. QM9 eval / experiment replay

```bash
sbatch --export=ALL,RUN_NAME=qm9-mu-eval,QM9_TARGET=mu,QM9_TEST_CKPT=/scratch-shared/$USER/platonic-transformers/runs/<train-run>/artifacts/lightning/<...>.ckpt scripts/jobs/eval_qm9_regr.job
```

## Verification

Basic static verification from the repo root:

```bash
bash -n setup.sh scripts/jobs/_job_common.sh scripts/jobs/setup_qm9_environment.job scripts/jobs/train_qm9_regr.job scripts/jobs/eval_qm9_regr.job
python3 -m py_compile scripts/jobs/prepare_qm9.py
```

Optional helper smoke test without building the heavy environment:

```bash
source scripts/jobs/_job_common.sh
resolve_repo_root
setup_cluster_layout
initialize_run_context dry-run
stage_repo_snapshot
```

## Known Risks

- Local macOS setup is still not a supported target for the full dependency set because the requirements include Linux/CUDA-specific packages.
- The immutable env cache key is based on `requirements.txt`, `pyproject.toml`, and `setup.sh`; if cluster modules change materially, rebuild by changing one of those inputs or deleting the matching env directory.
- QM9 evaluation currently reuses the training entrypoint because the repo has no standalone eval driver for QM9.
