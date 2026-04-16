# Platonic Transformers

Research code for the paper [Platonic Transformers: A Solid Choice For Equivariance](https://www.arxiv.org/abs/2510.03511).

This repository contains training code for:

- CIFAR-10
- QM9 regression
- OMol
- ImageNet with DALI

The main entrypoint is `meta_main.py`. Configuration lives in `configs/`.

## Installation

This repo now uses `uv` as the source of truth for environments and dependencies. The lockfile is committed in `uv.lock`.

Base install:

```bash
uv sync --frozen
```

Optional extras:

```bash
# OMol / fairchem stack
uv sync --frozen --extra omol

# ImageNet DALI stack
uv sync --frozen --extra imagenet

# torch-cluster for kNN graph mode
uv sync --frozen --extra knn
```

The extras are intentionally split:

- `omol` is Linux-only
- `imagenet` is Linux x86_64 only
- `knn` installs the pinned Linux wheel for `torch-cluster`

If you prefer the wrapper script:

```bash
./setup.sh
```

You can also pass extras through the script:

```bash
UV_EXTRAS=omol ./setup.sh
UV_EXTRAS=imagenet,knn ./setup.sh
```

## Quick Start

List entrypoints:

```bash
uv run python meta_main.py --help
```

Train CIFAR-10:

```bash
uv run python meta_main.py cifar10 --batch_size 256 --lr 8e-4
```

Train QM9:

```bash
uv run python meta_main.py qm9_regr --target mu --batch_size 96
```

Train OMol:

```bash
uv run python meta_main.py omol --predict_forces --force_weight 100
```

Train ImageNet:

```bash
uv run python meta_main.py imagenet --config configs/imagenet_dali.yaml --batch_size 128
```

## Configuration

Each dataset entrypoint loads a default config from `configs/`:

- `cifar10` -> `configs/cifar10_deit.yaml`
- `qm9_regr` -> `configs/qm9_regr.yaml`
- `omol` -> `configs/omol.yaml`
- `imagenet` -> `configs/imagenet_dali.yaml`

You can replace the config entirely:

```bash
uv run python meta_main.py qm9_regr --config configs/qm9_regr.yaml
```

Or override individual fields from the command line:

```bash
uv run python meta_main.py qm9_regr --target alpha --batch_size 128 --epochs 300
```

## Snellius

The cluster workflow lives in `scripts/jobs/`.

Relevant files:

- `scripts/jobs/_job_common.sh`
- `scripts/jobs/setup_qm9_environment.job`
- `scripts/jobs/train_qm9_regr.job`
- `scripts/jobs/eval_qm9_regr.job`

The job workflow is designed to avoid shared-state races:

- jobs do not run from the shared checkout directly
- jobs do not mutate a shared repo-local `.venv`
- each run gets a unique staged repo snapshot
- logs, checkpoints, temp files, and WandB state go to unique per-run directories
- QM9 dataset prep and stats creation are protected with filesystem locks

Required environment before `sbatch`:

```bash
export QM9_DATA_DIR=/scratch-shared/$USER/datasets/qm9
```

Optional:

```bash
export WANDB_API_KEY=...
export PT_CLUSTER_ROOT=/scratch-shared/$USER/platonic-transformers
```

Setup / staging:

```bash
sbatch scripts/jobs/setup_qm9_environment.job
```

QM9 training:

```bash
sbatch --export=ALL,RUN_NAME=qm9-mu,QM9_TARGET=mu scripts/jobs/train_qm9_regr.job
```

QM9 evaluation:

```bash
sbatch --export=ALL,RUN_NAME=qm9-mu-eval,QM9_TARGET=mu,QM9_TEST_CKPT=/scratch-shared/$USER/platonic-transformers/runs/<train-run>/artifacts/lightning/<checkpoint>.ckpt scripts/jobs/eval_qm9_regr.job
```

If you need extras in the shared cluster environment, set:

```bash
export UV_SYNC_EXTRAS=omol
```

or:

```bash
export UV_SYNC_EXTRAS=imagenet,knn
```

## Repository Layout

```text
.
├── meta_main.py
├── configs/
├── mains/
├── platonic_transformers/
├── scripts/
│   └── jobs/
├── pyproject.toml
├── uv.lock
└── setup.sh
```

## Notes

- `pytorch-lightning` is the package used by the current training code.
- `torch-cluster` is not part of the base environment because it is only needed for kNN graph mode.
- ImageNet training requires the `imagenet` extra and a working NVIDIA DALI installation on Linux.
- OMol requires the `omol` extra.

## Verification

Minimal infra checks:

```bash
bash -n setup.sh scripts/jobs/_job_common.sh scripts/jobs/setup_qm9_environment.job scripts/jobs/train_qm9_regr.job scripts/jobs/eval_qm9_regr.job
python3 -m py_compile scripts/jobs/prepare_qm9.py
uv sync --frozen --dry-run
```

## License

MIT. See `LICENSE`.
