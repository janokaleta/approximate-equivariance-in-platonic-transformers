#!/usr/bin/env bash
# Resume all 7 Tier-1 RGC runs from their original checkpoints, training to epoch 500.
# Usage (from repo root on Snellius):  bash scripts/jobs/submit_resume_tier1.sh

set -euo pipefail

RUNS_ROOT="/scratch-shared/scur0203/platonic-transformers/runs"
JOB="scripts/jobs/train_qm9_relaxed_group_convolution.job"

RUN_NAME=qm9-rgc-k1-l1e2-continued RGC_NUM_EXTRA_KERNELS=1 RGC_MIXING_L2=0.01 QM9_EPOCHS=500 \
  RESUME_CKPT="$RUNS_ROOT/qm9-rgc-k1-l1e2_20260517T223159Z_job22825920/artifacts/lightning/qm9-rgc-ablations/or0zlylq/checkpoints/epoch=346-step=397662.ckpt" \
  sbatch --time=22:00:00 "$JOB"

RUN_NAME=qm9-rgc-k2-l1e2-continued RGC_NUM_EXTRA_KERNELS=2 RGC_MIXING_L2=0.01 QM9_EPOCHS=500 \
  RESUME_CKPT="$RUNS_ROOT/qm9-rgc-k2-l1e2_20260517T223201Z_job22825921/artifacts/lightning/qm9-rgc-ablations/rn8v0ion/checkpoints/epoch=296-step=340362.ckpt" \
  sbatch --time=22:00:00 "$JOB"

RUN_NAME=qm9-rgc-k4-l1e1-continued RGC_NUM_EXTRA_KERNELS=4 RGC_MIXING_L2=0.1 QM9_EPOCHS=500 \
  RESUME_CKPT="$RUNS_ROOT/qm9-rgc-k4-l1e1_20260517T223201Z_job22825923/artifacts/lightning/qm9-rgc-ablations/bwjxrrkx/checkpoints/epoch=307-step=352968.ckpt" \
  sbatch --time=22:00:00 "$JOB"

RUN_NAME=qm9-rgc-k4-l1e2-continued RGC_NUM_EXTRA_KERNELS=4 RGC_MIXING_L2=0.01 QM9_EPOCHS=500 \
  RESUME_CKPT="$RUNS_ROOT/qm9-rgc-k4-l1e2_20260517T221534Z_job22825576/artifacts/lightning/qm9-rgc-ablations/yovtwuab/checkpoints/epoch=291-step=334632.ckpt" \
  sbatch --time=22:00:00 "$JOB"

RUN_NAME=qm9-rgc-k4-l1e3-continued RGC_NUM_EXTRA_KERNELS=4 RGC_MIXING_L2=0.001 QM9_EPOCHS=500 \
  RESUME_CKPT="$RUNS_ROOT/qm9-rgc-k4-l1e3_20260517T223202Z_job22825925/artifacts/lightning/qm9-rgc-ablations/l2ekz6b3/checkpoints/epoch=309-step=355260.ckpt" \
  sbatch --time=22:00:00 "$JOB"

RUN_NAME=qm9-rgc-k4-l1e4-continued RGC_NUM_EXTRA_KERNELS=4 RGC_MIXING_L2=0.0001 QM9_EPOCHS=500 \
  RESUME_CKPT="$RUNS_ROOT/qm9-rgc-k4-l1e4_20260517T223201Z_job22825926/artifacts/lightning/qm9-rgc-ablations/rixry336/checkpoints/epoch=311-step=357552.ckpt" \
  sbatch --time=22:00:00 "$JOB"

RUN_NAME=qm9-rgc-k8-l1e2-continued RGC_NUM_EXTRA_KERNELS=8 RGC_MIXING_L2=0.01 QM9_EPOCHS=500 \
  RESUME_CKPT="$RUNS_ROOT/qm9-rgc-k8-l1e2_20260517T223201Z_job22825922/artifacts/lightning/qm9-rgc-ablations/oompiyxn/checkpoints/epoch=280-step=322026.ckpt" \
  sbatch --time=22:00:00 "$JOB"
