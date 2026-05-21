#!/usr/bin/env bash
# Submit 7 continued Tier-1 RGC training runs (resuming from original run checkpoints).
# Each job is submitted in an isolated subshell so exports cannot bleed between submissions.
#
# Usage (from repo root on Snellius):
#   bash scripts/jobs/submit_resume_tier1.sh

set -euo pipefail

JOB="scripts/jobs/train_qm9_relaxed_group_convolution.job"
RUNS_ROOT="/scratch-shared/scur0203/platonic-transformers/runs"

# Exact checkpoint files from the original (round-1) runs.
# Using epoch-named files rather than the last.ckpt alias for reliability.
CKPT_K1_L1E2="$RUNS_ROOT/qm9-rgc-k1-l1e2_20260517T223159Z_job22825920/artifacts/lightning/qm9-rgc-ablations/or0zlylq/checkpoints/epoch=346-step=397662.ckpt"
CKPT_K2_L1E2="$RUNS_ROOT/qm9-rgc-k2-l1e2_20260517T223201Z_job22825921/artifacts/lightning/qm9-rgc-ablations/rn8v0ion/checkpoints/epoch=296-step=340362.ckpt"
CKPT_K4_L1E1="$RUNS_ROOT/qm9-rgc-k4-l1e1_20260517T223201Z_job22825923/artifacts/lightning/qm9-rgc-ablations/bwjxrrkx/checkpoints/epoch=307-step=352968.ckpt"
CKPT_K4_L1E2="$RUNS_ROOT/qm9-rgc-k4-l1e2_20260517T221534Z_job22825576/artifacts/lightning/qm9-rgc-ablations/yovtwuab/checkpoints/epoch=291-step=334632.ckpt"
CKPT_K4_L1E3="$RUNS_ROOT/qm9-rgc-k4-l1e3_20260517T223202Z_job22825925/artifacts/lightning/qm9-rgc-ablations/l2ekz6b3/checkpoints/epoch=309-step=355260.ckpt"
CKPT_K4_L1E4="$RUNS_ROOT/qm9-rgc-k4-l1e4_20260517T223201Z_job22825926/artifacts/lightning/qm9-rgc-ablations/rixry336/checkpoints/epoch=311-step=357552.ckpt"
CKPT_K8_L1E2="$RUNS_ROOT/qm9-rgc-k8-l1e2_20260517T223201Z_job22825922/artifacts/lightning/qm9-rgc-ablations/oompiyxn/checkpoints/epoch=280-step=322026.ckpt"

# ── Pre-flight: verify every checkpoint exists before submitting anything ──────
echo "Verifying checkpoints..."
all_ok=1
for ckpt in "$CKPT_K1_L1E2" "$CKPT_K2_L1E2" "$CKPT_K4_L1E1" "$CKPT_K4_L1E2" \
            "$CKPT_K4_L1E3" "$CKPT_K4_L1E4" "$CKPT_K8_L1E2"; do
  if [[ -f "$ckpt" ]]; then
    echo "  OK  $ckpt"
  else
    echo "  MISSING  $ckpt"
    all_ok=0
  fi
done

if [[ "$all_ok" -eq 0 ]]; then
  echo ""
  echo "ERROR: one or more checkpoints are missing. Fix paths above before submitting."
  exit 1
fi

echo ""
echo "All checkpoints verified. Submitting 7 jobs..."
echo ""

# ── Submission helpers ─────────────────────────────────────────────────────────

submit() {
  local name="$1" k="$2" l2="$3" ckpt="$4"
  (
    export RUN_NAME="$name"
    export RGC_NUM_EXTRA_KERNELS="$k"
    export RGC_MIXING_L2="$l2"
    export QM9_EPOCHS=500
    export QM9_EXTRA_ARGS="--testing.resume_ckpt $ckpt"
    job_id=$(sbatch --time=22:00:00 --parsable "$JOB")
    echo "  Submitted $name  →  job $job_id  (resume from epoch $(basename "$ckpt" | grep -o 'epoch=[0-9]*' | cut -d= -f2))"
  )
}

# ── Submit all 7 ──────────────────────────────────────────────────────────────
submit "qm9-rgc-k1-l1e2-continued"  1  0.01    "$CKPT_K1_L1E2"
submit "qm9-rgc-k2-l1e2-continued"  2  0.01    "$CKPT_K2_L1E2"
submit "qm9-rgc-k4-l1e1-continued"  4  0.1     "$CKPT_K4_L1E1"
submit "qm9-rgc-k4-l1e2-continued"  4  0.01    "$CKPT_K4_L1E2"
submit "qm9-rgc-k4-l1e3-continued"  4  0.001   "$CKPT_K4_L1E3"
submit "qm9-rgc-k4-l1e4-continued"  4  0.0001  "$CKPT_K4_L1E4"
submit "qm9-rgc-k8-l1e2-continued"  8  0.01    "$CKPT_K8_L1E2"

echo ""
echo "Done. Check status with:  squeue -u \$USER"
echo "Log files will appear in: scripts/jobs/out/"
