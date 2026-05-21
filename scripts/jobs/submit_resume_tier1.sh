#!/usr/bin/env bash
# Resume all 7 Tier-1 RGC runs from their original checkpoints, training to epoch 500.
#
# Generates a self-contained temporary job script for each run via sed substitution
# so no environment variables need to be forwarded through SLURM.
#
# Usage (from repo root on Snellius):  bash scripts/jobs/submit_resume_tier1.sh

set -euo pipefail

RUNS_ROOT="/scratch-shared/scur0203/platonic-transformers/runs"

CKPT_K1_L1E2="$RUNS_ROOT/qm9-rgc-k1-l1e2_20260517T223159Z_job22825920/artifacts/lightning/qm9-rgc-ablations/or0zlylq/checkpoints/epoch=346-step=397662.ckpt"
CKPT_K2_L1E2="$RUNS_ROOT/qm9-rgc-k2-l1e2_20260517T223201Z_job22825921/artifacts/lightning/qm9-rgc-ablations/rn8v0ion/checkpoints/epoch=296-step=340362.ckpt"
CKPT_K4_L1E1="$RUNS_ROOT/qm9-rgc-k4-l1e1_20260517T223201Z_job22825923/artifacts/lightning/qm9-rgc-ablations/bwjxrrkx/checkpoints/epoch=307-step=352968.ckpt"
CKPT_K4_L1E2="$RUNS_ROOT/qm9-rgc-k4-l1e2_20260517T221534Z_job22825576/artifacts/lightning/qm9-rgc-ablations/yovtwuab/checkpoints/epoch=291-step=334632.ckpt"
CKPT_K4_L1E3="$RUNS_ROOT/qm9-rgc-k4-l1e3_20260517T223202Z_job22825925/artifacts/lightning/qm9-rgc-ablations/l2ekz6b3/checkpoints/epoch=309-step=355260.ckpt"
CKPT_K4_L1E4="$RUNS_ROOT/qm9-rgc-k4-l1e4_20260517T223201Z_job22825926/artifacts/lightning/qm9-rgc-ablations/rixry336/checkpoints/epoch=311-step=357552.ckpt"
CKPT_K8_L1E2="$RUNS_ROOT/qm9-rgc-k8-l1e2_20260517T223201Z_job22825922/artifacts/lightning/qm9-rgc-ablations/oompiyxn/checkpoints/epoch=280-step=322026.ckpt"

# ── Pre-flight ─────────────────────────────────────────────────────────────────
echo "Verifying checkpoints..."
all_ok=1
for ckpt in "$CKPT_K1_L1E2" "$CKPT_K2_L1E2" "$CKPT_K4_L1E1" "$CKPT_K4_L1E2" \
            "$CKPT_K4_L1E3" "$CKPT_K4_L1E4" "$CKPT_K8_L1E2"; do
  if [[ -f "$ckpt" ]]; then
    echo "  OK  $(basename "$ckpt")"
  else
    echo "  MISSING  $ckpt"
    all_ok=0
  fi
done
[[ "$all_ok" -eq 1 ]] || { echo "Aborting — fix missing checkpoints above."; exit 1; }
echo ""

# ── submit() ──────────────────────────────────────────────────────────────────
# Writes a temporary job script with @@PLACEHOLDERS@@ replaced by sed,
# submits it, then deletes it. Nothing is forwarded through SLURM environment.
submit() {
  local name="$1" k="$2" l2="$3" ckpt="$4"
  local tmpjob
  tmpjob="$(mktemp /tmp/rgc_XXXXXX.job)"

  # The heredoc is quoted (<<'TEMPLATE') so bash does not expand anything.
  # sed then substitutes the four placeholders with the actual values.
  sed \
    -e "s|@@NAME@@|${name}|g" \
    -e "s|@@K@@|${k}|g" \
    -e "s|@@L2@@|${l2}|g" \
    -e "s|@@CKPT@@|${ckpt}|g" \
    > "$tmpjob" \
    <<'TEMPLATE'
#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=PT_QM9_RGC
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=22:00:00
#SBATCH --output=./scripts/jobs/out/%x_%A.out

set -euo pipefail

module purge
module load 2025

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR}/scripts/jobs/_job_common.sh"

resolve_repo_root
load_dotenv_if_present "$REPO_ROOT/.env"
setup_cluster_layout
ensure_uv
initialize_run_context "@@NAME@@"
ensure_shared_env
stage_repo_snapshot
link_shared_env_into_stage
write_run_metadata
print_runtime_diagnostics
validate_cuda_runtime

require_env_var "QM9_DATA_DIR"
ensure_qm9_dataset_ready
ensure_qm9_stats_ready "mu"

declare -a qm9_args=(
  "qm9_regr"
  "--config" "configs/qm9_regr.yaml"
  "--data_dir" "$QM9_DATA_DIR"
  "--target" "mu"
  "--batch_size" "96"
  "--epochs" "500"
  "--gpus" "1"
  "--num_workers" "16"
  "--precision" "32"
  "--model.relaxed_group_convolution.enabled" "true"
  "--model.relaxed_group_convolution.num_extra_kernels" "@@K@@"
  "--model.relaxed_group_convolution.scale" "1.0"
  "--model.relaxed_group_convolution.kernel_init" "normal"
  "--model.relaxed_group_convolution.mixing_init" "zeros"
  "--model.relaxed_group_convolution.mixing_l1" "0.0"
  "--model.relaxed_group_convolution.mixing_l2" "@@L2@@"
  "--model.relaxed_group_convolution.kernel_l2" "0.0"
  "--testing.resume_ckpt" "@@CKPT@@"
)

(
  cd "$STAGED_REPO_ROOT"
  log "Launching @@NAME@@ — resuming from @@CKPT@@"
  srun uv run --no-sync python meta_main.py "${qm9_args[@]}"
)
TEMPLATE

  local job_id
  job_id=$(sbatch --parsable "$tmpjob")
  rm -f "$tmpjob"
  echo "  Submitted @@NAME@@ → job ${job_id}" | sed -e "s|@@NAME@@|${name}|g"
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
echo "Done. Monitor: squeue -u \$USER"
echo "Logs:  scripts/jobs/out/PT_QM9_RGC_<jobid>.out"
