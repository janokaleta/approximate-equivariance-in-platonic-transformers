#!/usr/bin/env python
"""
Post-hoc RGC metric extraction from a trained checkpoint.

Reports:
  1. Equivariance (invariance) error — mean |f(R_g x) - f(x)| / |f(x)| over all group elements
  2. Mean ‖relaxed_mixing‖ per layer — indicates how much symmetry-breaking capacity was used
  3. Parameter delta vs baseline — extra parameters introduced by RGC

Usage:
  # Metrics 2 & 3 only (no data needed, runs on CPU):
  python scripts/eval_rgc_metrics.py <checkpoint.ckpt>

  # All three metrics (requires QM9 data dir):
  python scripts/eval_rgc_metrics.py <checkpoint.ckpt> --data_dir /path/to/qm9
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _section(title):
    print(f"\n{'=' * 55}\n  {title}\n{'=' * 55}")


def metrics_from_state_dict(state):
    mixing_norms = {k: v.norm().item() for k, v in state.items() if "relaxed_mixing" in k}
    relaxed_count = sum(v.numel() for k, v in state.items() if k.startswith("net.") and "relaxed_" in k)
    total_count = sum(v.numel() for k, v in state.items() if k.startswith("net."))
    return mixing_norms, relaxed_count, total_count


def compute_equivariance_error(model, test_loader, n_batches, device):
    """
    Measures invariance error for each of the 24 octahedral group elements.
    The model predicts a scalar (mu), so equivariance collapses to invariance:
      err_g = mean_molecules |f(R_g @ pos) - f(pos)| / (|f(pos)| + 1e-8)
    """
    from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS

    rotations = PLATONIC_GROUPS["octahedron"].elements.to(device=device, dtype=torch.float32)
    G = rotations.shape[0]

    model.eval().to(device)
    per_g_errors = torch.zeros(G)
    batches_used = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= n_batches:
                break
            batch = batch.to(device)
            orig_pos = batch.pos.clone()

            baseline = model(batch)

            for g, R in enumerate(rotations):
                batch.pos = orig_pos @ R.T
                rotated_out = model(batch)
                err = (rotated_out - baseline).abs() / (baseline.abs() + 1e-8)
                per_g_errors[g] += err.mean().item()

            batch.pos = orig_pos
            batches_used += 1

    return per_g_errors / max(batches_used, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .ckpt file")
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Path to QM9 data directory. Required for equivariance error (metric 1).",
    )
    parser.add_argument("--target", default="mu")
    parser.add_argument(
        "--n_batches",
        type=int,
        default=20,
        help="Number of test batches to use for equivariance error.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    print(f"\nCheckpoint : {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["state_dict"]

    # ── Metric 3: parameter delta ─────────────────────────────────────────────
    _section("Metric 3 — Parameter Delta vs Baseline")
    mixing_norms, relaxed_count, total_count = metrics_from_state_dict(state)
    baseline_count = total_count - relaxed_count
    pct = 100 * relaxed_count / baseline_count if baseline_count > 0 else 0
    print(f"  Total parameters:    {total_count:>12,}")
    print(f"  Baseline parameters: {baseline_count:>12,}")
    print(f"  Relaxed parameters:  {relaxed_count:>12,}  (+{pct:.2f}%)")

    # ── Metric 2: ‖relaxed_mixing‖ ───────────────────────────────────────────
    _section("Metric 2 — Relaxed Mixing Norms  ‖A‖")
    if not mixing_norms:
        print("  No relaxed_mixing parameters found (baseline checkpoint).")
    else:
        for k, v in mixing_norms.items():
            # shorten key: "net.blocks.3.linear1" style
            parts = k.split(".")
            label = ".".join(parts[1:parts.index("relaxed_mixing")])
            print(f"  {label:<48} {v:.6f}")
        mean_norm = sum(mixing_norms.values()) / len(mixing_norms)
        print(f"\n  Layers counted:      {len(mixing_norms)}")
        print(f"  Mean ‖A‖ per layer:  {mean_norm:.6f}")
        print(f"  Sum  ‖A‖ all layers: {sum(mixing_norms.values()):.6f}")

    # ── Metric 1: equivariance error ─────────────────────────────────────────
    _section("Metric 1 — Equivariance (Invariance) Error")
    if args.data_dir is None:
        print("  Skipped — pass --data_dir /path/to/qm9 to enable.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    from mains.main_qm9_regr import QM9Model
    from torch_geometric.datasets import QM9
    from torch_geometric.loader import DataLoader

    model = QM9Model.load_from_checkpoint(args.checkpoint, map_location=device)

    # avg_num_nodes is not a registered buffer — load from the cached stats file
    stats_file = os.path.join(args.data_dir, f"stats_{args.target}.npz")
    if not os.path.exists(stats_file):
        print(f"  Stats file not found: {stats_file}")
        print("  Run training first so the stats file is created, then retry.")
        return
    stats = np.load(stats_file)
    model.avg_num_nodes = torch.tensor(float(stats["avg_num_nodes"]))

    # Reconstruct the identical test split used during training
    all_targets = [
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
        "U0", "U", "H", "G", "Cv",
        "U0_atom", "U_atom", "H_atom", "G_atom", "A", "B", "C",
    ]
    dataset = QM9(root=args.data_dir)
    dataset.data.y = dataset.data.y[:, all_targets.index(args.target)]
    rng = np.random.RandomState(42)
    perm = torch.from_numpy(rng.permutation(np.arange(130831)))
    test_loader = DataLoader(
        dataset[perm[120000:]], batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    print(f"  Batches used:   {args.n_batches}  (x{args.batch_size} molecules)")
    per_g = compute_equivariance_error(model, test_loader, args.n_batches, device)

    print(f"  Group:          octahedron (G=24)")
    print(f"  Mean error:     {per_g.mean().item():.3e}  (average over all 24 elements)")
    print(f"  Max  error:     {per_g.max().item():.3e}  (worst element)")
    print(f"  Min  error:     {per_g.min().item():.3e}  (best element)")
    print()
    print("  Reference: equivariant baseline ≈ 1e-7 (numerical precision only)")
    print("  RGC model: > 0 reflects learned symmetry breaking")


if __name__ == "__main__":
    main()
