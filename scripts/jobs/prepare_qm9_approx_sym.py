#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from torch_geometric.datasets import QM9

from platonic_transformers.datasets.qm9_approx_sym import (
    approx_sym_construction_stats_path,
    approx_sym_model_stats_path,
    make_qm9_approx_sym_datasets,
    resolve_qm9_target,
)


def ensure_model_stats(train_dataset, stats_path: Path) -> Path:
    if stats_path.exists():
        print(f"Approximate-symmetry model stats already exist: {stats_path}")
        return stats_path

    print(f"Computing approximate-symmetry model stats: {stats_path}")
    ys = []
    total_num_nodes = 0
    for data in train_dataset:
        ys.append(data.y.cpu().numpy())
        total_num_nodes += data.num_nodes

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = stats_path.with_suffix(f".tmp.{os.getpid()}.npz")
    np.savez(
        tmp_path,
        shift=np.mean(np.concatenate(ys)),
        scale=np.std(np.concatenate(ys)),
        avg_num_nodes=total_num_nodes / len(train_dataset),
    )
    os.replace(tmp_path, stats_path)
    print(f"Wrote approximate-symmetry model stats: {stats_path}")
    return stats_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-stage QM9 approximate-symmetry assets.")
    parser.add_argument("--data-dir", required=True, help="Base QM9 root directory.")
    parser.add_argument("--cache-dir", required=True, help="Approximate-symmetry cache directory.")
    parser.add_argument("--target", default="mu", help="QM9 target name.")
    parser.add_argument("--break-strength", type=float, default=0.10)
    parser.add_argument("--views-per-molecule", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--rotation-seed", type=int, default=1729)
    parser.add_argument("--train-size", type=int, default=110000)
    parser.add_argument("--val-size", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    resolved_target, _ = resolve_qm9_target(args.target)
    base_dataset = QM9(root=str(data_dir))
    print(f"QM9 base dataset ready at {data_dir} with {len(base_dataset)} molecules.")

    construction_stats = approx_sym_construction_stats_path(
        cache_dir=cache_dir,
        target=resolved_target,
        break_strength=args.break_strength,
        views_per_molecule=args.views_per_molecule,
        split_seed=args.split_seed,
        rotation_seed=args.rotation_seed,
    )
    model_stats = approx_sym_model_stats_path(
        cache_dir=cache_dir,
        target=resolved_target,
        break_strength=args.break_strength,
        views_per_molecule=args.views_per_molecule,
        split_seed=args.split_seed,
        rotation_seed=args.rotation_seed,
    )

    datasets = make_qm9_approx_sym_datasets(
        base_dataset,
        target=resolved_target,
        break_strength=args.break_strength,
        views_per_molecule=args.views_per_molecule,
        split_seed=args.split_seed,
        rotation_seed=args.rotation_seed,
        train_size=args.train_size,
        val_size=args.val_size,
        construction_stats_path=construction_stats,
    )
    ensure_model_stats(datasets["train"], model_stats)
    print("QM9 approximate-symmetry assets ready.")


if __name__ == "__main__":
    main()
