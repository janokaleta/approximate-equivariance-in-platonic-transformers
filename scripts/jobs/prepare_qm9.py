#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import QM9


ALL_TARGETS = [
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
    "U0_atom",
    "U_atom",
    "H_atom",
    "G_atom",
    "A",
    "B",
    "C",
]


def resolve_target_name(target: str) -> str:
    atom_variant = f"{target}_atom"
    if atom_variant in ALL_TARGETS:
        return atom_variant
    if target in ALL_TARGETS:
        return target
    raise ValueError(f"Unknown QM9 target '{target}'. Expected one of: {ALL_TARGETS}")


def ensure_dataset(data_dir: Path) -> QM9:
    dataset = QM9(root=str(data_dir))
    print(f"QM9 dataset ready at {data_dir} with {len(dataset)} molecules.")
    return dataset


def ensure_stats(dataset: QM9, data_dir: Path, requested_target: str) -> Path:
    target_name = resolve_target_name(requested_target)
    target_index = {name: idx for idx, name in enumerate(ALL_TARGETS)}[target_name]
    stats_path = data_dir / f"stats_{requested_target}.npz"

    if stats_path.exists():
        print(f"Stats already exist: {stats_path}")
        return stats_path

    target_tensor = dataset.data.y[:, target_index]

    random_state = np.random.RandomState(seed=42)
    permutation = torch.from_numpy(random_state.permutation(np.arange(130831)))
    train_idx = permutation[:110000]
    train_dataset = dataset[train_idx]

    targets = target_tensor[train_idx].cpu().numpy()
    avg_num_nodes = sum(int(data.num_nodes) for data in train_dataset) / len(train_dataset)

    tmp_path = stats_path.with_suffix(f".tmp.{os.getpid()}.npz")
    np.savez(
        tmp_path,
        shift=np.mean(targets),
        scale=np.std(targets),
        avg_num_nodes=avg_num_nodes,
    )
    os.replace(tmp_path, stats_path)
    print(f"Wrote stats for target '{requested_target}' to {stats_path}")
    return stats_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-stage QM9 dataset assets for cluster jobs.")
    parser.add_argument("--data-dir", required=True, help="Target QM9 root directory.")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="QM9 target name to precompute stats for. Repeat to compute multiple targets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = ensure_dataset(data_dir)
    for target in args.target:
        ensure_stats(dataset, data_dir, target)


if __name__ == "__main__":
    main()
