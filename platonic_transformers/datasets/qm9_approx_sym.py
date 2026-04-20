from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


QM9_TARGETS = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv",
    "U0_atom", "U_atom", "H_atom", "G_atom", "A", "B", "C",
]

QM9_ATOMIC_NUMBERS = torch.tensor([1, 6, 7, 8, 9], dtype=torch.float32)


@dataclass(frozen=True)
class ApproxSymStats:
    break_mean: float
    break_std: float
    target_std: float


def resolve_qm9_target(target: str) -> tuple[str, int]:
    """Resolve target names the same way as the baseline QM9 loader."""
    target_map = {name: i for i, name in enumerate(QM9_TARGETS)}
    atom_version = f"{target}_atom"
    resolved = atom_version if atom_version in target_map else target
    if resolved not in target_map:
        raise ValueError(f"Target '{target}' not found in QM9 targets: {QM9_TARGETS}")
    return resolved, target_map[resolved]


def qm9_split_indices(
    total_size: int,
    train_size: int = 110000,
    val_size: int = 10000,
    seed: int = 42,
) -> dict[str, Tensor]:
    """Return split-first QM9 indices using the DimeNet seed-42 permutation."""
    if total_size <= 0:
        raise ValueError("total_size must be positive.")
    if train_size <= 0 or val_size <= 0:
        raise ValueError("train_size and val_size must be positive.")
    if train_size + val_size >= total_size:
        raise ValueError("train_size + val_size must leave at least one test sample.")

    random_state = np.random.RandomState(seed=seed)
    perm = torch.from_numpy(random_state.permutation(np.arange(total_size))).long()
    return {
        "train": perm[:train_size],
        "val": perm[train_size:train_size + val_size],
        "test": perm[train_size + val_size:],
    }


def _format_float_for_filename(value: float) -> str:
    text = f"{float(value):.8g}"
    return text.replace("-", "m").replace(".", "p")


def approx_sym_construction_stats_path(
    cache_dir: str | os.PathLike[str],
    target: str,
    break_strength: float,
    views_per_molecule: int,
    split_seed: int,
    rotation_seed: int,
) -> Path:
    filename = (
        "qm9_approx_sym_construction_"
        f"target-{target}_"
        f"break-{_format_float_for_filename(break_strength)}_"
        f"views-{views_per_molecule}_"
        f"split-{split_seed}_"
        f"rot-{rotation_seed}.npz"
    )
    return Path(cache_dir) / filename


def approx_sym_model_stats_path(
    cache_dir: str | os.PathLike[str],
    target: str,
    break_strength: float,
    views_per_molecule: int,
    split_seed: int,
    rotation_seed: int,
) -> Path:
    filename = (
        "qm9_approx_sym_model_stats_"
        f"target-{target}_"
        f"break-{_format_float_for_filename(break_strength)}_"
        f"views-{views_per_molecule}_"
        f"split-{split_seed}_"
        f"rot-{rotation_seed}.npz"
    )
    return Path(cache_dir) / filename


def deterministic_so3_rotation(
    molecule_index: int,
    view_index: int,
    rotation_seed: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Generate a deterministic SO(3) rotation for one molecule/view pair."""
    seed = int(rotation_seed) + 1_000_003 * int(molecule_index) + 9_176 * int(view_index)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    q = torch.randn(4, generator=generator, dtype=torch.float64)
    q = q / q.norm().clamp_min(1e-12)
    w, x, y, z = q
    rotation = torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=torch.float64)
    return rotation.to(dtype=dtype, device=device)


def select_qm9_target(data, target_idx: int) -> Tensor:
    y = data.y.reshape(-1)
    if y.numel() == 1:
        return y[0].to(torch.float32)
    return y[target_idx].to(torch.float32)


def atomic_numbers_from_data(data) -> Tensor:
    if hasattr(data, "z") and data.z is not None:
        return data.z.to(dtype=torch.float32)
    if not hasattr(data, "x") or data.x is None or data.x.shape[-1] < 5:
        raise ValueError("QM9 data must have either z or x with first five atom-type channels.")

    atom_type = data.x[:, :5].to(torch.float32).argmax(dim=-1)
    return QM9_ATOMIC_NUMBERS.to(device=data.x.device)[atom_type]


def centered_rotated_positions(pos: Tensor, rotation: Tensor) -> Tensor:
    centered = pos.to(torch.float32) - pos.to(torch.float32).mean(dim=0, keepdim=True)
    return centered @ rotation.to(device=pos.device, dtype=torch.float32).T


def lab_frame_breaking_term(data, rotation: Tensor, eps: float = 1e-8) -> Tensor:
    """Weak external-field proxy: charge-weighted z projection after deterministic rotation."""
    centered = data.pos.to(torch.float32) - data.pos.to(torch.float32).mean(dim=0, keepdim=True)
    rotated = centered @ rotation.to(device=data.pos.device, dtype=torch.float32).T
    z = atomic_numbers_from_data(data).to(device=rotated.device)
    rms_radius = centered.square().sum(dim=-1).mean().sqrt()
    numerator = (z * rotated[:, 2]).sum()
    denominator = z.sum().clamp_min(eps) * rms_radius.clamp_min(eps)
    return numerator / denominator


def compute_approx_sym_stats(
    base_dataset: Dataset,
    train_indices: Sequence[int] | Tensor,
    target_idx: int,
    views_per_molecule: int,
    rotation_seed: int,
) -> ApproxSymStats:
    targets = []
    terms = []
    for base_index in torch.as_tensor(train_indices, dtype=torch.long).tolist():
        data = base_dataset[int(base_index)]
        targets.append(select_qm9_target(data, target_idx))
        for view_index in range(views_per_molecule):
            rotation = deterministic_so3_rotation(base_index, view_index, rotation_seed)
            terms.append(lab_frame_breaking_term(data, rotation))

    target_values = torch.stack(targets).to(torch.float64)
    term_values = torch.stack(terms).to(torch.float64)
    target_std = target_values.std(unbiased=False).item()
    break_std = term_values.std(unbiased=False).item()
    return ApproxSymStats(
        break_mean=term_values.mean().item(),
        break_std=break_std if break_std > 1e-12 else 1.0,
        target_std=target_std if target_std > 1e-12 else 1.0,
    )


def load_or_compute_approx_sym_stats(
    base_dataset: Dataset,
    train_indices: Sequence[int] | Tensor,
    target_idx: int,
    views_per_molecule: int,
    rotation_seed: int,
    stats_path: str | os.PathLike[str] | None = None,
) -> ApproxSymStats:
    if stats_path is not None:
        stats_path = Path(stats_path)
        if stats_path.name.startswith("stats_"):
            raise ValueError("Approximate-symmetry construction stats must not reuse baseline stats_{target}.npz files.")
        if stats_path.exists():
            stats = np.load(stats_path)
            return ApproxSymStats(
                break_mean=float(stats["break_mean"]),
                break_std=float(stats["break_std"]),
                target_std=float(stats["target_std"]),
            )

    stats = compute_approx_sym_stats(
        base_dataset=base_dataset,
        train_indices=train_indices,
        target_idx=target_idx,
        views_per_molecule=views_per_molecule,
        rotation_seed=rotation_seed,
    )
    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            stats_path,
            break_mean=stats.break_mean,
            break_std=stats.break_std,
            target_std=stats.target_std,
        )
    return stats


class QM9ApproxSymDataset(Dataset):
    """Split-safe deterministic QM9 view dataset with a controlled approximate-symmetry target."""

    def __init__(
        self,
        base_dataset: Dataset,
        base_indices: Sequence[int] | Tensor,
        *,
        target_idx: int,
        break_strength: float,
        views_per_molecule: int,
        rotation_seed: int,
        stats: ApproxSymStats,
    ) -> None:
        if views_per_molecule <= 0:
            raise ValueError("views_per_molecule must be positive.")
        self.base_dataset = base_dataset
        self.base_indices = torch.as_tensor(base_indices, dtype=torch.long)
        self.target_idx = int(target_idx)
        self.break_strength = float(break_strength)
        self.views_per_molecule = int(views_per_molecule)
        self.rotation_seed = int(rotation_seed)
        self.stats = stats

    def __len__(self) -> int:
        return int(self.base_indices.numel() * self.views_per_molecule)

    def __getitem__(self, index: int):
        local_molecule_index = int(index) // self.views_per_molecule
        view_index = int(index) % self.views_per_molecule
        base_index = int(self.base_indices[local_molecule_index])

        data = self.base_dataset[base_index].clone()
        rotation = deterministic_so3_rotation(
            molecule_index=base_index,
            view_index=view_index,
            rotation_seed=self.rotation_seed,
            dtype=torch.float32,
            device=data.pos.device,
        )
        base_target = select_qm9_target(data, self.target_idx)
        breaking_term = lab_frame_breaking_term(data, rotation)
        normalized_term = (breaking_term - self.stats.break_mean) / self.stats.break_std
        target = base_target + self.break_strength * self.stats.target_std * normalized_term

        data.pos = centered_rotated_positions(data.pos, rotation)
        data.y = target.reshape(1)
        data.approx_sym_base_idx = torch.tensor(base_index, dtype=torch.long)
        data.approx_sym_view_idx = torch.tensor(view_index, dtype=torch.long)
        data.approx_sym_base_y = base_target.reshape(1)
        data.approx_sym_breaking_term = breaking_term.reshape(1)
        return data


def make_qm9_approx_sym_datasets(
    base_dataset: Dataset,
    *,
    target: str,
    break_strength: float,
    views_per_molecule: int,
    split_seed: int = 42,
    rotation_seed: int = 0,
    train_size: int = 110000,
    val_size: int = 10000,
    construction_stats_path: str | os.PathLike[str] | None = None,
) -> Mapping[str, QM9ApproxSymDataset]:
    resolved_target, target_idx = resolve_qm9_target(target)
    split_indices = qm9_split_indices(
        total_size=len(base_dataset),
        train_size=train_size,
        val_size=val_size,
        seed=split_seed,
    )
    stats = load_or_compute_approx_sym_stats(
        base_dataset=base_dataset,
        train_indices=split_indices["train"],
        target_idx=target_idx,
        views_per_molecule=views_per_molecule,
        rotation_seed=rotation_seed,
        stats_path=construction_stats_path,
    )
    return {
        split: QM9ApproxSymDataset(
            base_dataset=base_dataset,
            base_indices=indices,
            target_idx=target_idx,
            break_strength=break_strength,
            views_per_molecule=views_per_molecule,
            rotation_seed=rotation_seed,
            stats=stats,
        )
        for split, indices in split_indices.items()
    }
