import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from platonic_transformers.datasets.qm9_approx_sym import (
    approx_sym_construction_stats_path,
    approx_sym_model_stats_path,
    deterministic_so3_rotation,
    make_qm9_approx_sym_datasets,
    qm9_split_indices,
    select_qm9_target,
)


class TinyQM9Dataset:
    def __init__(self):
        self.data = []
        atom_types = torch.eye(5)
        for idx in range(8):
            scale = 1.0 + 0.1 * idx
            pos = torch.tensor([
                [0.0, 0.0, 0.0],
                [scale, 0.2, 0.1],
                [0.1, 0.8 * scale, 0.3],
                [0.3, 0.2, 0.9 * scale],
            ], dtype=torch.float32)
            x = atom_types[torch.tensor([0, 1, 2, 3])]
            y = torch.arange(19, dtype=torch.float32) + float(idx)
            self.data.append(Data(pos=pos, x=x, z=torch.tensor([1, 6, 7, 8]), y=y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[int(index)].clone()


class QM9ApproxSymDatasetTest(unittest.TestCase):
    def make_datasets(self, *, break_strength=0.1, rotation_seed=17, cache_dir=None):
        stats_path = None
        if cache_dir is not None:
            stats_path = approx_sym_construction_stats_path(
                cache_dir,
                target="mu",
                break_strength=break_strength,
                views_per_molecule=3,
                split_seed=3,
                rotation_seed=rotation_seed,
            )
        return make_qm9_approx_sym_datasets(
            TinyQM9Dataset(),
            target="mu",
            break_strength=break_strength,
            views_per_molecule=3,
            split_seed=3,
            rotation_seed=rotation_seed,
            train_size=4,
            val_size=2,
            construction_stats_path=stats_path,
        )

    def test_split_first_view_expansion_prevents_leakage(self):
        datasets = self.make_datasets()
        split_sets = {
            split: set(dataset.base_indices.tolist())
            for split, dataset in datasets.items()
        }

        self.assertTrue(split_sets["train"].isdisjoint(split_sets["val"]))
        self.assertTrue(split_sets["train"].isdisjoint(split_sets["test"]))
        self.assertTrue(split_sets["val"].isdisjoint(split_sets["test"]))
        self.assertEqual(len(datasets["train"]), 4 * 3)
        self.assertEqual(len(datasets["val"]), 2 * 3)
        self.assertEqual(len(datasets["test"]), 2 * 3)

    def test_break_strength_zero_recovers_base_target_for_all_views(self):
        base_dataset = TinyQM9Dataset()
        datasets = self.make_datasets(break_strength=0.0)

        for dataset in datasets.values():
            for item in dataset:
                base_idx = int(item.approx_sym_base_idx)
                expected = select_qm9_target(base_dataset[base_idx], dataset.target_idx)
                torch.testing.assert_close(item.y.reshape(()), expected, atol=0.0, rtol=0.0)

    def test_view_generation_is_deterministic_for_same_seed(self):
        first = self.make_datasets(rotation_seed=123)["train"]
        second = self.make_datasets(rotation_seed=123)["train"]

        for index in range(len(first)):
            first_item = first[index]
            second_item = second[index]
            torch.testing.assert_close(first_item.pos, second_item.pos)
            torch.testing.assert_close(first_item.y, second_item.y)

    def test_rotation_seed_changes_approximate_target(self):
        first = self.make_datasets(rotation_seed=123)["train"][0]
        second = self.make_datasets(rotation_seed=124)["train"][0]

        self.assertFalse(torch.allclose(first.y, second.y))

    def test_construction_stats_are_cached_under_task_specific_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets = self.make_datasets(cache_dir=tmp_dir)
            stats_path = approx_sym_construction_stats_path(
                tmp_dir,
                target="mu",
                break_strength=0.1,
                views_per_molecule=3,
                split_seed=3,
                rotation_seed=17,
            )

            self.assertTrue(stats_path.exists())
            self.assertNotEqual(stats_path.name, "stats_mu.npz")
            cached = self.make_datasets(cache_dir=tmp_dir)
            torch.testing.assert_close(datasets["train"][0].y, cached["train"][0].y)

    def test_model_stats_path_does_not_reuse_baseline_stats_name(self):
        path = approx_sym_model_stats_path(
            "/tmp/cache",
            target="mu",
            break_strength=0.1,
            views_per_molecule=2,
            split_seed=42,
            rotation_seed=1729,
        )

        self.assertNotEqual(path.name, "stats_mu.npz")
        self.assertTrue(path.name.startswith("qm9_approx_sym_model_stats_"))

    def test_config_disables_rotation_augmentation_by_default(self):
        config_path = Path(REPO_ROOT) / "configs" / "qm9_approx_sym.yaml"
        config = yaml.safe_load(config_path.read_text())

        self.assertEqual(config["dataset"]["name"], "qm9_approx_sym")
        self.assertFalse(config["training"]["train_augm"])

    def test_deterministic_rotation_is_valid_so3_matrix(self):
        rotation = deterministic_so3_rotation(2, 1, 123)

        torch.testing.assert_close(rotation.T @ rotation, torch.eye(3), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.det(rotation), torch.tensor(1.0), atol=1e-6, rtol=1e-6)

    def test_split_indices_requires_test_holdout(self):
        with self.assertRaisesRegex(ValueError, "leave at least one test sample"):
            qm9_split_indices(total_size=6, train_size=4, val_size=2, seed=1)


if __name__ == "__main__":
    unittest.main()
