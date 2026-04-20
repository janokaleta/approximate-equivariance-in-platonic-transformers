import os
import sys
from pathlib import Path
from typing import Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ml_collections
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Timer
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from mains.main_qm9_regr import QM9Model
from platonic_transformers.datasets.qm9_approx_sym import (
    approx_sym_construction_stats_path,
    approx_sym_model_stats_path,
    make_qm9_approx_sym_datasets,
    resolve_qm9_target,
)
from platonic_transformers.utils.callbacks import StopOnPersistentDivergence, TimerCallback
from platonic_transformers.utils.config_loader import get_arg_parser, load_with_defaults, print_config


class QM9ApproxSymModel(QM9Model):
    """QM9 model with task-specific approximate-symmetry target statistics."""

    def set_dataset_statistics(self, dataloader: DataLoader) -> None:
        resolved_target, _ = resolve_qm9_target(self.config.dataset.target)
        stats_file = approx_sym_model_stats_path(
            cache_dir=self.config.dataset.cache_dir,
            target=resolved_target,
            break_strength=self.config.dataset.break_strength,
            views_per_molecule=self.config.dataset.views_per_molecule,
            split_seed=self.config.dataset.split_seed,
            rotation_seed=self.config.dataset.rotation_seed,
        )
        if stats_file.name.startswith("stats_"):
            raise ValueError("Approximate-symmetry task must not reuse baseline QM9 stats files.")

        if stats_file.exists():
            print(f"Loading approximate-symmetry target statistics from cached file: {stats_file}")
            stats = np.load(stats_file)
            self.shift = torch.tensor(stats["shift"])
            self.scale = torch.tensor(stats["scale"])
            self.avg_num_nodes = torch.tensor(stats["avg_num_nodes"])
        else:
            print("Computing approximate-symmetry target statistics...")
            ys = []
            total_num_nodes = 0
            for data in dataloader:
                ys.append(data.y)
                total_num_nodes += data.num_nodes
            ys = np.concatenate(ys)

            self.shift = torch.tensor(np.mean(ys))
            self.scale = torch.tensor(np.std(ys))
            self.avg_num_nodes = torch.tensor(total_num_nodes / len(dataloader.dataset))

            print(f"Saving approximate-symmetry target statistics to {stats_file}")
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(stats_file, shift=self.shift, scale=self.scale, avg_num_nodes=self.avg_num_nodes)

        print(f"Approximate target statistics - Mean: {self.shift:.4f}, Std: {self.scale:.4f}")


def _dataset_config(config: ml_collections.ConfigDict) -> dict[str, object]:
    resolved_target, _ = resolve_qm9_target(config.dataset.target)
    cache_dir = Path(config.dataset.cache_dir)
    return {
        "target": resolved_target,
        "break_strength": float(config.dataset.break_strength),
        "views_per_molecule": int(config.dataset.views_per_molecule),
        "split_seed": int(config.dataset.split_seed),
        "rotation_seed": int(config.dataset.rotation_seed),
        "train_size": int(getattr(config.dataset, "train_size", 110000)),
        "val_size": int(getattr(config.dataset, "val_size", 10000)),
        "construction_stats_path": approx_sym_construction_stats_path(
            cache_dir=cache_dir,
            target=resolved_target,
            break_strength=float(config.dataset.break_strength),
            views_per_molecule=int(config.dataset.views_per_molecule),
            split_seed=int(config.dataset.split_seed),
            rotation_seed=int(config.dataset.rotation_seed),
        ),
    }


def load_data(config: ml_collections.ConfigDict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    base_dataset = QM9(root=config.dataset.base_data_dir)
    datasets = make_qm9_approx_sym_datasets(base_dataset, **_dataset_config(config))

    dataloaders = {
        split: DataLoader(
            split_dataset,
            batch_size=config.training.batch_size,
            shuffle=(split == "train"),
            num_workers=config.system.num_workers,
        )
        for split, split_dataset in datasets.items()
    }
    return dataloaders["train"], dataloaders["val"], dataloaders["test"]


def main(config: ml_collections.ConfigDict) -> None:
    if config.training.train_augm:
        raise ValueError(
            "training.train_augm must remain false for qm9_approx_sym unless labels are recomputed after augmentation."
        )

    print_config(config, "QM9 Approximate-Symmetry Training Configuration")
    pl.seed_everything(config.seed)

    train_loader, val_loader, test_loader = load_data(config)

    if config.system.gpus > 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = config.system.gpus
    else:
        accelerator = "cpu"
        devices = "auto"

    if config.logging.enabled:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(
            project=config.logging.project_name,
            config=config.to_dict(),
            save_dir=save_dir,
        )
    else:
        logger = None

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            every_n_epochs=1,
            save_last=config.checkpoint.save_last,
        ),
        TimerCallback(),
    ]
    if config.logging.enabled:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))
    if config.system.timer is not None:
        callbacks.append(Timer(duration=config.system.timer))
    if config.callbacks.early_stopping.enabled:
        es_config = config.callbacks.early_stopping
        callbacks.append(StopOnPersistentDivergence(
            monitor=es_config.monitor,
            threshold=es_config.threshold,
            patience=es_config.patience,
            grace_epochs=es_config.grace_epochs,
            verbose=False,
        ))

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip_val,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=config.system.enable_progress_bar,
        precision=config.system.precision,
    )

    test_ckpt = config.testing.test_ckpt
    if test_ckpt is None:
        model = QM9ApproxSymModel(config)
        model.set_dataset_statistics(train_loader)
        trainer.fit(model, train_loader, val_loader, ckpt_path=config.testing.resume_ckpt)
        best_ckpt_path = callbacks[0].best_model_path if callbacks[0].best_model_path else "last"
        trainer.test(model, test_loader, ckpt_path=best_ckpt_path)
    else:
        model = QM9ApproxSymModel.load_from_checkpoint(test_ckpt)
        model.set_dataset_statistics(train_loader)
        trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = get_arg_parser(default_config_path="configs/qm9_approx_sym.yaml")
    args, unknown_args = parser.parse_known_args()
    config = load_with_defaults(dataset_config=args.config, cli_args=unknown_args)
    main(config)
