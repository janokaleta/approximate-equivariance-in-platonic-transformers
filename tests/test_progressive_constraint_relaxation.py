"""Smoke tests for progressive constraint relaxation.

Run:
    uv run python tests/test_progressive_constraint_relaxation.py
"""

import os
import sys
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from platonic_transformers.models.platoformer.block import PlatonicBlock
from platonic_transformers.models.platoformer.platoformer import (
    PlatonicTransformer,
    constraint_relaxation_progress_for_epoch,
    constraint_relaxation_scale_for_progress,
)


def make_block(**kwargs):
    defaults = dict(
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        solid_name="cyclic_4",
        dropout=0.0,
        spatial_dims=2,
        freq_sigma=None,
        learned_freqs=False,
        attention=False,
    )
    defaults.update(kwargs)
    return PlatonicBlock(**defaults)


class ProgressiveConstraintRelaxationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = torch.randn(2, 3, 16)
        self.pos = torch.randn(2, 3, 2)
        self.mask = torch.ones(2, 3, dtype=torch.bool)

    def test_disabled_matches_baseline_state_and_output(self):
        torch.manual_seed(7)
        baseline = make_block()
        torch.manual_seed(7)
        disabled = make_block(relaxation_scale=0.0, relaxation_apply_to="both")
        baseline.eval()
        disabled.eval()

        self.assertEqual(set(baseline.state_dict().keys()), set(disabled.state_dict().keys()))
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, disabled.state_dict()[key]), key)

        with torch.no_grad():
            y_baseline = baseline(self.x, self.pos, mask=self.mask)
            y_disabled = disabled(self.x, self.pos, mask=self.mask)
        self.assertTrue(torch.allclose(y_baseline, y_disabled, atol=0.0, rtol=0.0))

    def test_enabled_starts_from_baseline_but_has_extra_parameters(self):
        torch.manual_seed(11)
        baseline = make_block()
        torch.manual_seed(11)
        enabled = make_block(relaxation_scale=1.0, relaxation_apply_to="both")
        baseline.eval()
        enabled.eval()

        self.assertTrue(any(key.startswith("relaxation_") for key in enabled.state_dict()))
        with torch.no_grad():
            y_baseline = baseline(self.x, self.pos, mask=self.mask)
            y_enabled = enabled(self.x, self.pos, mask=self.mask)
        self.assertTrue(torch.allclose(y_baseline, y_enabled, atol=0.0, rtol=0.0))

    def test_non_equivariant_branch_is_removed_when_scale_reaches_zero(self):
        torch.manual_seed(13)
        block = make_block(relaxation_scale=1.0, relaxation_apply_to="interaction")
        block.eval()

        with torch.no_grad():
            y_baseline = block(self.x, self.pos, mask=self.mask)
            block.relaxation_interaction.weight.fill_(0.02)
            block.relaxation_interaction.bias.fill_(0.01)
            y_relaxed = block(self.x, self.pos, mask=self.mask)
            block.set_relaxation_scale(0.0)
            y_strict = block(self.x, self.pos, mask=self.mask)

        self.assertFalse(torch.allclose(y_baseline, y_relaxed))
        self.assertTrue(torch.allclose(y_baseline, y_strict, atol=0.0, rtol=0.0))

    def test_training_time_schedule_anneals_to_zero(self):
        config = {
            "enabled": True,
            "max_scale": 0.4,
            "schedule": "linear",
            "start_epoch": 1,
            "end_epoch": 3,
        }
        progresses = [constraint_relaxation_progress_for_epoch(epoch, 5, config) for epoch in range(5)]
        scales = [constraint_relaxation_scale_for_progress(progress, config) for progress in progresses]

        self.assertEqual(progresses, [0.0, 0.0, 0.5, 1.0, 1.0])
        self.assertAlmostEqual(scales[0], 0.4)
        self.assertAlmostEqual(scales[1], 0.4)
        self.assertAlmostEqual(scales[2], 0.2)
        self.assertAlmostEqual(scales[3], 0.0)
        self.assertAlmostEqual(scales[4], 0.0)

    def test_transformer_can_disable_relaxation_for_final_eval(self):
        model = PlatonicTransformer(
            input_dim=3,
            input_dim_vec=0,
            hidden_dim=16,
            output_dim=2,
            output_dim_vec=0,
            nhead=4,
            num_layers=2,
            solid_name="cyclic_4",
            spatial_dim=2,
            dense_mode=True,
            dropout=0.0,
            rope_sigma=None,
            ape_sigma=None,
            learned_freqs=False,
            constraint_relaxation={
                "enabled": True,
                "max_scale": 0.3,
                "schedule": "cosine",
                "apply_to": "interaction",
            },
        )

        self.assertEqual([layer.relaxation_scale for layer in model.layers], [0.3, 0.3])
        model.set_constraint_relaxation_progress(1.0)
        self.assertEqual([layer.relaxation_scale for layer in model.layers], [0.0, 0.0])
        self.assertIsNotNone(model.layers[0].relaxation_interaction)
        self.assertIsNone(model.layers[0].relaxation_linear1)


if __name__ == "__main__":
    unittest.main()
