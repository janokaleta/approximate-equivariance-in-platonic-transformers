"""Smoke tests for relaxed group convolution.

Run:
    uv run python tests/test_relaxed_group_convolution.py
"""

import os
import sys
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from platonic_transformers.models.platoformer.linear import (
    PlatonicLinear,
    relaxed_group_convolution_regularization,
)
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer


class RelaxedGroupConvolutionTest(unittest.TestCase):
    def test_disabled_has_baseline_state_dict(self):
        layer = PlatonicLinear(6, 4, "cyclic_2")

        self.assertNotIn("relaxed_kernel", layer.state_dict())
        self.assertNotIn("relaxed_mixing", layer.state_dict())
        self.assertFalse(layer.relaxed_group_convolution)

    def test_zero_initialized_mixing_preserves_baseline_and_gets_gradients(self):
        torch.manual_seed(0)
        baseline = PlatonicLinear(6, 4, "cyclic_2")
        relaxed = PlatonicLinear(
            6,
            4,
            "cyclic_2",
            relaxed_group_convolution={
                "enabled": True,
                "num_extra_kernels": 2,
                "mixing_init": "zeros",
                "kernel_l2": 0.1,
            },
        )
        relaxed.load_state_dict(baseline.state_dict(), strict=False)

        x = torch.randn(5, 6)
        torch.testing.assert_close(relaxed(x), baseline(x), atol=0.0, rtol=0.0)

        loss = relaxed(x).square().sum()
        loss.backward()

        self.assertIsNotNone(relaxed.relaxed_mixing.grad)
        self.assertGreater(relaxed.relaxed_mixing.grad.abs().sum(), 0)
        self.assertGreater(relaxed_group_convolution_regularization(relaxed), 0)

    def test_config_reaches_nested_transformer_linears(self):
        model = PlatonicTransformer(
            input_dim=3,
            input_dim_vec=0,
            hidden_dim=4,
            output_dim=1,
            output_dim_vec=1,
            nhead=2,
            num_layers=1,
            solid_name="cyclic_2",
            spatial_dim=2,
            dense_mode=False,
            ffn_readout=False,
            attention=False,
            rope_sigma=None,
            ape_sigma=None,
            relaxed_group_convolution={
                "enabled": True,
                "num_extra_kernels": 1,
                "mixing_init": "zeros",
            },
        )

        relaxed_linears = [
            module for module in model.modules()
            if isinstance(module, PlatonicLinear) and module.relaxed_group_convolution
        ]
        parameter_names = {name for name, _ in model.named_parameters()}

        self.assertTrue(relaxed_linears)
        self.assertIn("layers.0.linear1.relaxed_mixing", parameter_names)
        self.assertIn("layers.0.interaction.q_proj.relaxed_kernel", parameter_names)

    def test_regularization_returns_zero_when_disabled(self):
        layer = PlatonicLinear(6, 4, "cyclic_2")
        reg = relaxed_group_convolution_regularization(layer)
        self.assertEqual(reg.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
