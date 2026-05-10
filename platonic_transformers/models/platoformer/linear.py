import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Mapping

# Import the pre-computed group data
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS


@dataclass(frozen=True)
class RelaxedGroupConvolutionConfig:
    """
    Optional additive relaxation for PlatonicLinear.

    Parameterization when enabled:
      W[h, g] = K[g^-1 h] + scale * sum_r A[h, g, r] * R[r, g^-1 h]

    K is the existing equivariant kernel, R are extra equivariant kernels, and
    A are untied group-pair mixing coefficients that provide the
    symmetry-breaking capacity. With A initialized to zero, the layer starts
    from the exact equivariant baseline while still allowing gradients into A.
    """
    enabled: bool = False
    num_extra_kernels: int = 0
    scale: float = 1.0
    kernel_init: str = "normal"
    kernel_init_std: float | None = None
    mixing_init: str = "zeros"
    mixing_init_std: float = 1e-3
    mixing_l1: float = 0.0
    mixing_l2: float = 0.0
    kernel_l2: float = 0.0


_RELAXED_GROUP_CONVOLUTION_CONFIG: ContextVar[RelaxedGroupConvolutionConfig | None] = ContextVar(
    "relaxed_group_convolution_config",
    default=None,
)


def _coerce_relaxed_group_convolution_config(config: Any = None) -> RelaxedGroupConvolutionConfig:
    if config is None:
        return RelaxedGroupConvolutionConfig()
    if isinstance(config, RelaxedGroupConvolutionConfig):
        return config
    if isinstance(config, bool):
        return RelaxedGroupConvolutionConfig(enabled=config)
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if isinstance(config, Mapping):
        valid_keys = RelaxedGroupConvolutionConfig.__dataclass_fields__.keys()
        unknown_keys = set(config.keys()) - set(valid_keys)
        if unknown_keys:
            raise ValueError(f"Unknown relaxed_group_convolution keys: {sorted(unknown_keys)}")
        return RelaxedGroupConvolutionConfig(**dict(config))
    raise TypeError(
        "relaxed_group_convolution must be None, bool, mapping, ConfigDict, "
        "or RelaxedGroupConvolutionConfig"
    )


@contextmanager
def platonic_linear_relaxed_group_convolution(config: Any = None):
    """Temporarily apply relaxed group-convolution config to nested PlatonicLinear constructors."""
    token = _RELAXED_GROUP_CONVOLUTION_CONFIG.set(_coerce_relaxed_group_convolution_config(config))
    try:
        yield
    finally:
        _RELAXED_GROUP_CONVOLUTION_CONFIG.reset(token)


def relaxed_group_convolution_regularization_enabled(config: Any = None) -> bool:
    """Return whether the config can produce a nonzero explicit relaxation penalty."""
    config = _coerce_relaxed_group_convolution_config(config)
    if not config.enabled or config.num_extra_kernels <= 0:
        return False
    return any(
        float(coefficient or 0.0) != 0.0
        for coefficient in (config.mixing_l1, config.mixing_l2, config.kernel_l2)
    )


class PlatonicLinear(nn.Module):
    """
    A Linear layer constrained to be a group convolution over a Platonic Solid group.
    This version includes a corrected initialization scheme to preserve variance.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        solid: str,
        bias: bool = True,
        relaxed_group_convolution: Any = None,
    ):
        super().__init__()
        
        if solid.lower() not in PLATONIC_GROUPS:
            raise ValueError(f"Solid '{solid}' not recognized. Available groups are: {list(PLATONIC_GROUPS.keys())}")
        
        group = PLATONIC_GROUPS[solid.lower()]
        self.G = group.G
        self.in_features = in_features
        self.out_features = out_features

        if in_features % self.G != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by the group order {self.G} for solid '{solid}'")
        if out_features % self.G != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by the group order {self.G} for solid '{solid}'")
            
        self.in_channels = in_features // self.G
        self.out_channels = out_features // self.G

        if relaxed_group_convolution is None:
            relaxed_group_convolution = _RELAXED_GROUP_CONVOLUTION_CONFIG.get()
        self.relaxed_group_convolution_config = _coerce_relaxed_group_convolution_config(relaxed_group_convolution)
        self.relaxed_group_convolution = (
            self.relaxed_group_convolution_config.enabled
            and self.relaxed_group_convolution_config.num_extra_kernels > 0
        )
        if self.relaxed_group_convolution_config.enabled and self.relaxed_group_convolution_config.num_extra_kernels <= 0:
            raise ValueError("relaxed_group_convolution.enabled=True requires num_extra_kernels > 0")

        self.kernel = nn.Parameter(torch.empty(self.G, self.out_channels, self.in_channels))
        if self.relaxed_group_convolution:
            self.relaxed_kernel = nn.Parameter(torch.empty(
                self.relaxed_group_convolution_config.num_extra_kernels,
                self.G,
                self.out_channels,
                self.in_channels,
            ))
            self.relaxed_mixing = nn.Parameter(torch.empty(
                self.G,
                self.G,
                self.relaxed_group_convolution_config.num_extra_kernels,
            ))
        else:
            self.register_parameter('relaxed_kernel', None)
            self.register_parameter('relaxed_mixing', None)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('cayley_table', group.cayley_table)
        self.register_buffer('inverse_indices', group.inverse_indices)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the kernel and bias with variance-preserving scaling.
        
        Standard initializers (like Kaiming) fail to correctly infer the
        effective fan-in of the full weight matrix. We must calculate it
        manually as (group_size * in_channels_per_group).
        """
        # Calculate the effective fan-in for the full weight matrix.
        fan_in = self.G * self.in_channels
        
        # Initialize the kernel from a normal distribution. The std is calculated
        # to ensure the output variance is approximately equal to the input variance.
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(self.kernel, mean=0.0, std=std)
        self._reset_relaxed_parameters(std)

        if self.bias is not None:
            # Initialize bias using the same correct fan-in.
            if fan_in > 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def _reset_relaxed_parameters(self, default_std: float) -> None:
        if not self.relaxed_group_convolution:
            return

        config = self.relaxed_group_convolution_config
        kernel_std = default_std if config.kernel_init_std is None else config.kernel_init_std
        if config.kernel_init == "normal":
            nn.init.normal_(self.relaxed_kernel, mean=0.0, std=kernel_std)
        elif config.kernel_init == "zeros":
            nn.init.zeros_(self.relaxed_kernel)
        else:
            raise ValueError("relaxed_group_convolution.kernel_init must be 'normal' or 'zeros'")

        if config.mixing_init == "zeros":
            nn.init.zeros_(self.relaxed_mixing)
        elif config.mixing_init == "normal":
            nn.init.normal_(self.relaxed_mixing, mean=0.0, std=config.mixing_init_std)
        else:
            raise ValueError("relaxed_group_convolution.mixing_init must be 'zeros' or 'normal'")

    def get_weight(self) -> Tensor:
        """
        Constructs the full [G*O, G*I] weight matrix from the fundamental kernel.
        """
        device = self.kernel.device
        h_indices = torch.arange(self.G, device=device).view(self.G, 1)
        g_indices = torch.arange(self.G, device=device).view(1, self.G)

        inv_g_indices = self.inverse_indices[g_indices]
        kernel_group_idx = self.cayley_table[inv_g_indices, h_indices]
        
        expanded_kernel = self.kernel[kernel_group_idx]
        if self.relaxed_group_convolution:
            expanded_relaxed_kernel = self.relaxed_kernel[:, kernel_group_idx]
            relaxed_kernel = torch.einsum(
                'hgr,rhgoi->hgoi',
                self.relaxed_mixing,
                expanded_relaxed_kernel,
            )
            expanded_kernel = expanded_kernel + self.relaxed_group_convolution_config.scale * relaxed_kernel

        weight = expanded_kernel.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return weight

    def relaxed_regularization_loss(self) -> Tensor:
        """
        Return the configured relaxation penalty for external training loops.

        The aggregate helper below lets training modules add this term to their
        task loss only when explicit regularization coefficients are configured.
        """
        loss = self.kernel.new_zeros(())
        if not self.relaxed_group_convolution:
            return loss

        config = self.relaxed_group_convolution_config
        if config.mixing_l1:
            loss = loss + config.mixing_l1 * self.relaxed_mixing.abs().sum()
        if config.mixing_l2:
            loss = loss + config.mixing_l2 * self.relaxed_mixing.square().sum()
        if config.kernel_l2:
            loss = loss + config.kernel_l2 * self.relaxed_kernel.square().sum()
        return loss

    def forward(self, x: Tensor) -> Tensor:
        """Applies the group-equivariant linear transformation."""
        weight = self.get_weight()
        output = F.linear(x, weight, None)
        
        if self.bias is not None:
            output_shape = output.shape
            output = output.view(*output_shape[:-1], self.G, self.out_channels)
            output = output + self.bias
            output = output.view(output_shape)
            
        return output
    
    def __repr__(self) -> str:
        relaxed = ""
        if self.relaxed_group_convolution:
            relaxed = f", relaxed_extra_kernels={self.relaxed_group_convolution_config.num_extra_kernels}"
        return f"{self.__class__.__name__}(G={self.G}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}{relaxed})"


def relaxed_group_convolution_regularization(module: nn.Module) -> Tensor:
    """Aggregate configured relaxed group-convolution penalties across a module tree."""
    loss = None
    for child in module.modules():
        if isinstance(child, PlatonicLinear):
            term = child.relaxed_regularization_loss()
            loss = term if loss is None else loss + term
    if loss is not None:
        return loss

    try:
        param = next(module.parameters())
        return param.new_zeros(())
    except StopIteration:
        return torch.zeros(())
    




# Equivariance test

def run_equivariance_test(solid_name: str):
    group = PLATONIC_GROUPS[solid_name]
    G = group.G
    
    I, O, B = 4, 8, 2
    in_feats = G * I
    out_feats = G * O

    print(f"Initializing PlatonicLinear(solid='{solid_name}')")
    print(f"  (G={G}, I={I}, O={O}) -> in_features={in_feats}, out_features={out_feats}")
    layer = PlatonicLinear(in_features=in_feats, out_features=out_feats, solid=solid_name)
    
    torch.manual_seed(42)
    input_signal = torch.randn(B, in_feats)
    
    print("--- Testing Right-Equivariance Property ---")
    
    all_tests_passed = True
    original_output = layer(input_signal)

    for h in range(G):
        # transform_indices = group.cayley_table[:, h]
        transform_indices = group.cayley_table[h, :]
        
        if len(torch.unique(transform_indices)) != G:
            print(f"[!] Cayley table error at h={h}. Column is not a permutation!")
            all_tests_passed = False
            break

        input_unflattened = input_signal.view(B, G, I)
        transformed_unflattened = input_unflattened[:, transform_indices]
        transformed_input = transformed_unflattened.reshape(B, in_feats)
        output_lhs = layer(transformed_input)

        original_output_unflattened = original_output.view(B, G, O)
        transformed_output_unflattened = original_output_unflattened[:, transform_indices]
        output_rhs = transformed_output_unflattened.reshape(B, out_feats)

        if not torch.allclose(output_lhs, output_rhs, atol=1e-5):
            print(f"  [!] Test FAILED for solid '{solid_name}', group element h = {h}")
            print(f"      Max difference: {torch.max(torch.abs(output_lhs - output_rhs))}")
            all_tests_passed = False
            break
            
    if all_tests_passed:
        print(f"  [✓] All equivariance tests passed successfully for '{solid_name}'!")

if __name__ == '__main__':
    for solid_name in PLATONIC_GROUPS:
        print(f"\n{'='*25} TESTING: {solid_name.upper()} {'='*25}")
        run_equivariance_test(solid_name)
