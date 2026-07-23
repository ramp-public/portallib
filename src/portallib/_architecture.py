"""Internal canonical task-latent to LoRA architecture."""

from __future__ import annotations

import torch
from torch import nn

from .config import PortalConfig

GeneratedLora = dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]


class PortalCore(nn.Module):
    """Base-agnostic canonical LoRA core shared during source training and refitting."""

    def __init__(self, config: PortalConfig):
        super().__init__()
        self.config = config
        self.l1 = nn.Linear(config.d_z + config.d_layer, config.hidden)
        self.l2 = nn.Linear(config.hidden, config.hidden)
        self.film = nn.Linear(config.d_z, 2 * config.hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.A = nn.ModuleDict({name: nn.Linear(config.hidden, config.rank * config.d_core) for name in config.modules})
        self.B = nn.ModuleDict({name: nn.Linear(config.hidden, config.d_core * config.rank) for name in config.modules})
        for head in self.B.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def hidden(self, z: torch.Tensor, layer_embeddings: torch.Tensor) -> torch.Tensor:
        expanded_z = z.unsqueeze(0).expand(layer_embeddings.shape[0], -1)
        hidden = torch.nn.functional.gelu(self.l1(torch.cat([expanded_z, layer_embeddings], dim=-1)))
        gamma, beta = self.film(z).chunk(2, dim=-1)
        hidden = (1.0 + gamma) * hidden + beta
        return torch.nn.functional.gelu(self.l2(hidden))


class PortalAlignment(nn.Module):
    """Thin base-specific linear alignment around the canonical LoRA factors."""

    def __init__(self, config: PortalConfig, *, zero_output: bool = False):
        super().__init__()
        self.config = config
        self.layer_embeddings = nn.Embedding(config.n_layers, config.d_layer)
        input_groups, output_groups = config.alignment_groups
        self.input = nn.ParameterDict(
            {
                group: nn.Parameter(torch.randn(config.d_core, in_features) * 0.02)
                for group, in_features in input_groups.items()
            }
        )
        self.output = nn.ParameterDict(
            {
                group: nn.Parameter(
                    torch.zeros(out_features, config.d_core)
                    if zero_output
                    else torch.randn(out_features, config.d_core) * 0.02
                )
                for group, out_features in output_groups.items()
            }
        )

    def forward(self, core: PortalCore, z: torch.Tensor) -> GeneratedLora:
        if z.shape != (self.config.d_z,):
            raise ValueError(f"expected task latent shape ({self.config.d_z},), got {tuple(z.shape)}")
        layer_ids = torch.arange(self.config.n_layers, device=z.device)
        hidden = core.hidden(z, self.layer_embeddings(layer_ids))
        generated: GeneratedLora = {}
        for target in self.config.projection_targets:
            canonical_a = core.A[target.module_name](hidden[target.layer_index]).view(
                self.config.rank, self.config.d_core
            )
            canonical_b = core.B[target.module_name](hidden[target.layer_index]).view(
                self.config.d_core, self.config.rank
            )
            generated[target.key] = (
                canonical_a @ self.input[target.input_group],
                self.output[target.output_group] @ canonical_b,
            )
        return generated
