from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def scalar_embedding(value: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=value.device, dtype=torch.float32)
        / max(half, 1)
    )
    phase = value.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


def fixed_2d_embedding(grid_size: int, dim: int) -> torch.Tensor:
    if dim % 4:
        raise ValueError("hidden size must be divisible by four")
    axis = torch.arange(grid_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    quarter = dim // 4
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(quarter, dtype=torch.float32) / max(quarter, 1)
    )
    x_phase = xx.reshape(-1, 1) * frequencies.reshape(1, -1)
    y_phase = yy.reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat((x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()), dim=1)


def modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * (1.0 + scale[:, None, :]) + shift[:, None, :]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.mlp(scalar_embedding(timestep, self.frequency_size))


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden size must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, hidden = value.shape
        qkv = self.qkv(value).reshape(batch, tokens, 3, self.heads, self.head_dim)
        query, key, content = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(query, key, content)
        attended = attended.transpose(1, 2).reshape(batch, tokens, hidden)
        return self.projection(attended)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention = SelfAttention(hidden_size, heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(condition).chunk(6, dim=-1)
        )
        attended = self.attention(modulate(self.norm1(state), shift_attn, scale_attn))
        state = state + gate_attn[:, None, :] * attended
        transformed = self.mlp(modulate(self.norm2(state), shift_mlp, scale_mlp))
        return state + gate_mlp[:, None, :] * transformed


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, output_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * output_channels)

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(state), shift, scale))


class FixedLoopDiT(nn.Module):
    """Two unique DiT-L blocks composed and repeated for a fixed number of loops."""

    def __init__(
        self,
        *,
        latent_size: int = 32,
        patch_size: int = 2,
        channels: int = 4,
        classes: int = 1000,
        hidden_size: int = 1024,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        unique_blocks: int = 2,
        loops: int = 12,
        activation_checkpoint_every: int = 0,
    ) -> None:
        super().__init__()
        if latent_size % patch_size:
            raise ValueError("latent size must be divisible by patch size")
        if unique_blocks < 1 or loops < 1:
            raise ValueError("unique_blocks and loops must be positive")
        if activation_checkpoint_every < 0:
            raise ValueError("activation_checkpoint_every must be non-negative")
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.channels = channels
        self.hidden_size = hidden_size
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.unique_blocks = unique_blocks
        self.loops = loops
        self.activation_checkpoint_every = activation_checkpoint_every
        self.grid_size = latent_size // patch_size

        self.patch_embed = nn.Conv2d(channels, hidden_size, patch_size, stride=patch_size)
        self.register_buffer(
            "position_embedding",
            fixed_2d_embedding(self.grid_size, hidden_size).unsqueeze(0),
            persistent=True,
        )
        self.time_embed = TimestepEmbedder(hidden_size)
        self.class_embed = nn.Embedding(classes + 1, hidden_size)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, heads, mlp_ratio) for _ in range(unique_blocks)]
        )
        self.final = FinalLayer(hidden_size, patch_size, channels)
        self.initialize_weights()

    @property
    def null_class(self) -> int:
        return self.class_embed.num_embeddings - 1

    @property
    def effective_depth(self) -> int:
        return self.unique_blocks * self.loops

    def initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        nn.init.xavier_uniform_(self.patch_embed.weight.reshape(self.patch_embed.weight.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final.modulation[-1].weight)
        nn.init.zeros_(self.final.modulation[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = patches.shape
        if tokens != self.grid_size * self.grid_size:
            raise ValueError("unexpected token count")
        patch = self.patch_size
        patches = patches.reshape(
            batch, self.grid_size, self.grid_size, patch, patch, self.channels
        )
        return torch.einsum("nhwpqc->nchpwq", patches).reshape(
            batch, self.channels, self.latent_size, self.latent_size
        )

    def forward(self, noisy: torch.Tensor, timestep: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if noisy.shape[1:] != (self.channels, self.latent_size, self.latent_size):
            raise ValueError(f"unexpected latent shape: {tuple(noisy.shape)}")
        state = self.patch_embed(noisy).flatten(2).transpose(1, 2)
        state = state + self.position_embedding.to(device=state.device, dtype=state.dtype)
        condition = self.time_embed(1000.0 * timestep) + self.class_embed(labels)
        application_index = 0
        for _ in range(self.loops):
            for block in self.blocks:
                application_index += 1
                should_checkpoint = (
                    self.training
                    and torch.is_grad_enabled()
                    and self.activation_checkpoint_every > 0
                    and application_index % self.activation_checkpoint_every == 0
                )
                if should_checkpoint:
                    state = checkpoint(block, state, condition, use_reentrant=False)
                else:
                    state = block(state, condition)
        return self.unpatchify(self.final(state, condition))

    def metadata(self) -> dict[str, int | float | str]:
        return {
            "architecture": (
                f"fixed_loop_dit_{self.unique_blocks}n{self.loops}l_d{self.hidden_size}"
            ),
            "latent_size": self.latent_size,
            "patch_size": self.patch_size,
            "channels": self.channels,
            "hidden_size": self.hidden_size,
            "heads": self.heads,
            "mlp_ratio": self.mlp_ratio,
            "unique_blocks": self.unique_blocks,
            "loops": self.loops,
            "effective_depth": self.effective_depth,
            "activation_checkpoint_every": self.activation_checkpoint_every,
            "stored_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "intermediate_supervision": "none",
            "loop_embedding": "none",
        }


__all__ = ["FixedLoopDiT"]
