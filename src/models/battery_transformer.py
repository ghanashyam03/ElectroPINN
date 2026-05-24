"""Shared temporal transformer for battery state estimation."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.temporal_transformer import TemporalTransformerEncoder


class BatteryStateTransformer(nn.Module):
    """
    Physics-guided temporal transformer.

    Inputs: [current, voltage, temperature, elapsed_time] per timestep.
    Outputs: [soc, voltage, temperature] per timestep.
    """

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        voltage_min: float = 2.5,
        voltage_max: float = 4.2,
        temp_min: float = 0.0,
        temp_max: float = 80.0,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        self.temp_min = temp_min
        self.temp_max = temp_max

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder = TemporalTransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_len=max_seq_len,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Return latent sequence (B, T, H) before readout."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.input_proj(x)
        latent, aux = self.encoder(h, return_attention=return_attention)
        if return_attention:
            return latent, aux  # type: ignore[return-value]
        return latent

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        raw = self.head(latent)
        out = self._constrain_outputs(raw)
        if return_latent:
            return out, latent
        return out

    def _constrain_outputs(self, raw: torch.Tensor) -> torch.Tensor:
        soc = torch.sigmoid(raw[..., 0:1])
        voltage = self.voltage_min + torch.sigmoid(raw[..., 1:2]) * (
            self.voltage_max - self.voltage_min
        )
        temperature = self.temp_min + torch.sigmoid(raw[..., 2:3]) * (
            self.temp_max - self.temp_min
        )
        return torch.cat([soc, voltage, temperature], dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
