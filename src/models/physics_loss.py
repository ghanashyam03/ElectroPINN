"""Physics-guided loss terms for temporal battery state estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class PhysicsLossConfig:
    """Weights and toggles for physics-guided terms."""

    lambda_data: float = 1.0
    lambda_coulomb: float = 0.5
    lambda_differential: float = 0.3
    lambda_monotonicity: float = 0.2
    lambda_voltage_smooth: float = 0.1
    lambda_voltage_current: float = 0.1
    lambda_thermal: float = 0.05
    nominal_capacity_ah: float = 5.0
    use_coulomb: bool = True
    use_differential: bool = True
    use_monotonicity: bool = True
    use_voltage_smooth: bool = True
    use_voltage_current: bool = True
    use_thermal: bool = True


@dataclass
class PhysicsLossToggles:
    """Ablation-friendly subset of physics terms."""

    coulomb: bool = True
    differential: bool = True
    monotonicity: bool = True
    voltage_smooth: bool = True
    voltage_current: bool = True
    thermal: bool = True


class PhysicsLossModule(nn.Module):
    """Composite physics-guided loss (PINN path only)."""

    def __init__(
        self,
        config: PhysicsLossConfig,
        toggles: PhysicsLossToggles | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.toggles = toggles or PhysicsLossToggles()
        self.mse = nn.MSELoss()
        self.huber_soc = nn.HuberLoss(delta=0.1, reduction="none")

    def data_loss(
        self,
        predictions: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Supervised combined loss on SoC, voltage, and temperature."""
        pred_soc = predictions[..., 0:1]
        pred_v = predictions[..., 1:2]
        pred_t = predictions[..., 2:3]
        target_soc = batch["soc_physical"]

        # Component 1 & 2: Huber loss and boundary emphasis weighting
        raw_loss = self.huber_soc(pred_soc, target_soc)
        weights = 1.0 + 0.5 * torch.abs(target_soc - 0.5)
        weighted_loss = (raw_loss * weights).mean()

        # Component 3: Range consistency penalty
        pred_range = pred_soc.max() - pred_soc.min()
        target_range = target_soc.max() - target_soc.min()
        range_loss = torch.relu(target_range - pred_range)
        total_soc_loss = weighted_loss + 0.2 * range_loss

        loss = total_soc_loss + self.mse(pred_v, batch["voltage"].unsqueeze(-1))
        loss = loss + self.mse(pred_t, batch["temperature"].unsqueeze(-1))
        return loss

    def coulomb_loss(
        self,
        soc: torch.Tensor,
        current: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Discrete Coulomb counting: SoC(t+dt) ≈ SoC(t) - I*dt/Q."""
        q_as = self.config.nominal_capacity_ah * 3600.0
        soc_t = soc[:, :-1, :]
        soc_next = soc[:, 1:, :]
        i = current[:, :-1].unsqueeze(-1)
        delta = dt[:, 1:].unsqueeze(-1).clamp(min=1e-6)
        expected = soc_t - (i * delta) / q_as
        return self.mse(soc_next, expected)

    def differential_coulomb_loss(
        self,
        soc: torch.Tensor,
        elapsed_time: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        """dSOC/dt = -I/Q via autograd w.r.t. elapsed time."""
        q_as = self.config.nominal_capacity_ah * 3600.0
        t = elapsed_time.detach().clone().requires_grad_(True)
        soc_sum = soc.sum()
        if not soc.requires_grad:
            # Finite-difference fallback when graph is detached
            dsoc = (soc[:, 1:, :] - soc[:, :-1, :]) / (
                elapsed_time[:, 1:].unsqueeze(-1) - elapsed_time[:, :-1].unsqueeze(-1) + 1e-6
            )
            i_mid = current[:, 1:].unsqueeze(-1)
            target = -i_mid / q_as
            return self.mse(dsoc, target)

        grads = torch.autograd.grad(
            soc_sum,
            t,
            grad_outputs=torch.ones_like(soc_sum),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grads is None:
            dsoc = (soc[:, 1:, :] - soc[:, :-1, :]) / (
                elapsed_time[:, 1:].unsqueeze(-1) - elapsed_time[:, :-1].unsqueeze(-1) + 1e-6
            )
            return self.mse(dsoc, -current[:, 1:].unsqueeze(-1) / q_as)

        dsoc_dt = grads.unsqueeze(-1)
        target = -current.unsqueeze(-1) / q_as
        return self.mse(dsoc_dt, target)

    def monotonicity_loss(self, soc: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """Penalize SoC increase during discharge."""
        dsoc = soc[:, 1:, :] - soc[:, :-1, :]
        discharging = current[:, :-1].unsqueeze(-1) > 0
        return (torch.relu(dsoc) * discharging.float()).mean()

    def voltage_smoothness_loss(self, voltage: torch.Tensor) -> torch.Tensor:
        """Penalize sharp voltage transients (second difference)."""
        d1 = voltage[:, 1:, :] - voltage[:, :-1, :]
        d2 = d1[:, 1:, :] - d1[:, :-1, :]
        return (d2**2).mean()

    def voltage_current_consistency(self, voltage: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """
        Soft consistency: voltage drop should correlate with discharge current magnitude.
        Penalize anti-correlation between |I| and -dV/dt during discharge.
        """
        dv = voltage[:, 1:, :] - voltage[:, :-1, :]
        i = current[:, 1:].unsqueeze(-1)
        # During discharge (I>0), expect voltage non-increasing (dv <= 0)
        discharge = i > 0
        violation = torch.relu(dv) * discharge.float()
        # Also penalize weak correlation proxy: |I| vs -dv
        corr_penalty = torch.relu(-dv * i)
        return violation.mean() + 0.1 * corr_penalty.mean()

    def thermal_smoothness_loss(self, temperature: torch.Tensor) -> torch.Tensor:
        """Penalize unrealistic temperature spikes."""
        d1 = temperature[:, 1:, :] - temperature[:, :-1, :]
        d2 = d1[:, 1:, :] - d1[:, :-1, :]
        return (d2**2).mean()

    def forward(
        self,
        predictions: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        toggles = self.toggles
        pred_soc = predictions[..., 0:1]
        pred_v = predictions[..., 1:2]
        pred_t = predictions[..., 2:3]

        l_data = self.data_loss(predictions, batch)
        zero = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)

        l_coulomb = (
            self.coulomb_loss(pred_soc, batch["current"], batch["dt"])
            if toggles.coulomb and cfg.use_coulomb
            else zero
        )
        l_diff = (
            self.differential_coulomb_loss(pred_soc, batch["elapsed_time"], batch["current"])
            if toggles.differential and cfg.use_differential
            else zero
        )
        l_mono = (
            self.monotonicity_loss(pred_soc, batch["current"])
            if toggles.monotonicity and cfg.use_monotonicity
            else zero
        )
        l_v_smooth = (
            self.voltage_smoothness_loss(pred_v)
            if toggles.voltage_smooth and cfg.use_voltage_smooth
            else zero
        )
        l_vi = (
            self.voltage_current_consistency(pred_v, batch["current"])
            if toggles.voltage_current and cfg.use_voltage_current
            else zero
        )
        l_thermal = (
            self.thermal_smoothness_loss(pred_t)
            if toggles.thermal and cfg.use_thermal
            else zero
        )

        total = (
            cfg.lambda_data * l_data
            + cfg.lambda_coulomb * l_coulomb
            + cfg.lambda_differential * l_diff
            + cfg.lambda_monotonicity * l_mono
            + cfg.lambda_voltage_smooth * l_v_smooth
            + cfg.lambda_voltage_current * l_vi
            + cfg.lambda_thermal * l_thermal
        )

        return {
            "loss": total,
            "data": l_data,
            "coulomb": l_coulomb,
            "differential": l_diff,
            "monotonicity": l_mono,
            "voltage_smooth": l_v_smooth,
            "voltage_current": l_vi,
            "thermal": l_thermal,
        }
