"""Supervised baseline — shared temporal transformer backbone."""

from __future__ import annotations

from models.battery_transformer import BatteryStateTransformer

# Backward-compatible alias: baseline uses the same architecture as the physics-guided model.
BaselineMLP = BatteryStateTransformer
