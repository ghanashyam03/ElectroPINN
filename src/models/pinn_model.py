"""Physics-guided model — identical transformer backbone; physics applied in training loss."""

from __future__ import annotations

from models.battery_transformer import BatteryStateTransformer

PINNModel = BatteryStateTransformer
