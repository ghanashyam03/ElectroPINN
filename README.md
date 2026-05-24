# Battery PINN — Physics-Guided Temporal Transformer for Li-Ion State Estimation

Research codebase for **robust lithium-ion battery state estimation under unseen dynamic current profiles**, using PyBaMM-generated data and a compact **causal temporal transformer** with optional **physics-guided losses**.

**Project identity (unchanged):** estimate **SoC**, **terminal voltage**, and **cell temperature** from current and measured signals, and demonstrate that physics-guided training generalizes better than supervised-only training on unseen drive cycles.

## Scientific framing

| Approach | Architecture | Supervision | Physics |
|----------|--------------|-------------|---------|
| Transformer baseline | Shared temporal transformer | SoC + V + T | None |
| Physics-guided model | **Identical** transformer | SoC + V + T | Coulomb, differential, monotonicity, voltage/thermal consistency |

The only intentional difference between baseline and physics-guided paths is the **loss function**.

**Generalization protocol**

- Train profiles: `constant`, `pulse`
- Test profiles: `random`, `wltp`
- **ID_RMSE:** validation on train-profile distribution
- **OOD_RMSE:** test on unseen profiles
- **Generalization score:** `OOD_RMSE / ID_RMSE` (lower is better; near 1.0 is ideal)

## Architecture (compact temporal transformer)

1. Input projection — `[current, voltage, temperature, elapsed_time]`
2. Relative positional encoding
3. Causal PreNorm transformer encoder (4 layers, 4 heads, 128 dim default)
4. Shared latent sequence
5. Multi-head readout — SoC, voltage, temperature

No giant models, no HuggingFace stacks, no graph networks. Designed for **≤4 GB VRAM**.

## Installation

```bash
cd battery-pinn
uv sync --all-extras
```

**Detailed step-by-step instructions:** see [RUNBOOK.md](RUNBOOK.md).

## Workflow

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make generate-data` | PyBaMM SPM simulations → Parquet + preprocess |
| `make train-baseline` | Supervised transformer (no physics losses) |
| `make train-pinn` | Physics-guided transformer |
| `make evaluate` | Metrics, ID/OOD, uncertainty, OOD detection, plots |
| `make ablation` | Automated physics-term ablation study |
| `make benchmark` | Latency and memory report |
| `make test` | Pytest suite |

Windows (no Make):

```powershell
uv run python -m data.generate_dataset
uv run python -m training.train_baseline
uv run python -m training.train_pinn
uv run python -m evaluation.evaluate
uv run python -m evaluation.ablation
uv run pytest tests/ -m "not slow"
```

Fast smoke: `$env:BATTERY_PINN_FAST_TEST="1"` before the commands above.

## Evaluation outputs

**Preserved plots:** `predicted_vs_actual_soc`, `voltage_trajectory`, `training_curve`, `physics_loss_curve`, `error_distribution`, `unseen_profile_generalization`

**Added plots:** `uncertainty_calibration`, `uncertainty_distribution`, `ood_detection_histogram`, `attention_visualization`, `ablation_results`

## Configuration

Hydra configs under `configs/` — model (`transformer`, `physics`, `ablation`), train, simulation.

## Reproducibility

Fixed seeds, deterministic cuDNN when CUDA is available, simulation validation rejects corrupted trajectories (NaNs, duplicate timestamps, out-of-bound SoC/voltage).

## License

MIT — portfolio and research extension.
