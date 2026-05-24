# ElectroPINN — Physics-Guided Temporal Transformer for Li-Ion State Estimation

Research-focused implementation of a compact **physics-guided temporal transformer** for lithium-ion battery state estimation under unseen dynamic current profiles using **PyBaMM-generated electrochemical trajectories**.

The project focuses on one problem only:

> Robust estimation of **State of Charge (SoC)**, terminal voltage, and cell temperature from sequential battery measurements under dynamic operating conditions.

The repository intentionally stays narrow in scope and avoids unnecessary expansion into unrelated battery domains.

---

# Core Idea

Two models are compared under the exact same setup:

| Model | Architecture | Supervision | Physics Constraints |
|---|---|---|---|
| Baseline | Temporal Transformer | SoC + Voltage + Temperature | None |
| Physics-Guided | Identical Transformer | SoC + Voltage + Temperature | Enabled |

The only intentional difference between the two paths is the addition of physics-guided loss terms.

This allows a scientifically fair comparison between:
- purely supervised temporal learning,
- and physics-guided temporal learning.

---

# Scientific Objective

The project evaluates whether physics-guided training improves:
- robustness,
- temporal consistency,
- and generalization

under unseen current profiles.

## Train Profiles
- `constant`
- `pulse`

## Unseen Test Profiles
- `random`
- `wltp`

## Metrics

- ID_RMSE → validation on train-profile distribution
- OOD_RMSE → evaluation on unseen profiles
- Generalization Score → `OOD_RMSE / ID_RMSE`

Lower generalization score is better.

---

# Architecture

Compact causal temporal transformer:

1. Input projection
   - `[current, voltage, temperature, elapsed_time]`

2. Relative positional encoding

3. Causal transformer encoder
   - PreNorm
   - Multi-head self-attention
   - GELU
   - LayerNorm
   - Residual connections

4. Shared latent temporal representation

5. Multi-head outputs
   - SoC
   - Voltage
   - Temperature

The architecture intentionally avoids:
- giant foundation models,
- HuggingFace dependencies,
- graph neural networks,
- overly complex battery PDE systems.

The focus is precision and scientific clarity, not scale.

---

# Physics-Guided Constraints

The physics-guided model introduces differentiable constraints including:

- Coulomb consistency
- Differential SoC consistency
- Monotonic discharge behavior
- Voltage-current consistency
- Voltage smoothness
- Thermal smoothness

These constraints improve physical plausibility and temporal stability under unseen dynamics.

---

# Dataset Pipeline

Battery trajectories are generated using:
- PyBaMM
- Single Particle Model (SPM)

The pipeline:
1. Generates electrochemical simulations
2. Validates trajectories
3. Exports Parquet datasets
4. Preprocesses train/validation/test splits
5. Trains transformer models
6. Evaluates robustness and uncertainty

---

# Installation

```bash
uv sync --all-extras
```

---

# Workflow

| Command | Description |
|---|---|
| `make generate-data` | Generate PyBaMM simulations |
| `make train-baseline` | Train supervised transformer |
| `make train-pinn` | Train physics-guided transformer |
| `make evaluate` | Generate metrics and plots |
| `make benchmark` | Benchmark latency and memory |
| `make ablation` | Run physics-loss ablation study |
| `make test` | Run test suite |

Windows equivalent:

```powershell
uv run python -m data.generate_dataset
uv run python -m training.train_baseline
uv run python -m training.train_pinn
uv run python -m evaluation.evaluate
```

---

# Evaluation Outputs

Generated outputs include:

- Predicted vs actual SoC
- Voltage trajectories
- Physics loss curves
- Error distributions
- Unseen-profile generalization plots
- Attention visualizations
- OOD detection histograms
- MC-dropout uncertainty plots

---

# Engineering Features

- Deterministic training
- Physics-guided losses
- Compact transformer implementation
- Sequence-aware evaluation
- MC-dropout uncertainty estimation
- Mahalanobis OOD detection
- Hydra configuration system
- Automated ablation pipeline
- GPU/CPU compatible execution

---

# Repository Structure

```text
src/
├── data/
├── simulation/
├── models/
├── training/
├── evaluation/
└── utils/

configs/
tests/
outputs/
```

---

# Notes

This repository is designed as:
- a focused research project,
- a scientifically grounded ML system,
- and a robust engineering implementation of physics-guided temporal modeling for batteries.

It intentionally prioritizes:
- correctness,
- clarity,
- and disciplined scope

over unnecessary feature expansion.
