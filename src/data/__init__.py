"""Data generation, preprocessing, and loading."""

# Keep this module lightweight — do not import torch/datamodule here.
# PyBaMM (CasADi) can crash on Windows when torch is initialized first.

__all__: list[str] = []
