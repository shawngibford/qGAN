"""Classical 1D-CNN critic with configurable dropout (v1.1 Phase 7)."""
from __future__ import annotations
import torch
import torch.nn as nn


class Critic(nn.Module):
    """1D-CNN critic with adaptive pooling to handle variable window length.

    Signature preserves the notebook's qGAN.define_critic_model contract.
    Filled in by plan 08-03.
    """

    def __init__(self, window_length: int = 10, dropout_rate: float = 0.2) -> None:
        super().__init__()
        raise NotImplementedError("Filled in by plan 08-03")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar critic score per sample in the batch. Filled in by plan 08-03."""
        raise NotImplementedError("Filled in by plan 08-03")
