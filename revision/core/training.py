"""WGAN-GP training loop with spectral-loss hook and multi-seed support."""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional
import torch


def compute_gradient_penalty(
    critic: torch.nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Two-sided gradient penalty (||grad||_2 - 1)^2. Filled in by plan 08-04."""
    raise NotImplementedError("Filled in by plan 08-04")


def train_wgan_gp(
    generator: torch.nn.Module,
    critic: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    num_epochs: int = 2000,
    n_critic: int = 9,
    lambda_gp: float = 2.16,
    lr_critic: float = 1.8046e-05,
    lr_generator: float = 6.9173e-05,
    seed: int = 42,
    spectral_loss_weight: float = 0.0,
    eval_every: int = 10,
    early_stopper: Optional[Any] = None,
    callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, list]:
    """WGAN-GP training loop — notebook-parity implementation.

    Returns dict of per-epoch metrics:
    {critic_loss_avg, generator_loss_avg, emd_avg, acf_avg, vol_avg,
     lev_avg, kurt_avg}.

    HPO-tuned defaults come from v1.1 Phase 4 and MUST match notebook behavior.
    ``spectral_loss_weight > 0`` activates v1.1 Phase 6 PSD penalty.
    ``callback(epoch, metrics_dict)`` is invoked on eval epochs — used by
    Phase 13 introspection.

    Filled in by plan 08-04.
    """
    raise NotImplementedError("Filled in by plan 08-04")
