"""WGAN-GP training loop with spectral-loss hook and multi-seed support.

Extracted from ``qgan_pennylane.ipynb`` cell 26 (``qGAN.train_qgan`` +
``qGAN._train_one_epoch`` + ``qGAN.compute_gradient_penalty``) and cell 31
(``EarlyStopping``).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Gradient penalty — mirrors qGAN.compute_gradient_penalty + the inline
# computation inside qGAN._train_one_epoch (cell 26).
# ─────────────────────────────────────────────────────────────────────────────
def compute_gradient_penalty(
    critic: torch.nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Two-sided WGAN-GP gradient penalty: ``mean((||grad||_2 - 1) ** 2)``.

    Mirrors the inline GP block of ``qGAN._train_one_epoch`` (cell 26): a single
    ``alpha`` per sample is drawn from ``U(0, 1)``, broadcast across remaining
    dims, used to interpolate between real and fake, and the gradient norm of
    the critic w.r.t. the interpolated input is penalised toward 1.

    Critic input shape contract (from 08-03 SUMMARY): ``(batch_size, 1,
    window_length)``. The function flattens per-sample gradients to ``(B, -1)``
    before taking the L2 norm so it works for both ``(B, L)`` and ``(B, 1, L)``
    critics — matching the notebook's ``gradients.norm(2, dim=[1, 2])`` for the
    3D critic-input case.
    """
    batch_size = real_samples.size(0)
    # Match cell 26: ``alpha = torch.rand(...).to(real_batch_tensor.device)`` —
    # ``device`` arg is preserved for API symmetry but the operative placement
    # is real_samples.device so the interpolation does not cross devices.
    alpha = torch.rand(
        batch_size, 1, dtype=real_samples.dtype, device=real_samples.device
    )
    # Expand alpha to match sample dims: (B, 1) -> (B, 1, 1) for 3D critic input.
    while alpha.dim() < real_samples.dim():
        alpha = alpha.unsqueeze(-1)
    interp = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interp = critic(interp)
    grad_outputs = torch.ones_like(d_interp)
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ─────────────────────────────────────────────────────────────────────────────
# EarlyStopping — verbatim port of cell 31's class.
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """EMD-based early stopping with checkpoint save/restore.

    Verbatim port of ``EarlyStopping`` from ``qgan_pennylane.ipynb`` cell 31.

    Monitors EMD (NOT critic loss). Saves best checkpoint when EMD improves;
    triggers stop after ``patience`` eval cycles without improvement, then
    reloads the best checkpoint into the model.

    Args:
        patience: Eval cycles without improvement before stopping. Default 50.
        warmup_epochs: Epochs before monitoring begins. Default 100.
        checkpoint_path: Path for best checkpoint file.
    """

    def __init__(
        self,
        patience: int = 50,
        warmup_epochs: int = 100,
        checkpoint_path: str = "best_checkpoint.pt",
    ) -> None:
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.checkpoint_path = checkpoint_path
        self.best_emd = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def check(self, epoch, emd, model, mu, sigma):
        """Check EMD and save checkpoint if improved.

        Returns True if training should stop (triggers checkpoint reload).
        Mirrors cell 31 verbatim.
        """
        # Skip during warmup
        if epoch < self.warmup_epochs:
            return False

        # Any decrease counts as improvement (no min_delta)
        if emd < self.best_emd:
            self.best_emd = emd
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(epoch, emd, model, mu, sigma)
            print(f"  [ES] New best EMD: {emd:.6f} at epoch {epoch + 1}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n[Early Stopping] Triggered at epoch {epoch + 1}")
                print(
                    f"  Best EMD: {self.best_emd:.6f} at epoch {self.best_epoch + 1}"
                )
                print(
                    f"  Patience exhausted: {self.counter}/{self.patience} "
                    f"eval cycles without improvement"
                )
                self._load_checkpoint(model)
                return True

        return False

    def _save_checkpoint(self, epoch, emd, model, mu, sigma):
        """Save model state to single checkpoint file (cell 31 verbatim)."""
        checkpoint = {
            "epoch": epoch,
            "emd": emd,
            "params_pqc": model.params_pqc.detach().clone(),
            "critic_state": model.critic.state_dict(),
            "c_optimizer": model.c_optimizer.state_dict(),
            "g_optimizer": model.g_optimizer.state_dict(),
            "mu": mu,
            "sigma": sigma,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def _load_checkpoint(self, model):
        """Load best checkpoint and restore model state (cell 31 verbatim)."""
        checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        model.params_pqc.data = checkpoint["params_pqc"]
        model.critic.load_state_dict(checkpoint["critic_state"])
        model.c_optimizer.load_state_dict(checkpoint["c_optimizer"])
        model.g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        # Re-register params_pqc with generator optimizer (cell 31 line).
        model.g_optimizer.param_groups[0]["params"] = [model.params_pqc]
        print(
            f"  [ES] Loaded best checkpoint from epoch "
            f"{checkpoint['epoch'] + 1} (EMD: {checkpoint['emd']:.6f})"
        )


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
