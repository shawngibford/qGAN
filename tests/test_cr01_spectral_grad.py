"""CR-01 regression: differentiable device-resident spectral PSD loss.

The pre-fix ``_spectral_psd_loss`` ran ``scipy.signal.welch`` on detached numpy
buffers and re-anchored a constant to ``fake.var()`` — so the spectral term
carried no real gradient w.r.t. the generator params. This suite asserts:

    - ``_spectral_psd_loss`` returns a ``torch.Tensor`` (not a python float),
      is finite, and carries ``grad_fn`` when ``fake.requires_grad``
    - a non-zero gradient flows into a leaf parameter through the loss
    - with ``spectral_loss_weight=0.0`` the call-site branch is skipped
      (default byte-unchanged — T-13-02): no scipy.welch import remains
"""
from __future__ import annotations

import inspect

import torch

from revision.core.training import _spectral_psd_loss


def test_returns_tensor_with_grad_fn_when_fake_requires_grad():
    fake = torch.randn(64, requires_grad=True)
    real = torch.randn(64)
    out = _spectral_psd_loss(fake, real)
    assert isinstance(out, torch.Tensor), "loss must be a torch.Tensor"
    assert out.dim() == 0, "loss must be a scalar"
    assert torch.isfinite(out), "loss must be finite"
    assert out.grad_fn is not None, "loss must carry grad_fn (differentiable)"


def test_nonzero_gradient_flows_into_params():
    # Tiny synthetic 'params' leaf feeding a linear map into the fake signal.
    params = torch.nn.Parameter(torch.randn(16))
    fake = params.repeat(4)  # (64,) derived from params
    real = torch.randn(64)
    loss = _spectral_psd_loss(fake, real)
    loss.backward()
    assert params.grad is not None, "no gradient reached params"
    assert params.grad.abs().sum() > 0, "gradient through PSD loss is zero"


def test_real_branch_is_detached():
    fake = torch.randn(32, requires_grad=True)
    real = torch.randn(32, requires_grad=True)
    loss = _spectral_psd_loss(fake, real)
    loss.backward()
    # Real is the target — no gradient must flow into it.
    assert real.grad is None


def test_no_scipy_welch_import_remains():
    src = inspect.getsource(_spectral_psd_loss)
    assert "welch" not in src, "scipy.signal.welch must be gone (CR-01)"
    assert "torch.fft.rfft" in src, "must use torch.fft.rfft periodogram"


def test_call_site_guard_preserved():
    """spectral_loss_weight=0.0 default must skip the spectral branch."""
    from revision.core import training as tr

    src = inspect.getsource(tr.train_wgan_gp)
    assert "if spectral_loss_weight > 0.0" in src, (
        "call-site guard must remain so weight=0.0 stays byte-unchanged"
    )
