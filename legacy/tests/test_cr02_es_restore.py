"""CR-02 regression: EarlyStopping restore is device/dtype consistent.

The pre-fix ``_load_checkpoint`` did an un-mapped ``torch.load`` + raw
``.data =`` assignment, so a checkpoint saved on one device/dtype could land
params (and optimizer state) on the wrong device — silent NaN/garbage in any
future early-stopped run (T-13-03). This suite asserts that after a restore:

    - ``model.params_pqc`` keeps the live device + dtype
    - every optimizer-state tensor sits on the live device
    - the same holds on MPS (skipped when MPS is unavailable)
"""
from __future__ import annotations

import torch
import pytest

from core.training import EarlyStopping, _ESAdapter


class _TinyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 1)

    def forward(self, x):  # pragma: no cover - not exercised
        return self.lin(x)


class _TinyGen:
    """Minimal stand-in exposing the params_pqc attribute the adapter proxies."""

    def __init__(self, device, dtype):
        self.params_pqc = torch.nn.Parameter(
            torch.randn(8, device=device, dtype=dtype)
        )


def _build_adapter(device, dtype):
    gen = _TinyGen(device, dtype)
    critic = _TinyCritic().to(device=device, dtype=dtype)
    c_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    g_opt = torch.optim.Adam([gen.params_pqc], lr=1e-3)
    return _ESAdapter(gen, critic, c_opt, g_opt), gen, critic, c_opt, g_opt


def _step_optimizers(critic, c_opt, g_opt, gen):
    # One fake step so optimizer .state has exp_avg / exp_avg_sq tensors.
    c_loss = critic(torch.zeros(2, 4, device=gen.params_pqc.device,
                                dtype=gen.params_pqc.dtype)).sum()
    c_opt.zero_grad()
    c_loss.backward()
    c_opt.step()
    g_loss = (gen.params_pqc ** 2).sum()
    g_opt.zero_grad()
    g_loss.backward()
    g_opt.step()


def _restore_roundtrip(device, tmp_path):
    dtype = torch.float32
    ckpt_path = str(tmp_path / "best_checkpoint.pt")

    adapter, gen, critic, c_opt, g_opt = _build_adapter(device, dtype)
    _step_optimizers(critic, c_opt, g_opt, gen)

    es = EarlyStopping(patience=1, warmup_epochs=0, checkpoint_path=ckpt_path)
    es._save_checkpoint(0, 0.123, adapter, mu=0.0, sigma=1.0)

    # Mutate live params so the restore must actually overwrite them.
    with torch.no_grad():
        gen.params_pqc.add_(1.0)

    es._load_checkpoint(adapter)

    assert adapter.params_pqc.device.type == device.type, (
        f"params on wrong device: {adapter.params_pqc.device} != {device}"
    )
    assert adapter.params_pqc.dtype == dtype, (
        f"params dtype drifted: {adapter.params_pqc.dtype} != {dtype}"
    )
    for opt in (c_opt, g_opt):
        for st in opt.state.values():
            for v in st.values():
                if torch.is_tensor(v):
                    assert v.device.type == device.type, (
                        f"optimizer state tensor on {v.device}, "
                        f"expected {device}"
                    )
    # g_optimizer must still point at the live params object.
    assert g_opt.param_groups[0]["params"][0] is adapter.params_pqc


def test_restore_device_dtype_consistent_cpu(tmp_path):
    _restore_roundtrip(torch.device("cpu"), tmp_path)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)
def test_restore_device_dtype_consistent_mps(tmp_path):
    _restore_roundtrip(torch.device("mps"), tmp_path)
