"""RED/GREEN test for Phase 10 Task 2 — non-adversarial baselines (BASE-02).

Asserts:
  - VAEBaseline.count_params() in [540,580] == sum(p.numel() for p in parameters())
  - encode/decode/reparameterize/forward shapes
  - sample(n, gen) -> (n,10), NO *0.1 (quantum artifact) — checked by docstring + Wave-2 smoke
  - ARBaseline.fit sets phi(p,), sigma2>0, p==2; count_params()==3; sample(n,rng)->(n,10)
  - D-10-13: no ELBO loop / Adam / fit *loop* in nonadversarial.py (model defs only)
"""
import numpy as np
import torch

from core.models.nonadversarial import VAEBaseline, ARBaseline


def test_vae_contract():
    v = VAEBaseline()
    n = sum(p.numel() for p in v.parameters())
    assert v.count_params() == n, (v.count_params(), n)
    assert 540 <= n <= 580, n
    x = torch.randn(7, 10)
    xh, mu, lv = v(x)
    assert tuple(xh.shape) == (7, 10)
    assert tuple(mu.shape) == (7, 4)
    assert tuple(lv.shape) == (7, 4)
    mu2, lv2 = v.encode(x)
    assert tuple(mu2.shape) == (7, 4) and tuple(lv2.shape) == (7, 4)
    z = v.reparameterize(mu2, lv2)
    assert tuple(z.shape) == (7, 4)
    assert tuple(v.decode(z).shape) == (7, 10)
    s = v.sample(5, torch.Generator())
    assert tuple(s.shape) == (5, 10)


def test_ar_contract():
    rng = np.random.default_rng(0)
    a = ARBaseline()
    a.fit(rng.standard_normal(400).astype("float64"))
    assert a.phi.shape == (2,)
    assert a.p == 2
    assert a.count_params() == 3
    assert a.sigma2 > 0
    ss = a.sample(6, np.random.default_rng(1))
    assert ss.shape == (6, 10)


def test_no_training_loop_in_module():
    """D-10-13: nonadversarial.py holds model defs only — no ELBO/Adam/fit-loop."""
    import pathlib
    src = pathlib.Path("core/models/nonadversarial.py").read_text()
    assert "torch.optim" not in src, "no optimizer in core model file (D-10-13)"
    assert ".backward()" not in src, "no training loop in core model file (D-10-13)"


if __name__ == "__main__":
    test_vae_contract()
    test_ar_contract()
    test_no_training_loop_in_module()
    print("OK nonadversarial VAE + AR")
