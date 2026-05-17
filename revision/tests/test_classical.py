"""RED/GREEN test for Phase 10 Task 1 — classical WGAN-GP generators.

Asserts the train_wgan_gp interface contract (RESEARCH Pitfall 1):
  - single live nn.Parameter named params_pqc
  - count_params() == params_pqc.numel() == sum(p.numel() for p in parameters())
  - exact param counts 74/73/78 (MLP/CNN/LSTM), all in [71,79]
  - forward((5,B)) -> (B,10)
  - one Adam step on [params_pqc] mutates params_pqc (autograd live)

Run: <main-repo>/qgan_env/bin/python -m pytest revision/tests/test_classical.py
or as a plain script:   ... revision/tests/test_classical.py
"""
import torch

from revision.core.models.classical import (
    WGANMLPGenerator,
    WGANCNNGenerator,
    WGANLSTMGenerator,
)


def test_classical_generators_contract():
    mm, cc, ll = WGANMLPGenerator(), WGANCNNGenerator(), WGANLSTMGenerator()

    assert mm.count_params() == 74 == sum(p.numel() for p in mm.parameters()), mm.count_params()
    assert cc.count_params() == 73 == sum(p.numel() for p in cc.parameters()), cc.count_params()
    assert ll.count_params() == 78 == sum(p.numel() for p in ll.parameters()), ll.count_params()

    for g in (mm, cc, ll):
        assert 71 <= g.count_params() <= 79, g.count_params()
        assert g.num_qubits == 5 and g.window_length == 10
        assert sum(1 for _ in g.parameters()) == 1, "must be exactly one nn.Parameter (params_pqc)"
        assert g.params_pqc.requires_grad
        o = g(torch.randn(5, 12))
        assert tuple(o.shape) == (12, 10), o.shape
        before = g.params_pqc.detach().clone()
        opt = torch.optim.Adam([g.params_pqc], lr=0.1)
        loss = g(torch.randn(5, 4)).pow(2).mean()
        loss.backward()
        opt.step()
        assert not torch.allclose(before, g.params_pqc.detach()), \
            "params_pqc not updated -> autograd detached (Pitfall 1)"


def test_barrel_imports():
    from revision.core.models import classical, nonadversarial  # noqa: F401


if __name__ == "__main__":
    test_classical_generators_contract()
    test_barrel_imports()
    print("OK classical 74/73/78, single-param, autograd-live")
