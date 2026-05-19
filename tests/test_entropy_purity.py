"""INTRO-03 regression: introspect() entropy/purity bounds + API.

Covers:
    - introspect(noise_vec) returns two python floats
    - Von Neumann entropy in [0, ln 4] for the 2|3 bipartition {0,1}|{2,3,4}
    - purity in [0.25, 1] for the same bipartition
    - the recorded bipartition metadata constant is {0,1}|{2,3,4}
"""
from __future__ import annotations

import math

import torch

from revision.core.models.quantum import QuantumGenerator

_EPS = 1e-6


def test_introspect_returns_two_floats():
    torch.manual_seed(0)
    g = QuantumGenerator()
    noise = torch.linspace(0.1, 1.3, 5, dtype=torch.float32)
    with torch.no_grad():
        ent, pur = g.introspect(noise)
    assert isinstance(ent, float)
    assert isinstance(pur, float)


def test_entropy_within_bounds():
    torch.manual_seed(0)
    g = QuantumGenerator()
    noise = torch.linspace(0.1, 1.3, 5, dtype=torch.float32)
    with torch.no_grad():
        ent, _ = g.introspect(noise)
    # 2-qubit subsystem -> max VN entropy = ln(4).
    assert -_EPS <= ent <= math.log(4) + _EPS, f"vn_entropy out of bounds: {ent}"


def test_purity_within_bounds():
    torch.manual_seed(0)
    g = QuantumGenerator()
    noise = torch.linspace(0.1, 1.3, 5, dtype=torch.float32)
    with torch.no_grad():
        _, pur = g.introspect(noise)
    # 2-qubit subsystem -> purity in [1/4, 1].
    assert 0.25 - _EPS <= pur <= 1.0 + _EPS, f"purity out of bounds: {pur}"


def test_bipartition_metadata_recorded():
    # The {0,1}|{2,3,4} balanced bipartition is recorded for plan-03 JSON.
    assert QuantumGenerator.INTROSPECT_BIPARTITION == ((0, 1), (2, 3, 4))
