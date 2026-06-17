"""ARCH-01 regression: topology selectability + byte-unchanged default.

Covers:
    - count_params() == 75 for the default (V1) and V3 (linear, 4 layers)
    - count_params() == 135 for V2 (8 layers, range)
    - default-topology fixed-seed forward equals a reference captured from the
      pre-change code (byte-unchanged default — T-13-01 mitigation)
    - topology="linear" emits ONLY nearest-neighbour CNOTs (no wrap-around)
    - topology="range" (default) emits the wrap-around pattern unchanged
    - invalid topology raises a clear ValueError
"""
from __future__ import annotations

import math

import pytest
import torch

from revision.core.models.quantum import QuantumGenerator

# Reference forward vector captured from the PRE-CHANGE quantum.py default path
# (torch.manual_seed(0); noise = torch.linspace(0.1, 1.3, 5, float32)).
# Any drift here means the default circuit changed -> Phases 8-12 break.
_REFERENCE_FORWARD = [
    -0.009733603877014163,
    -0.05267814272397531,
    0.14571186235125544,
    0.03954846298040593,
    -0.1332692710737413,
    -0.16811740807060788,
    0.16277590625059618,
    0.18223508251336035,
    -0.1553363281964492,
    -0.053407368463505134,
]


def _fixed_noise() -> torch.Tensor:
    return torch.linspace(0.1, 1.3, 5, dtype=torch.float32)


def test_default_param_count_is_75():
    g = QuantumGenerator()
    assert g.count_params() == 75


def test_v2_param_count_is_135():
    g = QuantumGenerator(num_layers=8)
    assert g.count_params() == 135


def test_v3_linear_param_count_is_75():
    g = QuantumGenerator(num_layers=4, topology="linear")
    assert g.count_params() == 75


def test_default_topology_is_range():
    g = QuantumGenerator()
    assert g.topology == "range"


def test_default_forward_byte_unchanged():
    """Fixed-seed forward on the default path must match the pre-change ref."""
    torch.manual_seed(0)
    g = QuantumGenerator()
    with torch.no_grad():
        out = g(_fixed_noise())
    assert tuple(out.shape) == (10,)
    ref = torch.tensor(_REFERENCE_FORWARD, dtype=out.dtype)
    assert torch.allclose(out, ref, atol=1e-12, rtol=0.0), (
        f"default forward drifted from pre-change reference:\n"
        f"got={out.tolist()}\nref={_REFERENCE_FORWARD}"
    )


def _cnot_pairs(generator: QuantumGenerator, noise: torch.Tensor):
    """Return the ordered list of (control, target) CNOT wire pairs in the tape."""
    import pennylane as qml

    tape = qml.workflow.construct_tape(generator.qnode)(
        noise, generator.params_pqc
    )
    pairs = []
    for op in tape.operations:
        if op.name == "CNOT":
            pairs.append(tuple(int(w) for w in op.wires))
    return pairs


def test_linear_topology_emits_only_nearest_neighbour_cnots():
    g = QuantumGenerator(num_layers=4, topology="linear")
    pairs = _cnot_pairs(g, _fixed_noise())
    assert pairs, "expected at least one CNOT in the linear tape"
    # Every CNOT must be (q, q+1) with no wrap-around (target != 0 when ctrl=n-1).
    for c, t in pairs:
        assert t == c + 1, f"linear topology emitted non-NN CNOT ({c},{t})"
        assert not (c == g.num_qubits - 1), "linear topology must not wrap around"


def test_range_topology_emits_wraparound_cnots():
    g = QuantumGenerator(num_layers=4, topology="range")
    pairs = _cnot_pairs(g, _fixed_noise())
    # Layer 0: range_param = (0 % 4) + 1 = 1 -> includes a wrap-around
    # (qubit 4 -> target (4+1)%5 = 0).
    assert (4, 0) in pairs, (
        f"range topology must emit the wrap-around CNOT (4,0); got {pairs}"
    )


def test_invalid_topology_raises_valueerror():
    with pytest.raises(ValueError):
        QuantumGenerator(topology="full")
