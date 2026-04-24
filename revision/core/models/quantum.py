"""PQC generator: data re-uploading ansatz with strongly-entangled Rot layers."""
from __future__ import annotations
from typing import Callable  # noqa: F401  (reserved for 08-03 QNode typing)
import torch
import torch.nn as nn


class QuantumGenerator(nn.Module):
    """Quantum generator with PAR_LIGHT conditioning hook.

    Signature preserves the notebook's qGAN.define_generator_circuit +
    define_generator_model contract. Filled in by plan 08-03.
    """

    def __init__(
        self,
        num_qubits: int = 5,
        num_layers: int = 4,
        window_length: int = 10,
        diff_method: str = "backprop",
    ) -> None:
        super().__init__()
        raise NotImplementedError("Filled in by plan 08-03")

    def count_params(self) -> int:
        """Return total PQC parameter count (IQP + entangled layers + final rotations).
        Filled in by plan 08-03."""
        raise NotImplementedError("Filled in by plan 08-03")

    def encoding_layer(self, noise_params: torch.Tensor) -> None:
        """IQP encoding: RZ(noise_params[i]) on wire i for all qubits.
        Filled in by plan 08-03."""
        raise NotImplementedError("Filled in by plan 08-03")

    def generator_circuit(self, noise_params: torch.Tensor, params_pqc: torch.Tensor):
        """Full QNode body.

        Hadamards -> IQP encoding -> strongly-entangled layers with
        Rot(phi, theta, lambda) + CNOT range pattern -> final RX/RY measurement
        prep -> return [<X_0>, <Z_0>, ..., <X_n>, <Z_n>].

        Filled in by plan 08-03.
        """
        raise NotImplementedError("Filled in by plan 08-03")

    def forward(
        self,
        noise_params: torch.Tensor,
        par_light: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generator forward pass.

        If ``par_light`` is provided, apply PAR_LIGHT conditioning hook
        (v1.1 Phase 7). Filled in by plan 08-03.
        """
        raise NotImplementedError("Filled in by plan 08-03")
