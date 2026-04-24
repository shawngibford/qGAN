"""PQC generator: data re-uploading ansatz with strongly-entangled Rot layers.

Extracted verbatim from ``qgan_pennylane.ipynb`` cell 26 (the ``qGAN`` class).
v1.0/v1.1 decisions preserved:
    - IQP-style RZ encoding (no Rot-gate redundancy)
    - Strongly-entangled Rot(phi, theta, omega) layers
    - Range-based CNOT pattern: ``r = (layer % (num_qubits - 1)) + 1``
    - Final RX + RY measurement-prep rotations
    - PauliX + PauliZ expectation values on every qubit (output dim = 2 * num_qubits)
    - ``diff_method="backprop"`` (v1.1 Phase 5 — parameter-shift has broadcasting bugs
      per PennyLane issue #4462)
    - Initialization scale 0.5 (v1.1 notebook value)

The ``par_light`` argument on ``forward`` is a conditioning hook preserved from v1.1
Phase 7. The final notebook runs in unconditioned mode (cell 65 ``RUN_NAME =
"unconditioned_wgan"``) so cell 26 does NOT modulate generator output with PAR_LIGHT.
The kwarg remains on the API so future phases can re-enable conditioning without
breaking callers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml


# v1.1 Phase 4 HPO notebook cell 26: ``torch.randn(...) * 0.5``.
_INIT_SCALE = 0.5


class QuantumGenerator(nn.Module):
    """Quantum generator with PAR_LIGHT conditioning hook.

    Matches ``qGAN.define_generator_circuit`` + ``qGAN.define_generator_model``
    from ``qgan_pennylane.ipynb`` cell 26.
    """

    def __init__(
        self,
        num_qubits: int = 5,
        num_layers: int = 4,
        window_length: int = 10,
        diff_method: str = "backprop",
    ) -> None:
        super().__init__()

        # v1.0 invariant: window_length = 2 * num_qubits (PauliX + PauliZ per wire).
        assert window_length == 2 * num_qubits, (
            f"window_length must equal 2 * num_qubits "
            f"(got window_length={window_length}, num_qubits={num_qubits})"
        )

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.window_length = window_length
        self.diff_method = diff_method

        # IQP (num_qubits) + num_layers * (num_qubits * 3 Rot params) + final RX/RY (num_qubits * 2)
        self.num_params = (
            num_qubits + num_layers * (num_qubits * 3) + num_qubits * 2
        )

        # Quantum device (statevector simulator, no shots — match cell 26).
        self.dev = qml.device("default.qubit", wires=num_qubits, shots=None)

        # Trainable PQC parameters — notebook uses ``torch.randn(...) * 0.5``.
        self.params_pqc = nn.Parameter(
            torch.randn(self.num_params, dtype=torch.float32) * _INIT_SCALE,
            requires_grad=True,
        )

        # QNode bound to ``generator_circuit`` with backprop diff (v1.1 Phase 5).
        self.qnode = qml.QNode(
            self.generator_circuit,
            self.dev,
            interface="torch",
            diff_method=diff_method,
        )

    def count_params(self) -> int:
        """Return total PQC parameter count.

        For (num_qubits=5, num_layers=4): 5 + 4*15 + 10 = 75.
        """
        return self.num_params

    def encoding_layer(self, noise_params: torch.Tensor) -> None:
        """IQP noise encoding: RZ(noise_params[i]) on wire i.

        Matches ``qGAN.encoding_layer`` from cell 26. Supports both 1D noise
        (shape ``(num_qubits,)``) and batched noise (shape ``(num_qubits,
        batch)``) — PennyLane broadcasts over the trailing dim when
        ``diff_method='backprop'``.
        """
        # noise_params may be 1D (num_qubits,) or 2D (num_qubits, batch)
        n = min(
            noise_params.shape[0] if noise_params.dim() >= 1 else len(noise_params),
            self.num_qubits,
        )
        for i in range(n):
            qml.RZ(noise_params[i], wires=i)

    def generator_circuit(
        self, noise_params: torch.Tensor, params_pqc: torch.Tensor
    ):
        """Full QNode body — verbatim port of ``qGAN.define_generator_circuit``.

        Structure:
            1. Hadamard on every qubit (superposition)
            2. IQP RZ encoding with trainable params_pqc (num_qubits consumed)
            3. IQP noise injection via ``encoding_layer(noise_params)``
            4. ``num_layers`` of:
                 - per-qubit ``qml.Rot(phi, theta, omega)``  (3 params/qubit)
                 - range-based CNOT: r = (layer % (num_qubits - 1)) + 1
                   target = (q + r) % num_qubits
            5. Final RX + RY measurement-prep rotations
            6. Return tuple of PauliX and PauliZ expectations for every qubit.

        Returned ordering (matches cell 26):
            (<X_0>, <Z_0>, <X_1>, <Z_1>, ..., <X_{n-1}>, <Z_{n-1}>)
        """
        idx = 0

        # Step 1: Hadamard initialization for superposition.
        for qubit in range(self.num_qubits):
            qml.Hadamard(wires=qubit)

        # Step 2: IQP encoding with parameterized RZ rotations.
        for qubit in range(self.num_qubits):
            if idx < len(params_pqc):
                qml.RZ(phi=params_pqc[idx], wires=qubit)
                idx += 1

        # Step 3: Apply noise encoding (IQP-style).
        self.encoding_layer(noise_params)

        # Step 4: Strongly Entangled Layers.
        for layer in range(self.num_layers):
            # Rot(phi, theta, omega) per qubit.
            for qubit in range(self.num_qubits):
                if idx + 2 < len(params_pqc):
                    qml.Rot(
                        phi=params_pqc[idx],
                        theta=params_pqc[idx + 1],
                        omega=params_pqc[idx + 2],
                        wires=qubit,
                    )
                    idx += 3

            # Range-based entangling CNOTs.
            if self.num_qubits > 1:
                range_param = (layer % (self.num_qubits - 1)) + 1
                for qubit in range(self.num_qubits):
                    target_qubit = (qubit + range_param) % self.num_qubits
                    qml.CNOT(wires=[qubit, target_qubit])

        # Step 5: Final measurement-preparation rotations.
        for qubit in range(self.num_qubits):
            if idx + 1 < len(params_pqc):
                qml.RX(phi=params_pqc[idx], wires=qubit)
                idx += 1
                qml.RY(phi=params_pqc[idx], wires=qubit)
                idx += 1

        # Step 6: Pauli-X and Pauli-Z expectations on each qubit.
        measurements = []
        for i in range(self.num_qubits):
            measurements.append(qml.expval(qml.PauliX(i)))
            measurements.append(qml.expval(qml.PauliZ(i)))

        return (*measurements,)

    def forward(
        self,
        noise_params: torch.Tensor,
        par_light: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generator forward pass.

        Accepts:
            - 1D ``noise_params`` of shape ``(num_qubits,)`` → returns
              ``(window_length,)`` tensor.
            - 2D batched ``noise_params`` of shape ``(num_qubits, batch)`` →
              returns ``(batch, window_length)`` tensor (matches notebook
              training-loop ``torch.stack(list(results)).T``).

        ``par_light``: reserved conditioning hook (v1.1 Phase 7). The final v1.1
        notebook runs in unconditioned mode so the generator does not modulate
        with PAR_LIGHT. When provided, it is currently a no-op so the API is
        forward-compatible with future conditioning phases. Raising or ignoring
        silently was the v1.1 choice — we ignore silently to match the
        unconditioned run's observable behavior.
        """
        results = self.qnode(noise_params, self.params_pqc)
        stacked = torch.stack(list(results))
        # Batched: stacked shape is (window_length, batch) → transpose to (batch, window_length).
        # Unbatched: stacked shape is (window_length,) → already correct.
        if stacked.dim() == 2:
            stacked = stacked.T
        # par_light hook reserved for future conditioning phases — currently a passthrough.
        _ = par_light
        return stacked
