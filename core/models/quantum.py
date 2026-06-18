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

    #: Allowed entangling-CNOT topologies. ``"range"`` is the v1.0/v1.1 default
    #: (wrap-around range pattern, byte-identical to pre-Phase-13 code).
    _TOPOLOGIES = ("range", "linear")

    #: Allowed circuit variants (Phase 14 D-14-01/04/07). ``"default_75"`` is
    #: the v1.0/v1.1 byte-frozen circuit: IQP encoding + SEL layers + final
    #: RX **and** RY per qubit (5 + L*15 + 2*5 = 75 at L=4). ``"iqp_sel_55"``
    #: is the recovered canonical paper circuit reconstructed from
    #: ``best_checkpoint.pt`` (D-14-02): IQP encoding + SEL layers + final
    #: RX-**only** per qubit (5 + L*15 + 5 = 55 at L=3). The 55-param variant
    #: is NON-default — selecting it never perturbs the frozen default tape
    #: (T-14-02). Mirrors the eager-validation shape of :attr:`_TOPOLOGIES`.
    _CIRCUIT_IDS = ("default_75", "iqp_sel_55")

    #: The balanced 2|3 bipartition used by :meth:`introspect` — recorded here
    #: so plan-03's JSON metadata can pin the exact partition (D-13-09).
    INTROSPECT_BIPARTITION = ((0, 1), (2, 3, 4))

    def __init__(
        self,
        num_qubits: int = 5,
        num_layers: int = 4,
        window_length: int = 10,
        diff_method: str = "backprop",
        topology: str = "range",
        circuit_id: str = "default_75",
    ) -> None:
        super().__init__()

        # ARCH-01: select the entangling-CNOT wiring. Default "range" keeps the
        # pre-Phase-13 circuit byte-identical (T-13-01). Validate eagerly with
        # an argparse-style message so bad config fails at construction.
        if topology not in self._TOPOLOGIES:
            raise ValueError(
                f"Unknown topology {topology!r}; expected one of "
                f"{self._TOPOLOGIES}"
            )

        # D-14-01/04/07: select the circuit variant. Default "default_75"
        # keeps the v1.0/v1.1 circuit byte-identical (T-14-02 — Phases 8-13
        # were baselined on it). "iqp_sel_55" is the recovered canonical paper
        # circuit (final RX-only, no RY). Eager validation mirrors topology.
        if circuit_id not in self._CIRCUIT_IDS:
            raise ValueError(
                f"Unknown circuit_id {circuit_id!r}; expected one of "
                f"{self._CIRCUIT_IDS}"
            )

        # v1.0 invariant: window_length = 2 * num_qubits (PauliX + PauliZ per wire).
        assert window_length == 2 * num_qubits, (
            f"window_length must equal 2 * num_qubits "
            f"(got window_length={window_length}, num_qubits={num_qubits})"
        )

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.window_length = window_length
        self.diff_method = diff_method
        self.topology = topology
        self.circuit_id = circuit_id

        # Param formula by variant:
        #   default_75 : IQP(num_qubits) + L*(num_qubits*3) + final RX+RY (num_qubits*2)
        #   iqp_sel_55 : IQP(num_qubits) + L*(num_qubits*3) + final RX-only (num_qubits)
        # The "default_75" branch is LITERALLY the pre-Phase-14 expression so
        # the frozen default circuit is byte-identical (T-14-02).
        _final_rot_factor = 2 if circuit_id == "default_75" else 1
        self.num_params = (
            num_qubits
            + num_layers * (num_qubits * 3)
            + num_qubits * _final_rot_factor
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

        # INTRO-03: read-only introspection QNode (same device + interface).
        # It clones Steps 1-5 of generator_circuit and returns VN-entropy +
        # purity on the {0,1}|{2,3,4} bipartition instead of Pauli expvals.
        self._introspect_qnode = qml.QNode(
            self._introspect_circuit,
            self.dev,
            interface="torch",
            diff_method="backprop",
        )

    def count_params(self) -> int:
        """Return total PQC parameter count.

        ``default_75`` (num_qubits=5, num_layers=4): 5 + 4*15 + 10 = 75.
        ``iqp_sel_55`` (num_qubits=5, num_layers=3): 5 + 3*15 + 5 = 55
        (recovered canonical paper circuit, D-14-01).
        """
        return self.num_params

    def last_param_index(self, noise_params: torch.Tensor) -> int:
        """Structural-introspection helper for the D-14-07 equivalence gate.

        Runs a single forward pass and returns how many ``params_pqc``
        entries the bound circuit body actually consumed. The reconstructed
        ``iqp_sel_55`` circuit must consume exactly 55 (Phase-14 Task 2 gate);
        ``default_75`` consumes 75. Read-only — never mutates state.
        """
        self.qnode(noise_params, self.params_pqc)
        return int(self._last_idx)

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

            # Entangling CNOTs — topology-selectable (ARCH-01). The "range"
            # branch body below is LITERALLY the pre-Phase-13 block so the
            # drawn tape is byte-identical for the default (T-13-01).
            if self.num_qubits > 1:
                if self.topology == "range":
                    range_param = (layer % (self.num_qubits - 1)) + 1
                    for qubit in range(self.num_qubits):
                        target_qubit = (qubit + range_param) % self.num_qubits
                        qml.CNOT(wires=[qubit, target_qubit])
                elif self.topology == "linear":
                    for q in range(self.num_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])

        # Step 5: Final measurement-preparation rotations.
        # default_75 — RX+RY per qubit (the body below is LITERALLY the
        # pre-Phase-14 block so the frozen default tape is byte-identical,
        # T-14-02). iqp_sel_55 — RX-only per qubit (recovered canonical
        # paper circuit, D-14-01: 5 + L*15 + 5 = 55).
        if self.circuit_id == "default_75":
            for qubit in range(self.num_qubits):
                if idx + 1 < len(params_pqc):
                    qml.RX(phi=params_pqc[idx], wires=qubit)
                    idx += 1
                    qml.RY(phi=params_pqc[idx], wires=qubit)
                    idx += 1
        elif self.circuit_id == "iqp_sel_55":
            for qubit in range(self.num_qubits):
                if idx < len(params_pqc):
                    qml.RX(phi=params_pqc[idx], wires=qubit)
                    idx += 1

        # Record how many params the body consumed (D-14-07 structural gate).
        self._last_idx = idx

        # Step 6: Pauli-X and Pauli-Z expectations on each qubit.
        measurements = []
        for i in range(self.num_qubits):
            measurements.append(qml.expval(qml.PauliX(i)))
            measurements.append(qml.expval(qml.PauliZ(i)))

        return (*measurements,)

    def _introspect_circuit(
        self, noise_params: torch.Tensor, params_pqc: torch.Tensor
    ):
        """Read-only entanglement probe (INTRO-03).

        Steps 1-5 are cloned VERBATIM from :meth:`generator_circuit` (including
        the topology switch) so the prepared state is identical to the one the
        generator produces; only the Step-6 measurement is replaced with
        Von-Neumann entropy + purity on the balanced 2|3 bipartition
        ``{0,1}|{2,3,4}`` (recorded in :attr:`INTROSPECT_BIPARTITION`).
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
            for qubit in range(self.num_qubits):
                if idx + 2 < len(params_pqc):
                    qml.Rot(
                        phi=params_pqc[idx],
                        theta=params_pqc[idx + 1],
                        omega=params_pqc[idx + 2],
                        wires=qubit,
                    )
                    idx += 3

            if self.num_qubits > 1:
                if self.topology == "range":
                    range_param = (layer % (self.num_qubits - 1)) + 1
                    for qubit in range(self.num_qubits):
                        target_qubit = (qubit + range_param) % self.num_qubits
                        qml.CNOT(wires=[qubit, target_qubit])
                elif self.topology == "linear":
                    for q in range(self.num_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])

        # Step 5: Final measurement-preparation rotations (variant-aware —
        # mirrors generator_circuit so the probed state matches the prepared
        # state for whichever circuit_id is selected; default_75 branch is
        # the byte-frozen RX+RY block, T-14-02).
        if self.circuit_id == "default_75":
            for qubit in range(self.num_qubits):
                if idx + 1 < len(params_pqc):
                    qml.RX(phi=params_pqc[idx], wires=qubit)
                    idx += 1
                    qml.RY(phi=params_pqc[idx], wires=qubit)
                    idx += 1
        elif self.circuit_id == "iqp_sel_55":
            for qubit in range(self.num_qubits):
                if idx < len(params_pqc):
                    qml.RX(phi=params_pqc[idx], wires=qubit)
                    idx += 1

        # Step 6 (replaced): entanglement diagnostics on the balanced
        # bipartition {0,1}|{2,3,4} (== INTROSPECT_BIPARTITION).
        return (
            qml.vn_entropy(wires=[0, 1]),
            qml.purity(wires=[0, 1]),
        )

    def introspect(self, noise_vec: torch.Tensor):
        """Return ``(vn_entropy, purity)`` for the 2|3 bipartition.

        Both are python floats. VN-entropy is bounded in ``[0, ln 4]`` and
        purity in ``[1/4, 1]`` for the 2-qubit subsystem ``{0,1}``. Apple MPS
        has no float64 statevector path (RESEARCH Pitfall 6), so params and
        noise are forced onto CPU before the read-only QNode is evaluated.
        Caller is expected to wrap this in ``torch.no_grad()``.
        """
        params_cpu = self.params_pqc.detach().cpu()
        if torch.is_tensor(noise_vec):
            noise_cpu = noise_vec.detach().cpu()
        else:
            noise_cpu = torch.as_tensor(noise_vec)
        ent, pur = self._introspect_qnode(noise_cpu, params_cpu)
        return float(ent), float(pur)

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
