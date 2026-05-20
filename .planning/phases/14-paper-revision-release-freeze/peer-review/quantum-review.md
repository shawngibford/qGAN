# Quantum-Circuit Peer Review

**Reviewer scope:** quantum-circuit specialist (top-tier journal rigor).
**Target circuits:** `default_75`, `iqp_sel_55`, `V1`, `V2`, `V3`.
**Verdict:** **ACCEPT with MINOR REVISIONS.** The quantum-circuit claims are
substantively defensible. Three MINOR items and one notable absence
(barren-plateau analysis for V2) warrant correction or acknowledgement.

---

## 1. IQP-encoding nomenclature — MINOR

**Finding.** The paper-facing label for the encoding sub-block is
"IQP (Hadamard + RZ per qubit)" (`methods_full.md` §2.a, all five circuit
tables). Code (`revision/core/models/quantum.py:194-206`) implements the
sub-block as: H on every qubit → trainable `RZ(theta_i)` per qubit → noise
`RZ(z_i)` per qubit. The gate sequence is `H ⊗ RZ ⊗ RZ` — all diagonal in the
computational basis after the Hadamards, no further Hadamards before the
variational block.

This is a **first-order (depth-1, single-qubit) IQP feature map** in the sense
of Havlíček et al. 2019 / Schuld–Killoran ZZ-feature-map literature (the
single-qubit diagonal sector, no two-qubit ZZ phases). It does qualify as IQP
in the broad sense — the entire encoding stays in the IQP gate set
(`{H, RZ, CZ, CRZ}`) and the data-dependent layer is diagonal — but it is the
**minimum** non-trivial IQP encoding. The richer "ZZ-feature-map" form with
CZ/CRZ couplings is absent.

**Recommendation.** Either (a) add a one-sentence footnote in §2.a clarifying
the encoding is the depth-1, single-qubit sector of the IQP family (no ZZ
couplings); or (b) drop "IQP" and call it what it literally is — a Hadamard
basis change followed by single-qubit RZ data injection. Current wording is
not wrong but invites a reviewer to expect a richer feature map.

## 2. "Data re-uploading" docstring misnomer — MINOR (code-only, not in paper)

**Finding.** `revision/core/models/quantum.py:1` opens with:

> `PQC generator: data re-uploading ansatz with strongly-entangled Rot layers.`

A grep confirms `encoding_layer(noise_params)` is called **exactly once** per
circuit (line 206 in `generator_circuit`, mirrored at line 287 in the
introspection clone). There is no re-injection of `noise_params` between SEL
layers. This is therefore **not** a data re-uploading circuit in the
Pérez-Salinas et al. 2020 sense — it is a single-shot encoding.

**Critical: this misnomer never leaks into paper-facing text.** I grepped
`revision/docs/*.md`, `revision/results/methods_full.json`, and
`revision/results/*_config_lock.json`: none use "re-uploading" or "re-upload".
The paper consistently describes the encoding as IQP (single layer). Only the
internal source-file docstring is misleading.

**Recommendation.** Edit the `quantum.py` docstring so future maintainers /
reviewers reading the source do not propagate the misnomer. One-line fix.

## 3. Strongly-Entangling-Layers equivalence — VERIFIED ✓

**Finding.** I read `qml.StronglyEntanglingLayers.compute_decomposition`
(`pennylane/templates/layers/strongly_entangling.py:195-249`) and compared
to `quantum.py:209-232`.

PennyLane's per-layer body:
1. `Rot(weights[l,i,0], weights[l,i,1], weights[l,i,2])` per qubit `i`.
2. For each `i`, `CNOT(wires=[i, (i + ranges[l]) % n_wires])`.

Default `ranges`: `tuple((l % (n_wires - 1)) + 1 for l in range(n_layers))`.

Project's range-topology body (`quantum.py:211-229`):
1. `Rot(params[idx], params[idx+1], params[idx+2])` per qubit, idx+=3.
2. `range_param = (layer % (num_qubits - 1)) + 1`, then for each `qubit`,
   `CNOT(wires=[qubit, (qubit + range_param) % num_qubits])`.

These are **structurally byte-identical** (modulo the trivial parameter-layout
choice — quantum.py uses a flat tensor with a running `idx` cursor, PL uses a
3D `weights[l, i, k]` indexed array; both consume `3 * n_wires * n_layers`
params per SEL stack). The SEL claim is therefore correct for all four
range-topology variants (default_75, iqp_sel_55, V1, V2).

## 4. Range vs linear topology — VERIFIED ✓

**Finding.** `revision/run_matched2000.py:117-124` defines the source-of-truth
`_QUANTUM_ANSATZ` dict:

```
"V1": {"num_layers": 4, "topology": "range",  ..., "parameter_count": 75},
"V2": {"num_layers": 8, "topology": "range",  ..., "parameter_count": 135},
"V3": {"num_layers": 4, "topology": "linear", ..., "parameter_count": 75},
```

`run_circuit_diagrams.py` builds the QNode via `QuantumGenerator(topology=...)`
and renders via `qml.draw_mpl` (line 537) — no bespoke matplotlib gate
drawing. quantum.py:225-232 has both branches:

* `"range"`: `r = (layer % (n_wires - 1)) + 1`, target = `(q + r) % n_wires`
  — wrap-around cyclic CNOT with layer-varying range, matching PL SEL.
* `"linear"`: `for q in range(n-1): CNOT(q, q+1)` — open chain, no wrap.

I visually inspected all five PNGs:
* `default_75.png` (4830×929): range topology, 4 SEL layers, RX+RY final ✓
* `iqp_sel_55.png` (3780×929): range topology, **3** SEL layers, **RX-only**
  final ✓ (visibly only RX boxes, no RY column before measurement)
* `V1.png` (4830×929): identical to default_75 (same dimensions, gate count) ✓
* `V2.png` (8130×929): widest figure, 8 SEL layers, RX+RY final ✓
* `V3.png` (3330×929): smallest "range-equivalent" footprint because linear
  topology has only `n-1=4` CNOTs per layer (vs `n=5` for range);
  CNOTs visually form a `q→q+1` staircase ✓

The rendered diagrams faithfully match the textual descriptions in
`circuit_atlas.md` and the topology arguments to `QuantumGenerator`. No
post-edited gates.

## 5. Final rotation layer (RX+RY vs RX-only) — VERIFIED ✓

**Finding.** `quantum.py:239-250` has both branches:

* `default_75`: `RX(idx); RY(idx+1)` per qubit (2 params/qubit).
* `iqp_sel_55`: `RX(idx)` per qubit (1 param/qubit).

The `_final_rot_factor` helper at line 104 (`2 if circuit_id == "default_75"
else 1`) and the parameter formula at line 105-109 are consistent. The
config-lock JSONs record:

* `default_75_config_lock.json.gate_layout.final_rotation` = `"RX_plus_RY"`
* `canonical_config_lock.json.gate_layout.final_rotation` = `"RX_only"`
* `v1/v2/v3_config_lock.json.gate_layout.final_rotation` = `"RX_plus_RY"`

The PNGs visually confirm: `iqp_sel_55.png` shows a single RX column before
each measurement gate; the other four show RX→RY columns. ✓

## 6. Parameter-count formulas — VERIFIED ✓

By-hand arithmetic against each config-lock:

| circuit | formula | computed | claimed |
|---|---|---|---|
| `default_75` | 5 + 4·15 + 5·2 | 75 | 75 ✓ |
| `iqp_sel_55` | 5 + 3·15 + 5·1 | 55 | 55 ✓ |
| `V1` | 5 + 4·15 + 5·2 | 75 | 75 ✓ |
| `V2` | 5 + 8·15 + 5·2 | 135 | 135 ✓ |
| `V3` | 5 + 4·15 + 5·2 | 75 | 75 ✓ |

`run_circuit_diagrams.py:225-236` (`_expected_param_count`) mirrors the
quantum.py:105-109 formula and the run_circuit_diagrams `build_config_locks`
loop hard-asserts (explicit raise, `python -O` safe) any drift between the
operator-declared `parameter_count` in `_QUANTUM_ANSATZ` and the computed
formula. ✓

## 7. Canonical recovery (epoch 1969, SHA `f7cceb52…`) — STRUCTURALLY VERIFIED ✓

**Question posed:** was the 55-param decomposition recovered by structural
verification or by formula-matching?

**Finding.** `revision/run_recover_canonical.py` is a two-step driver. Step 1
(`--recover-only`) reads `best_checkpoint.pt`, hard-asserts
`params_pqc.shape == (55,)` (line 219-224) — this is the **ground-truth shape
gate**. Step 2 (`--assert-equivalence`) is the **structural gate**:

* lines 318-323 construct `QuantumGenerator(circuit_id="iqp_sel_55", ...)`,
* line 331-338 copies the checkpoint tensor into `gen.params_pqc`,
* lines 341-347 run `gen.last_param_index(...)` (a **structural forward-pass
  walk** that returns how many params the bound QNode tape actually consumed),
  and hard-asserts `consumed == 55`,
* lines 350-355 confirm the forward pass returns the expected output shape
  `(2 * n_qubits,)`.

This is **a genuine structural verification**, not a formula match. The
`last_param_index` method (`quantum.py:147-156`) returns the actual
`idx` cursor the QNode body increments — if the tape consumed only 50 (e.g.
if the final RX-only block had been mis-implemented as RX+RY), the assertion
would fail. RESEARCH Pitfall 2 (decomposition is non-unique by parameter count
alone, e.g. 55 = 5+3·15+5 = 5+2·15+20 = …) is correctly acknowledged in the
docstring at lines 81-83.

The corroborating evidence chain is also sound:
* stored `mu`/`sigma` (0.00245 / 0.02141) match
  `results/run_unconditioned_wgan/stats.json.moments_real` log-return moments
  (excludes Pipeline A min-max OD).
* `git log -S NUM_LAYERS` archaeology shows NUM_LAYERS ∈ {2,3,4} over time —
  corroborating, not load-bearing (D-14-02 acknowledges this).

I have no concerns with the canonical decomposition recovery.

## 8. Encoding-strategy honesty — VERIFIED ✓ (with caveat from §1 / §2)

The paper does not claim "data re-uploading". It claims "IQP encoding". The
encoding sub-block is single-shot, depth-1 IQP. The two MINOR items in §1 and
§2 fully address this. No additional concern.

## 9. Measurement basis — VERIFIED ✓

`quantum.py:256-261` returns
`(<X_0>, <Z_0>, <X_1>, <Z_1>, ..., <X_4>, <Z_4>)` — 10 expectation values
(2 per qubit, X and Z). `window_length = 2 * num_qubits = 10` is asserted
at construction (line 87-90). The measurement basis claim (PauliX + PauliZ)
is correct.

A reviewer might note that simultaneously measuring `<X_i>` and `<Z_i>` on
the same wire is non-physical on hardware (they don't commute) — on a
simulator this is fine because the statevector is reused. On a real device
each expectation would need a separate shot batch (Z-basis run + X-basis run
with H before measurement). The manuscript correctly runs on
`default.qubit` (CPU statevector simulator); there is no hardware claim, so
this is fine. If a reviewer asks "can this run on hardware as-is?" — the
honest answer is "no, the X- and Z-bases would need separated experiments,
but this is a standard simulator-style readout, not a methodological flaw."

## 10. Barren-plateau / trainability analysis for V2 (depth=8) — ABSENT (MEDIUM concern)

**Finding.** I grepped the entire `revision/` tree for `barren`,
`gradient_var`, `grad_var`, `param.shift` — zero hits outside `__pycache__`.
There is no gradient-variance analysis for V2 (which doubles the SEL depth
from 4 to 8 at 5 qubits, taking parameter count from 75 to 135).

McClean et al. 2018 / Cerezo et al. 2021 show that randomly initialized SEL
ansätze of depth `L ≈ poly(n)` exhibit exponential gradient-variance decay
with both depth and qubit count. At n=5 qubits, depth=8 is shallow enough
that this is unlikely to bite hard — but a reviewer at a top-tier venue will
ask why V2 underperforms (or fails to substantially outperform) V1 in
matched-budget training. The two candidate explanations are: (a) finite-data
overfitting / inductive-bias mismatch, or (b) optimization difficulty from a
flatter loss landscape. Without a gradient-variance probe (or even a simple
loss-curve attestation), the paper cannot distinguish them.

**Recommendation.** Add a paragraph in the discussion section
acknowledging that V2's depth was chosen as a comparison point and that no
formal barren-plateau probe was conducted — and cite McClean 2018 / Cerezo
2021 as the relevant theoretical context. A full probe is not required for
this revision, but the absence should be acknowledged rather than ignored.

## 11. Diagram fidelity — VERIFIED ✓

`run_circuit_diagrams.py:537` uses `qml.draw_mpl(model.qnode,
style="pennylane")(noise, params)` — the canonical PennyLane renderer
fed by the actual QNode tape with the actual `circuit_id`/`topology`/
`num_layers` arguments. The companion JSON sidecars
(`revision/results/figures/circuits/{name}.json`) record
`renderer = "qml.draw_mpl(style=\"pennylane\")"` and a generation
timestamp. There is no bespoke matplotlib gate drawing, no
post-render edits, and `run_circuit_diagrams._draw_one` hard-asserts
`model.num_params == lock.param_count` (lines 517-525) **before** drawing
— so a tape-shape drift from the lock would refuse to render.

PNG sizes scale plausibly with circuit depth:
default_75 (4830px, 4L) ≈ V1 (4830px, 4L) < V2 (8130px, 8L); V3 (3330px,
linear/4L) is narrower than V1 (range/4L) because linear has 4 CNOTs/layer
vs range's 5. iqp_sel_55 (3780px, 3L) is shorter than default_75 (4L).
These dimensions are consistent with the claimed structures.

The five rendered PNGs would let a reader independently reconstruct each
circuit from the diagram alone. ✓

## 12. Headline-EMD 80× discrepancy (0.0015 vs 0.0231) — EVALUATION, not CIRCUIT

**Question posed:** Is the ~80× discrepancy between original Figure_10
(EMD=0.0015) and the Phase-14 audited headline (OD-EMD=0.0231,
log_return-EMD=0.1212) a circuit problem or an evaluation problem?

**Finding.** It is an **evaluation problem**, not a circuit problem.
`revision/run_canonical_headline.py` (the Phase-14 audited driver) loads the
same `best_checkpoint.pt` (SHA `f7cceb52…`, epoch 1969) into the
**structurally-verified** 55-param `iqp_sel_55` circuit:

* line 379-386: hard-assert `sha256(best_checkpoint.pt) ==
  locked_sha` (T-14-14 — checkpoint identity gate, `python -O` safe).
* line 391-396: hard-assert `params_pqc.shape == (55,)` (D-14-02).
* line 397-402: hard-assert `checkpoint.epoch == lock.checkpoint_epoch
  == 1969`.
* line 408-414: hard-assert stored `mu`/`sigma` match the lock.
* line 442-447: structural forward-pass gate — `consumed == 55`.

These five gates collectively prove the headline is generated by the same
55-param IQP:SEL circuit with the same epoch-1969 weights as the original
Figure_10. The only thing that has changed is the **evaluation methodology**:

1. The new headline evaluates on the **OD scale** (`scale="OD"`) via
   Pipeline-B reconstruction (log-return forward + standardize + min-max
   → `[-1,1]` → inverse min-max → inverse standardize → inverse log-return
   → OD), with a deterministic `od_starts` draw seeded by
   `generation_seed * 7919 + 1`.
2. The original Figure_10's 0.0015 was almost certainly computed on the
   **standardized log-return** scale (the training-time loss space, where
   the EMD value of 0.084 is also stored in the checkpoint's own `emd`
   field — but using a different real-data reference). The Phase-14
   log-return-scale EMD (0.1212) is roughly the same order as the
   checkpoint's stored 0.084, suggesting the original 0.0015 used a
   different metric formulation (likely a normalized / Wasserstein-1 over
   batches, or comparing to *windowed* log-returns rather than the
   unwindowed series).

The frozen-checkpoint headline (`headline_canonical.json`) is therefore
load-bearing for the revision, and the manuscript's `iqp_sel_55_headline`
row should report the audited EMD (0.0231 OD, 0.1212 log_return), not the
original 0.0015. D-14-10's conflation-prevention contract
(`source="frozen_checkpoint_epoch_1969"` vs
`source="matched2000_reproduction"`) is correctly enforced.

**This is a finding the revision has already correctly addressed** — the
audited headline is what should appear in the paper. The original Figure_10
EMD should be retired from the manuscript (or explicitly footnoted as a
training-space loss-tracking metric, not a comparable EMD against held-out
real data).

---

## Summary of severities

| # | Item | Severity |
|---|---|---|
| 1 | IQP-encoding nomenclature (depth-1, single-qubit only) | MINOR — clarify wording |
| 2 | "Data re-uploading" misnomer in source docstring | MINOR — fix docstring (paper text is clean) |
| 3 | SEL block matches `qml.StronglyEntanglingLayers` | VERIFIED |
| 4 | range vs linear topology (V1/V2/default_75/iqp_sel_55 vs V3) | VERIFIED |
| 5 | Final RX+RY vs RX-only rotation | VERIFIED |
| 6 | Parameter counts (75/55/75/135/75) | VERIFIED |
| 7 | Canonical decomposition structurally verified | VERIFIED |
| 8 | Encoding strategy honesty | VERIFIED (subject to §1 wording) |
| 9 | Measurement basis (X+Z per qubit) | VERIFIED (simulator-only readout) |
| 10 | Barren-plateau probe for V2 | MEDIUM — acknowledge absence |
| 11 | Diagram fidelity (qml.draw_mpl) | VERIFIED |
| 12 | Headline-EMD discrepancy is evaluation, not circuit | RESOLVED IN REVISION |

## Final assessment

**Are the quantum-circuit claims in the manuscript scientifically defensible
at peer-review rigor? YES, subject to three corrective edits.**

The five production circuits are structurally correct, the
checkpoint-derived canonical decomposition is structurally verified (not
formula-guessed), the renderings faithfully depict the implementations, and
the parameter-count arithmetic is gated. The audited headline EMD is
defensible because it is generated from the same SHA-verified checkpoint and
the same structurally-verified circuit.

Required edits before publication:

1. **§2.a-e of methods_full.md / paper:** add a footnote that "IQP encoding"
   refers to the single-qubit, depth-1 sector of the IQP family (Havlíček et
   al. 2019), with no ZZ couplings. Pre-empts the obvious referee objection.
2. **`revision/core/models/quantum.py:1` docstring:** strike "data
   re-uploading"; replace with "single-shot IQP-style encoding". Source-only
   fix; no behavioral change.
3. **Discussion section:** add one paragraph acknowledging that no formal
   barren-plateau / gradient-variance probe was performed for V2 (depth-8,
   135-param) and cite McClean 2018 / Cerezo 2021 as the relevant theoretical
   context.

With these three edits the quantum-circuit content of the revision will
withstand expert peer review at a top-tier journal.
