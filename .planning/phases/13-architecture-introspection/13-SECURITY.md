---
phase: 13
slug: architecture-introspection
status: secured
threats_total: 14
threats_open: 0
threats_closed: 14
asvs_level: 1
block_on: high
created: 2026-05-19
---

# SECURITY.md — Phase 13: Architecture & Introspection

**Audit date:** 2026-05-19
**ASVS Level:** 1
**block_on:** high
**Disposition:** SECURED — all 14 registered threats CLOSED
**Scope:** Local-Mac scientific-compute research codebase. No network, no auth, no PII. Threats are research-integrity / reproducibility, truthfully scoped by the planner. Verification is grep/code-presence of the declared mitigation in the cited implementation files, corroborated by the green 30-test pytest suite.

---

## Threat Verification

| Threat ID | Category | Disposition | Status | Evidence |
|-----------|----------|-------------|--------|----------|
| T-13-01 | Tampering | mitigate | CLOSED | `core/models/quantum.py:184` — pre-Phase-13 range block kept as LITERAL first branch `if self.topology == "range":` (range_param = (layer % (self.num_qubits - 1)) + 1); idx accounting untouched; `tests/test_ansatz_variants.py:42,70` assert count_params==75 + fixed-seed forward `allclose(atol=1e-12)` vs hardcoded reference. Suite green. |
| T-13-02 | Tampering | mitigate | CLOSED | `core/training.py:390` call-site guard `if spectral_loss_weight > 0.0:` unchanged; `tests/test_cr01_spectral_grad.py:63-69` `test_call_site_guard_preserved` asserts the literal guard string present. |
| T-13-03 | Info disclosure / corruption | mitigate | CLOSED | `core/training.py:189-204` `_load_checkpoint`: `map_location=dev` (line 192), recast `ckpt["params_pqc"].to(device=dev, dtype=dt)` (194), opt-state-to-device loop (198-202), param_groups re-register (204); `tests/test_cr02_es_restore.py` CPU + MPS-skipif (MPS executed on this host per SUMMARY). |
| T-13-04 | Tampering | mitigate | CLOSED | `run_ansatz_comparison.py:15,80-81,108` reads `transform_ablation/runs/B/{42..46}` explicitly, V1 source string "reused 09.1/10 ... (D-13-01, no recompute)", by-construction `data_equivalence` note; no `train_wgan_gp`/`QuantumGenerator(` in aggregator; only `out_dir.mkdir` (line 333) — no V1 training dir created. Emitted `ansatz_comparison.json`: 300 rows (V1/V2/V3 = 100 each), V1 source string verified. |
| T-13-05 | Tampering | mitigate | CLOSED | `run_ansatz_sweep.sh:408` `xargs -P 2 -L 1` dispatch only; zero non-comment `multiprocessing.Pool` (only the preserved ban header lines 35-42); `run_ansatz.py` no Pool. |
| T-13-06 | Tampering | mitigate | CLOSED | `run_ansatz_sweep.sh`: `flock -x 9` (lines 204, 359) + `tempfile.mkstemp` (266, 373) + `os.rename` (272, 378) atomic status update, cloned verbatim from the proven baseline sweep. |
| T-13-07 | Tampering | mitigate | CLOSED | `run_ansatz_sweep.sh:175-182` `is_complete()` requires the full 5-file bundle (config.yaml/checkpoint.pt/samples.npy/metrics.json/inverse_kwargs.npz all `-s`); `run_ansatz.py:361` idempotent `shutil.rmtree(run_dir)` on rerun. |
| T-13-08 | Tampering | mitigate | CLOSED | `run_introspect.py:141,146` closure copies the training noise contract verbatim (`rng.uniform(NOISE_LOW, NOISE_HIGH, size=(NUM_QUBITS, BATCH_SIZE))` float32 → `.to(torch.float64) * 0.1` on CPU); `tests/test_introspect_callback.py:110` `test_snapshot_std_same_order_as_metrics_std`. |
| T-13-09 | Repudiation / Info | mitigate | CLOSED | `run_introspect.py` closure appends one record per SNAP epoch; `tests/test_introspect_callback.py:75,99` assert exact snapshot count (`len(records) == 4` on short run). 4 `_introspect_*.json` intermediates each carry 5 snapshots (SUMMARY-13-03 + on-disk). Note: callback hook is try/except-wrapped (`training.py:444`) but the count assertion is the declared mitigation and is present. |
| T-13-10 | Tampering | mitigate | CLOSED | `run_introspect.py:74` `SNAP = {0, 250, 500, 750, 999}`; line 271 terminal relabel `int(epochs) if e == max(SNAP) else e`; emitted `entanglement_trajectory.json` epochs include 1000 (vn_entropy/purity len 5). |
| T-13-11 | Tampering | mitigate | CLOSED | `run_introspect.py:278-281` builds bipartition from `QuantumGenerator.INTROSPECT_BIPARTITION = ((0,1),(2,3,4))` → `"{0,1}|{2,3,4}"`; emitted `entanglement_trajectory.json` contains exactly the verbatim string `{0,1}|{2,3,4}` in metadata (verified on disk). |
| T-13-12 | Tampering | mitigate | CLOSED | `run_introspect_figures.py`: `matplotlib.use("Agg")` (line 26), `savefig` (81); grep confirms NO `train_wgan_gp` / `QuantumGenerator(` / `.sample(` / `load_and_preprocess` — render-only. 6 figure files present and non-empty. |
| T-13-13 | Tampering | mitigate | CLOSED | `run_introspect_figures.py:68` raises `FileNotFoundError` when a companion JSON is absent; loads all three named JSON constants (lines 32-34); SUMMARY-13-04 verified loud-failure on `--figures-dir /tmp/nonexistent`. |
| T-13-SC | Tampering (supply chain) | accept | CLOSED (accepted risk, documented below) | See Accepted Risks Log. Zero experiment-dependency installs; the single `pytest` test-runner install is infrastructure mandated by the plan's own verification gate, disclosed in 13-01-SUMMARY. Note remains truthful and adequate (see assessment below). |

**Closed:** 14/14 (13 mitigate + 1 accept)

---

## Accepted Risks Log

### T-13-SC — Supply-chain (npm/pip/cargo installs)

**Disposition:** accept
**Registered claim:** "zero experiment-dep installs (deps resident in ./qgan_env, used by Phases 8-12) — N/A."

**Auditor assessment of the disclosed deviation (per prompt directive):**
13-01-SUMMARY discloses that `pytest 9.0.3` was installed into the shared `qgan_env`. This is the ONLY install in Phase 13. The accepted-risk note remains **truthful and adequate** for the following reasons:

1. **Scope is experiment dependencies, not test infrastructure.** The threat is slopsquatting / malicious experiment dependencies that could corrupt research results. `pytest` is the canonical, unambiguous Python test runner named verbatim in every PLAN file's verification gate (`./qgan_env/bin/python -m pytest`). It does not participate in any experiment data path — no `core/`, driver, or aggregator imports it.
2. **Disclosed, not hidden.** The install is explicitly recorded in 13-01-SUMMARY "Decisions Made" and "Issues Encountered", and in `tech-stack.added`. There is no concealment.
3. **No experiment dependency was added.** All scientific dependencies (torch, pennylane, numpy, scipy, matplotlib) were already resident in `qgan_env` from Phases 8-12; Phase 13 added none. The frozen `core/` reproducibility surface is untouched (T-13-01 byte-unchanged verified).
4. **Residual risk:** a single well-known PyPI package (`pytest`) added to a local research venv with no network exposure. Accepted as immaterial to research integrity.

**Decision:** Accepted risk stands. No escalation.

---

## Unregistered Flags

None. SUMMARY files for plans 01–04 contain no `## Threat Flags` section introducing unmapped attack surface. The SUMMARY-disclosed driver-local fixes (Pitfall-6 CPU pin in `run_ansatz.py` / `run_introspect.py`; snapshot device-restore) all live in in-scope driver files and explicitly leave `core/` byte-unchanged — they reinforce, not widen, the T-13-01 trust boundary.

## Auditor Notes

- 13-02-SUMMARY recorded the plan-02 sweep as INCOMPLETE (paused at a Bash-permission blocker, 2/10 runs). Post-SUMMARY state shows `results/ansatz_comparison.json` (75KB, 300 rows, V1/V2/V3 × 100) emitted with the correct provenance — the resumable sweep was driven to completion and aggregated after the SUMMARY was written. The T-13-04/05/06/07 mitigations are verified by code presence in the driver/sweep/aggregator regardless of run-state, and the emitted JSON corroborates T-13-04 (V1 reuse string, no recompute, no V1 training dir).
- `results/` is gitignored (Phase 8-12 precedent), so sweep run-dirs / `sweep_status.json` are intentionally local-only; their absence from git is expected and not a gap.
- Full pytest suite: **30 passed** (re-run by auditor).
- No implementation file was modified by this audit. Only this SECURITY.md was created.

---

*Phase 13 SECURITY.md — gsd-security-auditor*
