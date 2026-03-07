---
phase: 02
slug: wgan-gp-correctness-and-quantum-circuit-redesign
status: validated
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | None — Jupyter notebook research project |
| **Config file** | none |
| **Quick run command** | N/A (inline `python3 -c` assertions run during plan execution) |
| **Full suite command** | N/A |
| **Estimated runtime** | N/A |

---

## Sampling Rate

- **After every task commit:** Inline `python3 -c` structural assertions on notebook JSON
- **After every plan wave:** VERIFICATION.md static analysis (19/19 truths confirmed)
- **Before `/gsd:verify-work`:** VERIFICATION.md review
- **Max feedback latency:** N/A (no persistent test suite)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | QC-05, WGAN-01, WGAN-02, WGAN-07 | structural | `python3 -c "..."` (inline, plan 02-01 T1) | ❌ manual-only | ✅ passed at exec |
| 02-01-02 | 01 | 1 | QUAL-06 | structural | `python3 -c "..."` (inline, plan 02-01 T2) | ❌ manual-only | ✅ passed at exec |
| 02-02-01 | 02 | 2 | QC-01, QC-02, QC-03, QC-04, PERF-01 | structural | `python3 -c "..."` (inline, plan 02-02 T1) | ❌ manual-only | ✅ passed at exec |
| 02-02-02 | 02 | 2 | WGAN-03 | structural | `python3 -c "..."` (inline, plan 02-02 T2) | ❌ manual-only | ✅ passed at exec |
| 02-03-01 | 03 | 3 | BUG-02, PERF-05, QC-04 | structural | `python3 -c "..."` (inline, plan 02-03 T1) | ❌ manual-only | ✅ passed at exec |
| 02-03-02 | 03 | 3 | BUG-03, PERF-04, WGAN-04, WGAN-05, WGAN-08 | structural | `python3 -c "..."` (inline, plan 02-03 T2) | ❌ manual-only | ✅ passed at exec |
| 02-04-01 | 04 | 4 | WGAN-06 | structural | `python3 -c "..."` (inline, plan 02-04 T1) | ❌ manual-only | ✅ passed at exec |
| 02-04-02 | 04 | 4 | BUG-02, BUG-03 | structural | `python3 -c "..."` (inline, plan 02-04 T2) | ❌ manual-only | ✅ passed at exec |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements via inline verification commands and VERIFICATION.md (19/19 truths, all requirements satisfied).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Config cell has WGAN-GP paper values (N_CRITIC=5, LAMBDA=10, swapped LRs) | WGAN-01, WGAN-02, WGAN-07 | Jupyter notebook; no test framework | Inspect Cell 28 for N_CRITIC=5, LAMBDA=10, LR_CRITIC=8e-5, LR_GENERATOR=3e-5 |
| WINDOW_LENGTH = 2 * NUM_QUBITS computed dynamically | QC-05 | Jupyter notebook; no test framework | Inspect Cell 28 for `WINDOW_LENGTH = 2 * NUM_QUBITS` |
| normalize() returns 3-tuple, call site unpacks | QUAL-06 | Jupyter notebook; no test framework | Inspect Cell 14 return and Cell 15 unpacking |
| Data re-uploading circuit with RX encoding at every layer | QC-01, QC-02, QC-03, QC-04 | Jupyter notebook; no test framework | Inspect Cell 26 define_generator_circuit for encoding_layer calls inside layer loop, RX gates, PauliX+PauliZ measurements |
| diff_method='backprop' on default.qubit | PERF-01 | Jupyter notebook; no test framework | Inspect Cell 26 define_generator_model for diff_method='backprop' |
| Critic has no Dropout, uses LeakyReLU(0.2) | WGAN-03 | Jupyter notebook; no test framework | Inspect Cell 26 define_critic_model — no Dropout, negative_slope=0.2 |
| Broadcasting noise shape (num_qubits, batch_size) for single QNode call | PERF-05 | Jupyter notebook; no test framework | Inspect Cell 26 _train_one_epoch for noise shape and no per-sample loops |
| GEN_SCALE applied in critic training, generator training, and evaluation | BUG-02 | Jupyter notebook; no test framework | Search Cell 26 for `* GEN_SCALE` — should appear 3 times |
| One-sided GP with per-sample alpha | WGAN-04 | Jupyter notebook; no test framework | Inspect Cell 26 for `torch.clamp(grad_norms - 1, min=0)` and alpha using actual_batch_size |
| EMD on raw 1D arrays via wasserstein_distance | WGAN-04 | Jupyter notebook; no test framework | Inspect Cell 26 for `wasserstein_distance(all_real_flat, fake_flat)` |
| Dynamic histogram bins from data range | WGAN-05 | Jupyter notebook; no test framework | Inspect Cell 26 for `np.linspace(all_real_flat.min(), all_real_flat.max(), ...)` |
| Evaluation fires every EVAL_EVERY epochs | PERF-04 | Requires runtime execution | Run training with NUM_EPOCHS=30, EVAL_EVERY=10; confirm eval at epochs 10, 20, 30 only |
| Stylized facts include kurtosis at stitched and window level | WGAN-08 | Jupyter notebook; no test framework | Inspect Cell 26 stylized_facts method for kurtosis computation |
| Early stopping monitors EMD with patience=50, warmup=100 | WGAN-06 | Jupyter notebook; no test framework | Inspect Cell 30 EarlyStopping class for best_emd, patience, warmup_epochs |
| Checkpoint saves mu, sigma alongside model weights | WGAN-06 | Jupyter notebook; no test framework | Inspect Cell 30 _save_checkpoint for 'mu' and 'sigma' keys |
| Standalone generation uses identical GEN_SCALE and full_denorm_pipeline | BUG-02, BUG-03 | Jupyter notebook; no test framework | Inspect Cell 40 for GEN_SCALE, full_denorm_pipeline, broadcasting, 4*pi noise |
| full_denorm_pipeline denormalization function exists | BUG-03 | Jupyter notebook; no test framework | Inspect Cell 23 for full_denorm_pipeline with rescale, lambert_w_transform, denormalize |
| Generator gradient flow via backprop QNode | PERF-01 | Requires runtime execution | Run 1 training step; confirm "Generator grad" is non-zero in eval output |
| Standalone generation numerically equivalent to training eval | BUG-02, BUG-03 | Requires runtime execution | Run both with same noise; compare outputs to floating-point precision |

---

## Validation Audit 2026-03-07

| Metric | Count |
|--------|-------|
| Requirements | 19 |
| Gaps found | 19 (no persistent test files) |
| Resolved | 0 |
| Escalated to manual-only | 19 |

**Rationale:** This is a Jupyter notebook research project with no test framework. All 19 requirements were verified at execution time via inline `python3 -c` structural assertions and confirmed by VERIFICATION.md (19/19 truths passed, 0 gaps). The inline verifications are not persistent test files but were run and passed during plan execution. Three additional human verification items require notebook execution.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < N/A (notebook project)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** manual-only 2026-03-07
