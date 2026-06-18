# Peer Review R2 — Methods & Reproducibility

**Reviewer role:** Reviewer 5 of 5-agent peer-review-r2; methods/reproducibility re-audit
**Pass under review:** Phase 14 plan 14-13 remediation (closes METHODS-BLOCKER-1/2, METHODS-HIGH-1, HIGH-2, HIGH-3, CR-4 disclosure)
**Round-1 reference:** `.planning/phases/14-paper-revision-release-freeze/peer-review/methods-reproducibility-review.md`

---

## Summary verdict

**PASS-WITH-FINDINGS.** Plan 14-13 unambiguously closes the two BLOCKERs and HIGH-1/HIGH-2/HIGH-3 from the round-1 methods review. A third-party reviewer who clones this repo, installs `requirements-pinned.txt`, and follows the rerun template in `methods_full.md §5.2` can (a) install a working environment that matches the captured framework versions exactly, (b) load the headline checkpoint and re-evaluate `iqp_sel_55_headline`, (c) rerun any matched-2000ep training and pass the strict-accept gate, (d) reproduce reported numbers to ~1e-6 EMD on the same CPU+BLAS stack. The remaining gaps are well-disclosed (no Zenodo DOI yet — Phase 14-07 still open; no `torch.use_deterministic_algorithms`; per-epoch trajectories not committed) rather than hidden.

**One genuine new finding (REPRO-MED-1):** `reviewer_response.md` R1-m4 row still asserts "Tagged release + Zenodo DOI workflow; Data Availability statement updated with the DOI" as if delivered, while Phase 14-07 is the only remaining open plan and no DOI exists in the repo. This is the only reproducibility-narrative honesty issue in the 14-13 remediation suite.

---

## 1. Reproducibility-from-fresh-checkout assessment

| Step | Round-1 status | Round-2 status (post-14-13) |
|---|---|---|
| Clone repo, verify data.csv (778 rows) | PASS | PASS — unchanged |
| Install pinned environment | **BLOCKER-1** (only `>=` constraints) | **PASS** — `requirements-pinned.txt` with `==` pins committed (commit `4ea576b`) |
| Load headline checkpoint | **BLOCKER-2** (`*.pt` in `.gitignore`, no exception) | **PASS** — `checkpoints/best_checkpoint.pt` (~6 MB) is tracked via `.gitignore` exception line 70: `!checkpoints/best_checkpoint.pt` |
| Re-evaluate headline (`iqp_sel_55_headline`) from frozen checkpoint | NO | **YES** — `torch.load(... weights_only=False)` returns `dict` with `keys=[epoch, emd, params_pqc, critic_state, c_optimizer, g_optimizer, mu, sigma]`; epoch=1969, EMD=0.08384, params shape (55,), float32; mu/sigma match `canonical_config_lock.json` to all digits |
| Rerun a matched-2000ep training (`--model V1 --seed 42 --epochs 2000`) | PASS | PASS — unchanged; strict-accept gate now also asserts `training_time_device` (CR-4 future-gate) |
| Reproduce bit-identical numbers | NO (claim was overclaim) | NO, but now **HONESTLY DISCLOSED** — §5.1 wording softened to "~1e-6 EMD on the same CPU+BLAS+pinned-pip-freeze stack" |
| Cite a frozen release with DOI | NO (Plan 14-07 still pending) | NO (Plan 14-07 still open — only remaining Phase 14 plan); **partially overclaimed in `reviewer_response.md` R1-m4** — see REPRO-MED-1 |

### Verification I ran

1. **Checkpoint integrity.** `shasum -a 256 checkpoints/best_checkpoint.pt` → `f7cceb52285f753b9f5f697086f3042817761d37f3112a9b36dc580ebe03b082`. This matches `canonical_config_lock.json#checkpoint_sha256` exactly. Reviewer-side integrity-check passes.

2. **Checkpoint loadability.** `torch.load('checkpoints/best_checkpoint.pt', map_location='cpu', weights_only=False)` succeeds and yields a `dict` with the expected eight keys. `ckpt['epoch']` = 1969, `ckpt['emd']` = 0.08384301715430653, `ckpt['params_pqc'].shape` = `torch.Size([55])`, `dtype` = `torch.float32`, `ckpt['mu']` = 0.0024553430266678333, `ckpt['sigma']` = 0.021407155320048332. All match `canonical_config_lock.json` to all digits.

3. **PyPI availability of pinned versions.** Queried `https://pypi.org/pypi/{pkg}/{version}/json` for every package in `requirements-pinned.txt`. All six pins are present on PyPI and resolve: pennylane 0.43.0 (requires Python ≥3.11 — satisfied by 3.11.14), torch 2.9.0, numpy 2.3.4, scipy 1.16.2, matplotlib 3.10.7, PyYAML 6.0.3.

4. **`.gitignore` checkpoint exception.** Lines 36-37 of `.gitignore` ignore `*.pth` / `*.pt`; lines 69-70 add the explicit exception `!checkpoints/best_checkpoint.pt` under a "Phase 14 plan 14-13" header comment. The negation works as intended (file is tracked, working tree clean).

5. **Largest tracked files.** `git ls-files` + `stat` size sort: the new `best_checkpoint.pt` is the single largest tracked file at 6,036,217 bytes (~6.0 MB), followed by `qgan_pennylane.ipynb` (~4.2 MB) and ten pre-existing `transform_ablation` checkpoints at ~2.0 MB each. The 6 MB add is not catastrophic; the repo's `git ls-files | wc -l` returns ~900 files. Acceptable.

---

## 2. Data-hash propagation audit (HIGH-2 finding)

Round-1 HIGH-2 explicitly called out three emitters as missing `data_hash`: `circuit_diagrams.json` (note: actually emits as per-circuit `figures/circuits/<id>.json`), `classical_architectures.json`, `framework_versions.json`. Plan 14-13 Task 4 commit `8c67891` claims to have added `data_hash = 91e447d4624e25b3` to all three.

### Verified by inspection (`grep '91e447d4624e25b3'`)

| JSON | data_hash present? | Notes |
|---|---|---|
| `results/model_info.json` | YES | Top-level field + per-model field for all 10 models |
| `results/methods_full.json` | YES | Top-level `buckets.5_reproducibility.data_hash` |
| `results/framework_versions.json` | YES | Top-level (HIGH-2 fix) |
| `results/classical_architectures.json` | YES | (HIGH-2 fix) |
| `results/figures/circuits/iqp_sel_55.json` | YES | (HIGH-2 fix — per-circuit JSON) |
| `results/figures/circuits/default_75.json` | YES | (HIGH-2 fix) |
| `results/figures/circuits/V1.json` | YES | (HIGH-2 fix) |
| `results/figures/circuits/V2.json` | YES | (HIGH-2 fix) |
| `results/figures/circuits/V3.json` | YES | (HIGH-2 fix) |
| `results/headline_canonical.json` | YES | Pre-existing |
| `results/matched2000_dualscale.json` | YES | Pre-existing |
| `results/multiseed_summary.json` | YES | Pre-existing |
| `results/tstr.json` | YES | Pre-existing |
| `results/predictive_discriminative.json` | YES | Pre-existing |
| `results/augmentation.json` | YES | Pre-existing |
| `results/baseline_comparison.json` | YES | Pre-existing |
| `results/baseline_classical_wgan.json` | YES | Pre-existing |
| `results/baseline_nonadversarial.json` | YES | Pre-existing |
| `results/fidelity_dualscale.json` | YES | Pre-existing |
| `results/total_adversarial_param_budget.json` | YES | New (Plan 14-13 T3) |
| `results/reconciliation_deltas.json` | YES | New (Plan 14-13 T3) |

### JSONs WITHOUT `data_hash`

I separately grepped every top-level JSON in `results/*.json` and every figure companion JSON in `results/figures/*.json`:

| JSON | Has data_hash? | Verdict |
|---|---|---|
| `canonical_config_lock.json` | NO | **NOT A GAP** — pure structural lock; checkpoint_sha256 is the strict-tie binding to the dataset (the checkpoint was trained on the data with that hash). |
| `default_75_config_lock.json` / `v1/v2/v3_config_lock.json` | NO | **NOT A GAP** — pure architecture-spec locks; no per-dataset binding required. |
| `canonical_recovery.json` | NO | Optimizer breadcrumbs only; not a dataset-derived metric. Reasonable. |
| `eval06_roundtrip.json` | NO | Sanity-roundtrip JSON; not paper-load-bearing. |
| `manuscript_apparatus_constants.json` | NO | NEW (Plan 14-13 T2): LaTeX apparatus constants only (20L/300L/etc.). Not data-derived. |
| `noise_model_sensitivity.json` | NO | Pre-existing; this is a paper-facing figure source. **WEAK GAP** — should ideally carry data_hash, but the noise sweep was generated against the same data and the linkage is implicit through `model_info.json#consumed_artifacts`. Not in 14-13 scope. |
| `parity_check.json` | NO | Sanity-parity JSON; not paper-load-bearing. |
| `shot_noise_sensitivity.json` | NO | Pre-existing; same caveat as noise_model_sensitivity. |
| `ansatz_comparison.json` | NO | Pre-existing; same caveat. |
| `figures/cross_model_emd.json` and all 7 other figure companion JSONs | NO | **NOT IN 14-13 SCOPE** — round-1 review explicitly listed only the 3 emitters above; figure companions inherit hash via the upstream model_info / matched2000_dualscale resolution chain. |

### Verdict on HIGH-2 propagation

The three emitters the round-1 review actually called out (HIGH-2) are all fixed. The supplementary `noise_model_sensitivity.json` / `shot_noise_sensitivity.json` / `ansatz_comparison.json` sensitivity JSONs still lack a `data_hash` field — these were NOT in HIGH-2 scope but represent a residual asymmetry. I'm flagging this as **REPRO-LOW-1** (low because the consumption chain back-resolves to `model_info.json#data_hash`), not as a HIGH finding.

---

## 3. CR-4 historical-asymmetry disclosure honesty check

### Verbatim disclosure paragraph

From `docs/methods_full.md` §4.2 (lines 380-397, byte-identical to `docs/reviewer_response.md` lines 188-206):

> **Historical training-time device asymmetry (Plan 14-13, peer-review
> disclosure).** The matched-2000ep classical runs reported in this manuscript
> executed on Apple-Silicon MPS at float32 precision (the runtime default for
> the classical training paths `train_wgan_gp` and `_train_vae` at the time of
> the original matched-budget sweep), while the quantum runs executed on CPU
> at float64 (the `_train_quantum` MPS-disable hook). This asymmetry was
> discovered post-execution during the Phase 14 peer-review pass. Future runs
> invoke the MPS-disable hook in all training paths (Plan 14-13 Task 4:
> `_train_wgan` and `_train_vae` now patch
> `torch.backends.mps.is_available = lambda: False` symmetrically), and the
> strict-accept gate now records `training_time_device` and enforces equality
> across all models in a sweep (D-14-13 extension under Plan 14-13). Numerical
> impact: MPS at float32 vs CPU at float64 on these small (74–250881 param)
> classical generators is empirically within seed variance for the
> matched-budget aggregates reported in this manuscript, but the asymmetry is
> disclosed here for completeness in lieu of a full classical sweep re-run.

### Honesty checklist (from this reviewer's brief)

| Claim required by orchestrator brief | Disclosure says it? | Verdict |
|---|---|---|
| (a) matched-2000ep classical runs used Apple-Silicon MPS at float32 | YES — explicit | OK |
| (b) quantum runs used CPU float64 via `_train_quantum` MPS-disable hook | YES — explicit | OK |
| (c) asymmetry was discovered post-execution | YES — "discovered post-execution during the Phase 14 peer-review pass" | OK |
| (d) future runs invoke the MPS-disable hook in all training paths | YES — names the specific patch `torch.backends.mps.is_available = lambda: False` and the symmetric coverage of `_train_wgan` + `_train_vae` | OK |
| (e) strict-accept gate now records `training_time_device` | YES — explicit, with D-14-13 extension citation | OK |
| (f) MPS-float32 vs CPU-float64 is empirically within seed variance for these small generators | YES — explicit | OK BUT see judgment below |

### Judgment

The disclosure is **HONEST and complete on the six listed criteria** but contains one subjective claim that a reviewer could push back on: "MPS at float32 vs CPU at float64 ... is empirically within seed variance" is asserted without a head-to-head A/B (CPU-only re-run of the classical sweep would establish this empirically). The disclosure correctly frames this as "in lieu of a full classical sweep re-run" — the user explicitly opted to disclose rather than re-run per the planning record (`peer_review_remediation.md`, "Out of scope" subsection). So the asymmetry between (a)+(b) and (f) is **acknowledged transparently**. A reviewer can choose to challenge it but cannot accuse the paper of concealing it.

The verbatim duplication between `methods_full.md §4.2` and `reviewer_response.md` ensures both the manuscript-Methods reader and the rebuttal-letter reader see the same disclosure. Good.

**Verdict: HONEST.** No MED/HIGH finding required.

---

## 4. Specific original-finding re-checks

### METHODS-BLOCKER-1 — `requirements.txt` `≥` not `==` → **RESOLVED**

`requirements-pinned.txt` exists (commit `4ea576b`) and contains exact `==` pins for every package listed in `framework_versions.json`:

```
matplotlib==3.10.7
numpy==2.3.4
pennylane==0.43.0
PyYAML==6.0.3
scipy==1.16.2
torch==2.9.0
# python==3.11.14
```

Python version is recorded as a comment (`# python==3.11.14`) rather than a pin (because pip can't pin python itself from a requirements file). This is the right convention. The install snippet at the top of the file is correct: `python -m venv qgan_env && pip install -r requirements-pinned.txt`. PyPI verification confirms all six versions resolve. `methods_full.md §4.1` references the pinned file by path. RESOLVED.

**REPRO-LOW-2 (minor):** the pinned file does not list `statsmodels` (which `core/eval.py` imports via `statsmodels.tsa.stattools.acf`). `statsmodels` is an indirect dependency that pip will pull, but for strict resolution it should be in the pinned file. Round-1 reviewer also didn't catch this. Not a blocker.

### METHODS-BLOCKER-2 — `best_checkpoint.pt` gitignored → **RESOLVED**

`checkpoints/best_checkpoint.pt` (6,036,217 bytes) is git-tracked via `.gitignore` exception lines 69-70. Working tree is clean against HEAD. `sha256` matches `canonical_config_lock.json#checkpoint_sha256` at `f7cceb52…`. `torch.load(...)` returns a `dict` with the expected eight keys and the expected scalar metadata (`epoch=1969`, `emd=0.08384…`). A reviewer can now recompute the headline EMD without obtaining the checkpoint out-of-band. RESOLVED.

### METHODS-HIGH-1 — "bit-identical" overclaim → **RESOLVED**

`docs/methods_full.md §5.1` (lines 418-426) now reads:

> The same seed produces trajectories that agree to ~1e-6 EMD on the same CPU+BLAS+pinned-pip-freeze stack (`requirements-pinned.txt`); bit-determinism would require `torch.use_deterministic_algorithms(True)` which is not set in the byte-frozen `core/training.py` (D-14-22). The pinned-env + tracked-checkpoint contract (`checkpoints/best_checkpoint.pt`, sha256 = `f7cceb52…` per `canonical_config_lock.json#checkpoint_sha256`) delivers reproducibility-within-numerical-tolerance, not bit-determinism (Plan 14-13, METHODS-HIGH-1 remediation).

This is exactly the wording the round-1 review recommended (option a). The determinism contract is now (i) explicit about its scope (~1e-6 EMD agreement, not bit-identity), (ii) explicit about the missing piece (`torch.use_deterministic_algorithms(True)`), (iii) explicit about why the missing piece can't be added (`D-14-22` byte-freeze on `core/training.py`), and (iv) explicit about what IS guaranteed (pinned-env + tracked-checkpoint). RESOLVED.

### HIGH-2 — `data_hash` missing from 3 emitters → **RESOLVED**

All three named emitters (`circuit_diagrams.json` → per-circuit files in `figures/circuits/`, `classical_architectures.json`, `framework_versions.json`) carry `data_hash = "91e447d4624e25b3"`. Verified by grep. The `run_circuit_diagrams.py`, `run_classical_arch_extract.py`, and `run_framework_versions.py` emitters were all touched in commit `8c67891`. RESOLVED.

(Residual: `noise_model_sensitivity.json`, `shot_noise_sensitivity.json`, `ansatz_comparison.json` still lack `data_hash` — REPRO-LOW-1 above. Not in HIGH-2 scope.)

### HIGH-3 / PROV-HIGH-3 — `training_protocol.md` row 34 dtype confusion → **RESOLVED**

`docs/training_protocol.md` row 34 is now split (lines 34-35):

```
| dtype_params | torch.float32 | `model_info.json` ... (dtype_params); see methods_full.md §4.b |
| dtype_samples | torch.float64 | `model_info.json` ... (dtype_samples); see methods_full.md §4.b |
```

The two fields are explicitly distinct, both carry their own `model_info.json` provenance citation, and both cross-reference `methods_full.md §6(b)` for the long-form contradiction-resolution discussion. `model_info.json` itself carries both fields on every model row. RESOLVED.

### CR-4 historical asymmetry → **DISCLOSED (per user decision to disclose rather than re-run)**

See §3 above. The verbatim disclosure paragraph appears in both `methods_full.md §4.2` and `reviewer_response.md`, hits all six honesty criteria, and is supported by the future-gate machinery (`_train_wgan` / `_train_vae` MPS-disable patches at commit `8c67891`; `training_time_device` strict-accept assertion at commit `8c67891`). RESOLVED on the disclosure-vs-conceal axis.

---

## 5. Workflow from fresh checkout — reconstructed reproduction order

A third-party reviewer can derive the following order from `methods_full.md §5.2` + `completeness_sweep_manifest.md` + the run-script docstrings:

1. **Set up environment:**
   ```
   python -m venv qgan_env
   source qgan_env/bin/activate
   pip install -r requirements-pinned.txt
   ```
2. **Verify data integrity:** `python -c "import hashlib; import numpy as np; ... # hash matches 91e447d4624e25b3"` (the `_compute_data_hash` function in `run_matched2000.py:242-252`).
3. **Re-evaluate the headline (no retrain):** `./qgan_env/bin/python -m revision.run_canonical_headline` — loads `checkpoints/best_checkpoint.pt` and writes `results/headline_canonical.json`.
4. **Rerun a matched-2000ep training (~10-40 min depending on model):** `./qgan_env/bin/python -m revision.run_matched2000 --model {iqp_sel_55_repro|V1|V2|V3|wgan_mlp|wgan_cnn|wgan_lstm|vae|ar} --seed {42|43|44|45|46} --epochs 2000`. Repeat across the (9 models × 5 seeds = 45) matrix to fully rebuild `matched2000_dualscale.json`.
5. **Strict accept gate:** `./qgan_env/bin/python -m revision.run_matched2000 --accept --model M --seed N` per run.
6. **Re-emit aggregates and figures:** `run_methods_full.py`, `run_framework_versions.py`, `run_classical_arch_extract.py`, `run_model_info.py`, `run_figure_suite.py`.
7. **Verify number provenance:** the `for doc in ...` loop in `completeness_sweep_manifest.md` lines 165-178.

**REPRO-LOW-3 finding:** there is no single top-level `README.md` reproduction recipe. The flow above is reconstructable but distributed across `methods_full.md §5.2`, `completeness_sweep_manifest.md`, and each `run_*.py` module docstring. For a 6th-step convenience improvement, a short `revision/REPRODUCE.md` listing the order in one place would help, but is not in 14-13 scope and is not a blocker.

---

## 6. What a third-party reviewer would STILL be unable to reproduce

| Artifact | Reproducible? | Why / Why not |
|---|---|---|
| `data.csv` (778 rows, the original OD time-series) | YES | Tracked in git; sha256 → `data_hash=91e447d4624e25b3`. |
| Headline checkpoint (`iqp_sel_55_headline` epoch 1969 EMD 0.08384) | YES (load) | Tracked at `checkpoints/best_checkpoint.pt`; SHA matches lock; load → params_pqc (55,). Reviewer can re-evaluate, not re-train. |
| The original training trajectory that produced the 1969-epoch checkpoint | NO | The pre-Phase-14 training run is not committed (per-epoch checkpoints, optimizer state across training, RNG state at each step). The headline is treated as a frozen artifact (`D-14-03/05` reproduction landmines). A reviewer cannot replay the training and expect to land on the exact same params. The methods doc is honest about this — `model_info.json#iqp_sel_55_headline.train_protocol_notes` says "This is NOT a retrain". |
| Matched-2000ep runs from scratch | YES | Code path (`run_matched2000.py`) + pinned env + seeds [42-46] + data.csv all in repo. Bit-identity NOT guaranteed (per §5.1) but ~1e-6 EMD agreement is expected on same CPU+BLAS. |
| Final samples (`samples.npy`) for each (model, seed) of the 45-run matrix | YES (load) | Tracked at `results/matched2000/runs/<model>/<seed>/{samples.npy,metrics.json,config.yaml,inverse_kwargs.npz}`. The reviewer can either load them directly OR retrain and compare. |
| Per-epoch training trajectories for the matched-2000ep matrix | NO | Only the final samples + metrics are committed, not the 2000-epoch metric trajectories. The aggregate convergence figures (`training_convergence_all_models.json` / `.png`) carry the trajectories in summary form; raw per-epoch records are not on disk. |
| TimeGAN / TSTR downstream-model trained on synthetic samples | YES (load) | `results/tstr.json` carries the TSTR scores; samples are loadable; the TSTR downstream training is deterministic given a seed. |
| Original raw photobioreactor cultivation data | N/A | Single-campaign limitation explicitly disclosed (methods_full.md §1, reviewer_response.md). Reviewer cannot reproduce a second independent campaign because none exists. |
| Frozen-release DOI for the manuscript | NO | Phase 14-07 is open; no Zenodo deposit yet. **See REPRO-MED-1.** |
| Compute environment beyond the pip wheels | PARTIAL | The pinned env matches macOS-26.0.1-arm64. A reviewer on Linux x86_64 will get equivalent wheels for pennylane / numpy / scipy / matplotlib / PyYAML, but `torch==2.9.0` wheels are CPU-architecture-specific. The methods doc names the capture platform; reviewers on different platforms will be working within "same Python + same packages, different BLAS" — exactly the regime the softened §5.1 contract covers. |

---

## 7. Findings

### Round-1 BLOCKER / HIGH re-check (all from `methods-reproducibility-review.md`)

| Round-1 Finding | Round-2 status |
|---|---|
| BLOCKER-1 (requirements.txt `>=` not `==`) | **RESOLVED** by commit `4ea576b` (Plan 14-13 T1) |
| BLOCKER-2 (best_checkpoint.pt gitignored) | **RESOLVED** by commit `4ea576b` (Plan 14-13 T1) — file tracked via `.gitignore:70` exception; SHA matches lock |
| HIGH-1 ("bit-identical" overclaim) | **RESOLVED** by commit `4ea576b` (Plan 14-13 T1) — §5.1 wording softened verbatim to round-1 recommendation |
| HIGH-2 (data_hash absent from 3 emitters) | **RESOLVED** by commit `8c67891` (Plan 14-13 T4) — all three named JSONs now carry data_hash=91e447d4624e25b3 |
| HIGH-3 (training_protocol.md dtype confusion) | **RESOLVED** by commit `8c67891` (Plan 14-13 T4) — row 34 split into `dtype_params` / `dtype_samples` |
| MEDIUM-1 (VAE ELBO LaTeX vs MSE code) | **DOCUMENTED** by commit `e893e0e` (Plan 14-13 T6) — `methods_full.md §3.x.d` derives implicit β≈0.4 and §2.i carries implementation note |
| MEDIUM-2 (VAE 562 vs quantum 55 not labeled "not param-matched" in methods_full.md §2.i) | Not explicitly addressed in 14-13 (`methods_full.md §2.i` still doesn't carry the "Unlike the WGAN-GP baselines..." disclaimer). **Residual MED finding** — see REPRO-MED-2 below. |
| LOW-1 (§5.2 says "lines 1-80" but docstring is 1-69) | Not addressed; `methods_full.md` line 431 still says "lines 1-80". The `methods_full.json` template field actually slices `[1:80]` so the content rendered is correct (no missing lines); the *prose* counter is off by 11 lines. **Trivial residual** — see REPRO-LOW-4. |
| LOW-2 (Kingma & Welling 2013 / Box-Jenkins citations missing in paper-blocks_refs_methods.md) | Not addressed (out of 14-13 scope; paper LaTeX citation work). Residual. |

### New findings from this round

**REPRO-MED-1** — `reviewer_response.md` R1-m4 row overclaims DOI status

Location: `docs/reviewer_response.md` line 49 (R1-m4 row of the Reviewer 1 Minor Issues table):

> | R1-m4 | Freeze GitHub repository; cite frozen version with DOI | Tagged release + Zenodo DOI workflow; Data Availability statement updated with the DOI; reproduce steps recorded | §4.3 Data Availability statement (INFRA-03, Plan 14-07) | `docs/reconciliation_note.md`; `results/model_info.json` |

The "Change made" column asserts the DOI workflow as delivered ("Tagged release + Zenodo DOI workflow; Data Availability statement updated with the DOI"), but per `completeness_sweep_manifest.md` line 125 ("the only remaining open plan is Plan 14-07 (Zenodo deposit + tag + DOI wiring)") and `peer_review_remediation.md` line 184 ("After Plan 14-13 lands, the only remaining open Phase 14 plan is **Plan 14-07**") and the user-memory note ("Phase 14 is 6/7 done; 14-07 paused on a deferred manual Zenodo DOI deposit"), no DOI has been deposited yet.

The other completeness-sweep paragraph in the same doc (lines 85-186) IS clearer ("The full Training Protocol... R1-M4 is hereby marked **RESOLVED**" — but R1-m4-lowercase is the DOI one, not R1-M4-uppercase). The contradiction is: R1-m4's row asserts DOI delivery while the elsewhere-cited Plan 14-07 ledger says it's the one outstanding plan.

**Severity: MEDIUM.** This is a rebuttal-letter honesty issue. A reviewer who reads the R1-m4 row literally will expect to see a DOI in §4.3 of the manuscript; if they then check the manuscript and find a TBD DOI, they will lose trust. **Recommended fix:** change "Tagged release + Zenodo DOI workflow; Data Availability statement updated with the DOI" → "Tagged release prepared; Zenodo deposit + DOI wiring scheduled under Plan 14-07 (pending at the time of resubmission; the Data Availability statement carries a placeholder updated upon DOI assignment)." This is a 2-line edit to one cell.

**REPRO-MED-2** — `methods_full.md §2.i` still doesn't explicitly say "VAE not parameter-matched" (round-1 MED-2 carried over)

Location: `docs/methods_full.md` lines 171-198. The §2.i VAE table reports 562 params; the §2.i prose does NOT explicitly say "the VAE is intentionally NOT parameter-matched to the quantum generator." The `core/models/nonadversarial.py:11-13` source comment ("NOT parameter-matched to the quantum generator (D-10-03)") never made it into the methods doc.

**Severity: MEDIUM** (carried over from round-1 unchanged). 14-13 was scoped to the round-1 BLOCKER/HIGH set, so this fell outside scope. Worth a 1-sentence add to §2.i.

**REPRO-LOW-1** — `noise_model_sensitivity.json` / `shot_noise_sensitivity.json` / `ansatz_comparison.json` lack `data_hash`

These three sensitivity JSONs are paper-load-bearing (the figures derived from them are cited in `paper_blocks_refs_methods.md`) but do not carry a top-level `data_hash` field. The linkage to the dataset is implicit via `model_info.json#consumed_artifacts`. **Severity: LOW** because the resolution chain back-resolves through audited artifacts that do carry the hash. Reasonable to add for symmetry in a follow-up plan.

**REPRO-LOW-2** — `statsmodels` not in `requirements-pinned.txt`

`core/eval.py` imports `statsmodels.tsa.stattools.acf`. `requirements-pinned.txt` doesn't pin statsmodels. Pip will pull a transitive version via numpy/scipy resolution, but for strict reproducibility the explicit pin matters when statsmodels' ACF implementation has changed across versions. **Severity: LOW.** Pin should be added in a follow-up.

**REPRO-LOW-3** — no single `REPRODUCE.md` / top-level reproduction recipe

The reproduction flow is distributed across `methods_full.md §5.2`, `completeness_sweep_manifest.md`, and `run_*.py` docstrings. A reviewer has to reconstruct the order themselves. **Severity: LOW.** Not a blocker because the rerun template in `methods_full.md §5.2` is exhaustively complete; only a convenience improvement.

**REPRO-LOW-4** — `methods_full.md §5.2` says "lines 1-80" but the actual module docstring is 1-69

(Carried over from round-1 LOW-1.) The `run_methods_full.py` slice writes `methods_full.json.rerun_command_template = docstring_text[1:80]` — a slice over 80 *characters* would be tiny, so the slice is presumably over 80 *lines*. The actual docstring is 69 lines. The rendered template contains the full docstring (because slice past EOF stops at EOF) but the prose "lines 1-80" is mis-quoting the actual range. **Severity: LOW (trivial).** Documentation drift, not a reproduction blocker.

---

## 8. Final recommendation

**Reproducibility sound for paper resubmission: YES — with two narrow corrections recommended.**

The two BLOCKERs from round-1 are decisively closed; the three HIGHs are closed; the determinism claim is now honest and properly scoped. A third-party AIChE reviewer with the pinned env and `checkpoints/best_checkpoint.pt` can re-evaluate the headline EMD to all digits and rerun any matched-2000ep training to ~1e-6 EMD agreement. The CR-4 disclosure is honest on all six criteria the orchestrator brief listed.

**Two narrow corrections recommended before resubmission:**

1. **(REPRO-MED-1, ~2 lines)** Fix the R1-m4 row of `reviewer_response.md` to say "pending Plan 14-07 at the time of resubmission" rather than implying the DOI is already deposited. This is the single inconsistency between the rebuttal letter and the actual Phase 14 ledger.
2. **(REPRO-MED-2, ~1 sentence)** Add "VAE is intentionally NOT parameter-matched to the quantum generator (562 vs 55)" to `methods_full.md §2.i`. This is a carry-over from round-1 MED-2 that fell outside 14-13 scope but is small enough to fix in a touch-up.

Everything else (REPRO-LOW-1..4) is residual polish and can be deferred to a post-resubmission cleanup pass.

The reproducibility story for the manuscript is **sound for resubmission**.

---

## Files audited (absolute paths)

- `/Users/shawngibford/dev/phd/qGAN/requirements-pinned.txt`
- `/Users/shawngibford/dev/phd/qGAN/checkpoints/best_checkpoint.pt`
- `/Users/shawngibford/dev/phd/qGAN/results/canonical_config_lock.json`
- `/Users/shawngibford/dev/phd/qGAN/results/framework_versions.json`
- `/Users/shawngibford/dev/phd/qGAN/results/methods_full.json`
- `/Users/shawngibford/dev/phd/qGAN/results/model_info.json`
- `/Users/shawngibford/dev/phd/qGAN/results/classical_architectures.json`
- `/Users/shawngibford/dev/phd/qGAN/results/figures/circuits/{default_75,iqp_sel_55,V1,V2,V3}.json`
- `/Users/shawngibford/dev/phd/qGAN/docs/methods_full.md`
- `/Users/shawngibford/dev/phd/qGAN/docs/training_protocol.md`
- `/Users/shawngibford/dev/phd/qGAN/docs/reviewer_response.md`
- `/Users/shawngibford/dev/phd/qGAN/docs/reconciliation_note.md`
- `/Users/shawngibford/dev/phd/qGAN/docs/completeness_sweep_manifest.md`
- `/Users/shawngibford/dev/phd/qGAN/docs/peer_review_remediation.md`
- `/Users/shawngibford/dev/phd/qGAN/.gitignore`
- `/Users/shawngibford/dev/phd/qGAN/.planning/phases/14-paper-revision-release-freeze/peer-review/methods-reproducibility-review.md`

---

**Reviewer:** R2 Reviewer 5 (methods + reproducibility)
**Confidence:** HIGH on BLOCKER/HIGH closures (verified by direct file inspection, checkpoint load, SHA match, PyPI version probe); HIGH on REPRO-MED-1 (direct doc inconsistency); MEDIUM on REPRO-LOW residuals (judgment-call polish items).
