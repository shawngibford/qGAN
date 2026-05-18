# Phase 11: Utility Evaluation - Research

**Researched:** 2026-05-17
**Domain:** Evaluation-only ML infrastructure — TSTR soft-sensor, TimeGAN predictive/discriminative scores, Orlandi-style augmentation, dual-scale fidelity reporting on frozen GAN sample artifacts
**Confidence:** HIGH (codebase contracts verified on disk; TimeGAN definitions pinned to the canonical Yoon et al. NeurIPS 2019 repo source)

## Summary

Phase 11 is a **pure consumer phase**: it reads 60 frozen `samples.npy` artifacts (50 Phase 10 baseline runs + 10 Phase 09.1 quantum runs across Pipelines A/B × seeds 42–46) and produces three utility verdicts plus a dual-scale fidelity re-emission. No model is trained or regenerated. Every reusable contract the planner needs has been verified on disk: sample shape `(3840, 10)` float64 in `[-1,1]` window space, the `reconstruct_od` OD-reconstruction recipe, the `HELD_OUT_N=320` real-split convention, the long-form JSON schema, and the data-hash invariant.

The single highest-value finding: **the canonical TimeGAN predictive/discriminative score definitions are now pinned to exact source** from `jsyoon0823/TimeGAN` (the original NeurIPS 2019 repo, the implementation the reviewers will recognize). Both post-hoc nets are **single-layer GRU, `hidden_dim = int(input_dim/2)`, Adam at TF default LR (1e-3), batch 128**; predictive uses **5000 iterations**, discriminative **2000 iterations** with an **80/20 train/test split** (`train_test_divide`, `train_rate=0.8`). The predictive score is next-step-feature MAE (train-on-synthetic / test-on-real); the discriminative score is `|0.5 − test_accuracy|`. These numbers must be adapted to this project's univariate length-10 windows (see Pitfall 1).

The second critical finding: **sklearn is NOT installed in the project venv** (`qgan_env`). Phase 10 deliberately used an inline `r2_score_inline`. Phase 11 must reuse that inline R²/MAE/RMSE math (or guard a sklearn install behind a checkpoint) — do not assume `from sklearn.metrics import ...` will work.

**Primary recommendation:** Build two new drivers — `revision/run_utility.py` (TSTR + augmentation, both on the shared one-step-ahead OD forecast task) and `revision/run_timegan_scores.py` (faithful single-layer-GRU predictive + discriminative). Reuse `reconstruct_od` / `train_eval_tstr` patterns verbatim from `revision/_build_baseline_notebook.py` (they are NOT in `revision/core/` — D-10-13/D-11-10 keep eval orchestration out of core). Emit all three JSONs in the existing long-form `{model_kind, pipeline, seed, metric_name, scale, value}` schema, and wrap every `revision.core.eval` fidelity call with a `scale: "log_return" | "OD"` tag for EVAL-05.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-11-01:** Soft-sensor task is **one-step-ahead OD forecasting**: given `OD[t-k..t]`, predict `OD[t+1]`. No PAR_LIGHT conditioning.
- **D-11-02:** TSTR protocol — train soft-sensor on **synthetic** windows, evaluate on **held-out real** windows. Report R², MAE, RMSE. Reuse Phase 10 TSTR-lite held-out real split (D-10-21, 320 held-out real windows).
- **D-11-03:** Faithful TimeGAN post-hoc nets (not Phase 10 TSTR-lite scaffolding). Predictive score = MAE of post-hoc sequence predictor trained on synthetic, tested on real, next-step. Discriminative score = `|0.5 − test_accuracy|` of post-hoc real-vs-synthetic classifier. Lower is better for both.
- **D-11-04:** Post-hoc nets follow canonical TimeGAN architecture (GRU-based; hidden dim ≈ input_dim, ~1–2 recurrent layers). Exact hyperparameters are a research item (pinned in this document).
- **D-11-05:** Scores reported as **mean ± std across 5-seed set {42,43,44,45,46}**, reusing existing per-seed sample artifacts.
- **D-11-06:** Augmentation = **mixing-ratio sweep**, not single condition. Downstream task = **same one-step-ahead OD soft-sensor** as D-11-01.
- **D-11-07:** Conditions: `real-only` baseline, then `real + synthetic` at multiple injection ratios → lift curve per generator (suggested grid `{+25%, +50%, +100%, synthetic-only}`; exact grid is planner discretion). Delta table = downstream R²/MAE/RMSE change vs. real-only baseline, per generator.
- **D-11-08:** **Reuse Phase 10 / Phase 09.1 artifacts as-is.** Read existing `samples.npy` from 50 Phase 10 baseline run dirs + Phase 09.1 quantum runs. **No regeneration, no retraining.**
- **D-11-09:** Both **Pipeline A and Pipeline B** evaluated. Pipeline B is headline (D-10-06); Pipeline A is supplementary raw-OD control. EVAL-05 dual-scale (`log_return` + `OD`) applies to every metric.
- **D-11-10:** Evaluation/aggregation logic stays **out of `revision/core/`**. New TSTR/score/augmentation orchestration lives in new `revision/run_*.py` driver(s) + JSON emitters, patterned after `revision/run_baselines.py`. Reuse `revision/core/eval.py` fidelity helpers unchanged for EVAL-05.

### Claude's Discretion

- Exact post-hoc GRU hyperparameters (depth, hidden dim, epochs) — pin to cited TimeGAN reference implementation during research. **(Pinned in this document — see "TimeGAN Score Definitions".)**
- Even though user selected "faithful" (not "faithful + cite"), **pin and record the reference implementation (ydata-synthetic / original TimeGAN repo) in JSON metadata** — zero-cost, strictly more defensible.
- Soft-sensor architecture (1D-CNN vs LSTM per EVAL-01's "or") — planner selects; a single architecture used consistently across all generators preferred over comparing two.
- Augmentation mixing-ratio grid resolution.
- TSTR/score sample sizes drawn from existing artifacts (subsampling strategy if needed).

### Deferred Ideas (OUT OF SCOPE)

- **PAR_LIGHT-conditioned soft-sensor** — deferred in favor of the cleaner one-step-ahead task.
- **Small-real-regime augmentation** — deliberately shrinking the real training set. Not selected (full real set used); backlog robustness check only.
- **CR-01 (spectral-loss hook) / CR-02 (EarlyStopping checkpoint restore)** — training-loop bugs, owned by Phase 13. Out of scope here.
- Shot-noise / noise-model / multi-seed roll-up → Phase 12. Ansatz / introspection → Phase 13. Manuscript integration → Phase 14.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVAL-01 | TSTR pipeline — train 1D-CNN or LSTM soft-sensor on synthetic OD windows, evaluate on held-out real data; report R², MAE, RMSE | `reconstruct_od` + `train_eval_tstr` patterns verified; `HELD_OUT_N=320` split confirmed; sample shapes `(3840,10)` confirmed across all 6 model kinds × 2 pipelines × 5 seeds; inline `r2_score_inline` available (sklearn absent) |
| EVAL-02 | TimeGAN-style predictive score for quantum + classical WGAN-GP + non-adversarial | Canonical Yoon et al. predictive_metrics.py pinned: single-layer GRU, hidden=int(dim/2), 5000 iters, Adam(default LR), batch 128, MAE next-step, 80/20 split |
| EVAL-03 | TimeGAN-style discriminative score for same three models | Canonical Yoon et al. discriminative_metrics.py pinned: single-layer GRU, hidden=int(dim/2), 2000 iters, Adam(default LR), batch 128, `|0.5−acc|`, 80/20 split, real=1 / synth=0 |
| EVAL-04 | Real-only vs. synthetic-augmented training comparison (Orlandi et al. [26] style) | Shares the D-11-01 one-step-ahead OD task; real-only baseline already computed in Phase 10 (`tstr["real_only_baseline"]`, R²=-13.35); mixing-ratio design documented below |
| EVAL-05 | All fidelity metrics (EMD, ACF, moments, DTW) on both log-return and OD scales — explicit `scale` field | `revision.core.eval` helpers verified pure; `reconstruct_od` already returns both `od_samples` (OD) and `transformed` (log-return, Pipeline B); long-form schema already carries `scale` field |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Read frozen samples.npy | Artifact I/O (driver) | — | `revision/run_*.py` drivers; never `revision/core/` (D-11-10) |
| OD reconstruction (inverse transform) | `revision/core/preprocessing.py` (`inverse_logreturns`, `inverse_minmax_od`) | driver glue (`reconstruct_od` wrapper) | Inverse math is core; the per-pipeline orchestration wrapper is driver-level |
| Fidelity metrics (EMD/ACF/moments/DTW) | `revision/core/eval.py` (unchanged) | driver (scale-tagging loop) | Metric math is core; the `scale` field wrap is driver-level (EVAL-05) |
| TSTR soft-sensor train/eval | driver (`run_utility.py`) | — | Post-hoc evaluation model; not a project model family — stays out of core |
| TimeGAN post-hoc GRU nets | driver (`run_timegan_scores.py`) | — | Evaluation-only nets; per D-10-13/D-11-10 not promoted to core |
| Augmentation mixing-ratio sweep | driver (`run_utility.py`) | — | Orchestration over the same TSTR task |
| JSON emission (long-form schema) | driver | — | `revision/results/*.json` contract Phase 14 reads |
| Data-hash consistency assert | driver | `revision/core/data.py::load_and_preprocess` | Hash recomputed from canonical OD tensor |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | (project venv `qgan_env`) | Post-hoc GRU nets, TSTR LSTM/CNN, all training | Already the project's only DL framework; `train_eval_tstr` uses it |
| numpy | (project venv) | Sample I/O, R²/MAE/RMSE math, subsampling RNG | Already pervasive; `np.load`, `np.random.default_rng` |
| `revision.core.eval` | in-tree | EMD/moments/ACF/DTW/JSD/PSD | D-11-10 reuse-unchanged mandate |
| `revision.core.preprocessing` | in-tree | `inverse_logreturns`, `inverse_minmax_od` | OD-scale reconstruction (ABL-01 verified ≤1e-8 round-trip) |
| `revision.core.data` | in-tree | `load_and_preprocess`, `rolling_window` | Real OD windows + data-hash source |
| pyyaml | (project venv) | Read run `config.yaml`, write driver config | Already used by `run_baselines.py` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| statsmodels | installed (verified) | ACF (`compute_acf` uses `statsmodels.tsa.stattools.acf`) | Transitively via `revision.core.eval` — no direct import needed |
| fastdtw | installed (verified) | DTW (`compute_dtw`) | Transitively via `revision.core.eval` |
| scipy | installed (verified) | wasserstein_distance, kurtosis, skew, jensenshannon | Transitively via `revision.core.eval` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Inline `r2_score_inline` / hand MAE+RMSE | `sklearn.metrics` | **sklearn is NOT installed in `qgan_env`** (verified: `ModuleNotFoundError: No module named 'sklearn'`). Adding it is a new dependency requiring a guarded install + checkpoint. Phase 10 deliberately used inline math — reuse that. |
| Original TF1 `jsyoon0823/TimeGAN` metric code | Re-implement the exact algorithm in PyTorch | The reference is TensorFlow-1 (`tf.nn.rnn_cell.GRUCell`, `tf.train.AdamOptimizer`). The project is torch-only and statevector-CPU. Re-implement the **algorithm faithfully in torch**, citing the reference in JSON metadata (the user's discretion note explicitly endorses this). |
| `ydata-synthetic` package | Original Yoon repo definitions | `ydata-synthetic`'s TimeGAN wraps but slightly differs from the original (3-layer GRU embedding; post-hoc nets documented as 2-layer LSTM hidden=4×dim in some write-ups). The **original Yoon repo metric files are the canonical definition** and the strongest rebuttal anchor. Cite `jsyoon0823/TimeGAN` commit. Do NOT pip-install `ydata-synthetic` (heavy TF dependency tree; unnecessary). |

**Installation:** No new packages required. Phase 11 runs entirely on the existing `qgan_env` venv (torch + numpy + scipy + statsmodels + fastdtw + pyyaml, all verified present). **Do not `pip install scikit-learn` or `ydata-synthetic` without an explicit `checkpoint:human-verify` task** — neither is needed if inline metric math is reused.

**Version verification:** `pennylane 0.43.0` confirmed (not needed for Phase 11 — no quantum execution, samples are frozen). sklearn confirmed ABSENT. statsmodels + fastdtw + scipy confirmed present via import test.

## Package Legitimacy Audit

> Phase 11 installs **zero external packages**. All dependencies are already in the project venv or in-tree.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| (none — no installs) | — | — | — | — | n/a | No external installs in this phase |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

The only package a naive implementation might reach for is `scikit-learn`. It is a legitimate, well-known package, but it is **not installed in `qgan_env` and must not be added** — the established project pattern (Phase 10 `r2_score_inline`) avoids it deliberately. If the planner judges a sklearn install necessary, gate it behind a `checkpoint:human-verify` task (it would be a new runtime dependency that also touches the `v2.0-revision` freeze surface).

## Architecture Patterns

### System Architecture Diagram

```
                          data.csv  (sha256[:16] = 91e447d4624e25b3)
                              │
              load_and_preprocess() ──► real OD tensor ──► rolling_window(10, stride 2)
                              │                                   │
                              │                          real_windowed_OD  (M=384, 10)
                              ▼                                   │
   ┌─── FROZEN ARTIFACTS (read-only, D-11-08) ───┐                │
   │  quantum:  transform_ablation/runs/{A,B}/{42..46}/samples.npy │
   │  baselines: baselines/runs/{wgan_mlp,wgan_cnn,wgan_lstm,vae,ar}/{A,B}/{42..46}/samples.npy
   │            + inverse_kwargs.npz + config.yaml (data_hash)     │
   └──────────────────────────┬───────────────────────────────────┘
                              │  reconstruct_od(model_kind, pipeline, seed)
                              ▼  (samples [-1,1] → OD scale; also returns log-return for Pipeline B)
              ┌───────────────┼───────────────────────────┐
              ▼               ▼                            ▼
   ┌──────────────────┐  ┌─────────────────────┐  ┌──────────────────────────┐
   │ EVAL-01 TSTR     │  │ EVAL-02/03 TimeGAN  │  │ EVAL-05 dual-scale       │
   │ soft-sensor      │  │ post-hoc GRU nets   │  │ fidelity re-emit         │
   │ (one-step OD)    │  │ predictive (MAE)    │  │ for each eval.py metric: │
   │ train=synth      │  │ discriminative      │  │  scale="OD"  +           │
   │ eval=real[:320]  │  │ (|0.5-acc|)         │  │  scale="log_return"      │
   └───────┬──────────┘  └─────────┬───────────┘  └────────────┬─────────────┘
           │  (shares task)        │                           │
           ▼                       ▼                           ▼
   ┌──────────────────┐    predictive_discriminative   rows[] appended to
   │ EVAL-04 augment  │    .json (mean±std/5 seeds)    tstr.json / augmentation.json
   │ real-only vs     │
   │ real+synth @     │           all JSONs: long-form {model_kind,pipeline,seed,
   │ {+25,+50,+100,   │           metric_name,scale,value} + data_hash assertion
   │  synth-only}     │
   └───────┬──────────┘
           ▼
   tstr.json   augmentation.json   predictive_discriminative.json
   (Phase 14 PAPER reads these)
```

### Recommended Project Structure

```
revision/
├── run_utility.py              # NEW — EVAL-01 TSTR + EVAL-04 augmentation (shared one-step-ahead OD task)
├── run_timegan_scores.py       # NEW — EVAL-02/03 faithful single-layer-GRU predictive + discriminative
├── run_dualscale_fidelity.py   # NEW (or fold into a Wave-4 emit notebook) — EVAL-05 scale-tagged re-emit
├── _build_utility_notebook.py  # OPTIONAL — deterministic notebook source (mirrors _build_baseline_notebook.py)
├── 07_utility_eval.ipynb       # OPTIONAL — generated analysis notebook (orchestrate+plot+JSON only)
├── core/                       # UNCHANGED — D-11-10 invariant (git diff revision/core/ must be empty)
└── results/
    ├── tstr.json                       # EVAL-01
    ├── predictive_discriminative.json  # EVAL-02/03
    ├── augmentation.json               # EVAL-04
    └── (EVAL-05 scale rows fold into the above + a fidelity_dualscale.json if separated)
```

### Pattern 1: Driver-mirrors-run_baselines.py

**What:** Each new driver is a CLI entrypoint that reads frozen artifacts, computes, and writes JSON. Pattern lifted from `revision/run_baselines.py`.
**When to use:** All three Phase 11 deliverables.
**Example:**
```python
# Source: revision/run_baselines.py (verified in-tree) — the canonical driver shape
# argparse → resolve run dir → load samples.npy + inverse_kwargs.npz →
# reconstruct → compute → write JSON to revision/results/. One concern per driver.
# Idempotent; no multiprocessing.Pool (Phase 09.1 Pitfall 4 — xargs -P only).
```

### Pattern 2: reconstruct_od (the OD-reconstruction contract — copy verbatim)

**What:** Given `(model_kind, pipeline, seed)`, resolves the run dir, loads `samples.npy` (`[-1,1]`, shape `(3840,10)`, float64) + `inverse_kwargs.npz`, and reconstructs OD-scale windows. For Pipeline B it also returns the log-return array (needed for EVAL-05 `scale="log_return"`).
**When to use:** Every EVAL-01/04/05 OD computation.
**Example:**
```python
# Source: revision/_build_baseline_notebook.py:167-210 (verified, copy VERBATIM — D-11-10)
def _run_base(model_kind, pipeline, seed):
    if model_kind == "quantum":                      # reused 09.1 quantum runs (D-10-18)
        return Path(f"revision/results/transform_ablation/runs/{pipeline}/{seed}")
    return Path(f"revision/results/baselines/runs/{model_kind}/{pipeline}/{seed}")

def reconstruct_od(model_kind, pipeline, seed, n_synth_subsample=None):
    base = _run_base(model_kind, pipeline, seed)
    samples_pm1 = np.load(base / "samples.npy").astype(np.float64)   # (3840, 10)
    inv = np.load(base / "inverse_kwargs.npz", allow_pickle=True)
    if pipeline == "A":
        od_min, od_max = float(inv["od_min"]), float(inv["od_max"])
        od = ((samples_pm1 + 1.0) / 2.0) * (od_max - od_min) + od_min
        return {"od_samples": od, "transformed": None, ...}
    if pipeline == "B":
        r_min, r_max = float(inv["r_min"]), float(inv["r_max"])
        mu, sigma = float(inv["mu"]), float(inv["sigma"])
        od_starts_pool = np.asarray(inv["od_starts"])               # (384,)
        r_norm = ((samples_pm1 + 1.0)/2.0)*(r_max - r_min) + r_min
        rng = np.random.default_rng(seed * 7919 + 1)
        od_start = rng.choice(od_starts_pool, size=r_norm.shape[0], replace=True)
        od_full = inverse_logreturns(torch.tensor(r_norm), torch.tensor(od_start),
                                     torch.tensor(mu), torch.tensor(sigma)).cpu().numpy()
        if od_full.shape[1] == 11: od_full = od_full[:, :10]
        return {"od_samples": od_full, "transformed": r_norm, ...}
```

### Pattern 3: TSTR one-step-ahead split (the D-11-01/D-11-02 task)

**What:** Window `(N,10)` → `X = window[:, :9]`, `y = window[:, 9:10]`. Train on synthetic OD, evaluate on `real_windowed_OD[:320]`. Real-only baseline trains on `real_windowed_OD[320:]` (= 64 windows; Phase 10 reported `n_train_real=65` — confirm exact count on the canonical 384-window split).
**When to use:** EVAL-01 TSTR and EVAL-04 augmentation (same task — D-11-06).
**Example:**
```python
# Source: revision/_build_baseline_notebook.py:394-440 (verified) — the exact split
HELD_OUT_N = 320
real_eval  = real_windowed_OD[:HELD_OUT_N]          # eval set, identical to Phase 10 D-10-21
real_train = real_windowed_OD[HELD_OUT_N:]          # real-only baseline train set
Xtr = train_windows[:, :9];  ytr = train_windows[:, 9:10]   # one-step-ahead
# r2_score_inline = 1 - ss_res/ss_tot  (sklearn absent — inline by design)
```
**EVAL-01 upgrade vs Phase 10 TSTR-lite:** Phase 10 used a 1-layer LSTM-32, 3 init seeds, MSE+R². Phase 11's headline TSTR must additionally report **MAE and RMSE** (ROADMAP SC-1 explicitly lists R²/MAE/RMSE) and the planner picks ONE consistent soft-sensor architecture (LSTM **or** 1D-CNN — discretion). RMSE = sqrt(MSE); MAE = mean(|y−ŷ|) — both trivially added to the existing `train_eval_tstr` return dict. Keep the same `HELD_OUT_N=320`, init seeds {40,41,42}, and synthetic-pool-across-5-training-seeds protocol so numbers are comparable to the Phase 10 scaffolding table.

### Pattern 4: Long-form JSON schema (extend, never replace)

**What:** Every metric row is `{model_kind, pipeline, seed, metric_name, scale, value}`. Top-level adds `models[]`, `data_hash`, `data_hash_verification`, `schema` string.
**When to use:** All three Phase 11 JSON outputs.
**Example:**
```json
{"model_kind": "quantum", "pipeline": "A", "seed": 42,
 "metric_name": "emd", "scale": "OD", "value": 1.0520125260633941}
```
Verified present in `revision/results/baseline_comparison.json` (`rows[]` has 1710 entries; `schema` = `"long-form rows[] + models[] aggregate (D-10-16)"`).

### Anti-Patterns to Avoid

- **Editing `revision/core/`** — D-11-10 / D-10-13 invariant. `git diff revision/core/` MUST be empty after Phase 11. Post-hoc nets and TSTR forecaster live in drivers/notebook source, NOT core (Phase 09.1 plan-04 explicitly verified zero core diff).
- **`from sklearn.metrics import ...`** — sklearn is absent from `qgan_env`. Use inline `r2_score_inline` + hand MAE/RMSE (Phase 10 precedent).
- **`multiprocessing.Pool`** — Phase 09.1 RESEARCH Pitfall 4 (carried into D-10-24). Use OS-process parallelism (`xargs -P 2`) only if a sweep driver is even needed; most Phase 11 work is single-process aggregation over already-trained artifacts and is fast (<5 min).
- **Regenerating samples** — D-11-08. Phase 11 never calls `train_wgan_gp`, never instantiates a generator for sampling. It reads `samples.npy` only.
- **Real-data leakage into TSTR training** — the soft-sensor trains on synthetic ONLY; real data appears only in the held-out eval set (`real_windowed_OD[:320]`). For augmentation (EVAL-04), the "real" portion mixed in must come from `real_windowed_OD[320:]` (the train partition), NEVER from `[:320]` (the eval partition), or the lift number is contaminated.
- **Recomputing the quantum data-hash by grep** — Phase 09.1 quantum `config.yaml` files predate D-10-15 and carry **no `data_hash` field** (verified). Quantum data-equivalence is established BY CONSTRUCTION (same `load_and_preprocess('data.csv')` OD tensor). Assert the hash on the 50 baseline configs + recompute-from-source; do not expect it in quantum configs.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OD reconstruction from `[-1,1]` samples | New inverse-transform code | `reconstruct_od` (verbatim from `_build_baseline_notebook.py:167-210`) + `revision.core.preprocessing.inverse_logreturns` | Pipeline-B anchoring uses a specific seeded `od_starts` draw (`seed*7919+1`); re-deriving it risks divergence from Phase 10 numbers |
| EMD / ACF / moments / DTW / JSD | New metric math | `revision.core.eval.{compute_emd,compute_acf,compute_moments,compute_dtw,compute_jsd}` | D-11-10 mandate; v1.0/v1.1 locked behavioral decisions (raw-sample EMD, FFT ACF, Fisher kurtosis, ddof=0) baked in |
| Real windowed OD + data hash | Re-reading CSV manually | `revision.core.data.load_and_preprocess('data.csv')` then `rolling_window(OD,10,2)`; `sha256(OD.tobytes())[:16]` | Hash `91e447d4624e25b3` is the cross-phase invariant; deviating breaks the BASE-01/03 consistency proof |
| TSTR forecaster + R² | New LSTM/R² | `TSTRLiteLSTM` + `r2_score_inline` + `train_eval_tstr` (verbatim from `_build_baseline_notebook.py:394-440`); add MAE/RMSE to its return | Phase 10 numbers (`tstr["real_only_baseline"]` R²=-13.35, quantum|B R²=0.994) must remain reproducible as the comparison anchor |
| TimeGAN scores | A custom "similarity score" | Faithful re-implementation of Yoon et al. `predictive_metrics.py` / `discriminative_metrics.py` algorithm (definitions pinned below) | Reviewers explicitly asked for *standard* utility tests; a bespoke score weakens the rebuttal (CONTEXT `<specifics>`) |

**Key insight:** Almost every building block already exists in-tree and is verified-on-disk. Phase 11 is ~90% wiring existing verified components and ~10% net-new faithful TimeGAN GRU code. The risk is not "can we build it" but "do the numbers stay reproducible against the Phase 10 anchor" — so verbatim reuse of `reconstruct_od` / `train_eval_tstr` is mandatory, not optional.

## TimeGAN Score Definitions (the EVAL-02/03 pin)

**Reference (cite in JSON metadata):** Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, "Time-series Generative Adversarial Networks," NeurIPS 2019. Canonical code: `github.com/jsyoon0823/TimeGAN`, files `metrics/predictive_metrics.py`, `metrics/discriminative_metrics.py`, `utils.py`. **Record the repo URL + the commit/branch (`master`) in `predictive_discriminative.json` metadata** (D-11-04 discretion note: zero-cost, strictly more defensible).

The reference is **TensorFlow 1** (`tf.nn.rnn_cell.GRUCell`, `tf.train.AdamOptimizer()`). The project is **torch-only, CPU statevector**. Re-implement the *algorithm* faithfully in torch — same architecture, same metric formula, same split. Note these are designed for *multivariate variable-length sequences*; this project has *univariate fixed length-10 windows*. Adaptation is required and must be documented (Pitfall 1).

### Predictive score — `[CITED: jsyoon0823/TimeGAN/metrics/predictive_metrics.py]`

| Property | Canonical value | This-project adaptation |
|----------|-----------------|--------------------------|
| Post-hoc net | **Single-layer GRU**, `tanh` activation, + 1 sigmoid FC output | Single-layer `torch.nn.GRU(input_size=1, hidden_size=H)` + `Linear(H,1)` |
| `hidden_dim` | `int(dim/2)` where `dim` = feature count | `dim` = 1 (univariate) → `int(1/2)=0` is **degenerate**. Adapt: treat the length-10 window as the sequence; pin `hidden_dim` to a documented small value (recommend `H = max(1, int(WINDOW_LENGTH/2)) = 5`, or follow the TimeGAN-spirit `≈ input_dim` per D-11-04). **The planner must pick and record one value; flag as ASSUMED until confirmed.** |
| Iterations | **5000** | Reuse 5000 (cheap on length-10 windows) or document a reduced value if wall-time forces it |
| Batch size | **128** | Reuse 128 (3840 synth windows ≫ 128) |
| Optimizer / LR | `tf.train.AdamOptimizer()` = **Adam, LR 1e-3** (TF1 default) | `torch.optim.Adam(lr=1e-3)` |
| Train data | **Synthetic** (`generated_data`) | Synthetic OD windows (per generator × pipeline) |
| Test data | **Original/real** | `real_windowed_OD` |
| Input X | `generated_data[i][:-1, :(dim-1)]` — all steps except last, all but final feature | Univariate adaptation: `X = window[:, :-1]` (steps 0..8), predict step 9 — i.e. the **same one-step-ahead construction as D-11-01** (this is why D-11-01 deliberately aligns the TSTR task with the predictive score) |
| Target Y | `generated_data[i][1:, (dim-1)]` — next-step of final feature | `y = window[:, -1]` (one-step-ahead) |
| Metric | **Mean Absolute Error**, averaged over test samples. Lower = better | `mean(|y − ŷ|)` over `real_windowed_OD` |

### Discriminative score — `[CITED: jsyoon0823/TimeGAN/metrics/discriminative_metrics.py]`

| Property | Canonical value | This-project adaptation |
|----------|-----------------|--------------------------|
| Post-hoc net | **Single-layer GRU**, `tanh` + 1 FC → sigmoid logit | Single-layer `torch.nn.GRU(input_size=1, hidden_size=H)` + `Linear(H,1)` |
| `hidden_dim` | `int(dim/2)` | Same univariate caveat as predictive — pin & record H (recommend same H as predictive for consistency) |
| Iterations | **2000** | Reuse 2000 |
| Batch size | **128** | Reuse 128 |
| Optimizer / LR | Adam, LR 1e-3 (TF1 default) | `torch.optim.Adam(lr=1e-3)` |
| Labels | real → `tf.ones_like` (1); synthetic → `tf.zeros_like` (0) | real windows label 1, synth windows label 0 |
| Train/test split | `train_test_divide(...)`, **`train_rate = 0.8`** (`utils.py`); random permutation, 80% train / 20% test, applied to BOTH real and synthetic pools | 80/20 split of real pool and of synthetic pool independently; `np.random.permutation` with a recorded seed |
| Metric | `discriminative_score = np.abs(0.5 − accuracy)` on the held-out test set. Lower = better (classifier can't tell real from synth) | identical formula |

**Per-seed roll-up (D-11-05):** compute both scores per training seed {42,43,44,45,46} (each seed's synthetic pool), then report `mean ± std` across the 5 seeds, per (model_kind, pipeline). Quantum + wgan_mlp/cnn/lstm + vae + ar = 6 model kinds × 2 pipelines.

## Orlandi-style Augmentation Design (EVAL-04)

**Reference:** Orlandi et al. [26] (AIChE) — cited in REQUIREMENTS R1-m5 / EVAL-04. The methodology: train a downstream model on real-only data, then on real+synthetic mixtures, and report the downstream-task performance lift attributable to synthetic augmentation. (Could not fetch the exact Orlandi paper text in this session — `[ASSUMED]` that the standard "mixing-ratio lift curve" interpretation applies; the CONTEXT D-11-06/07 already locks the concrete design, so this is low-risk.)

**Concrete design (locked by D-11-06/07):**
- Downstream task = the **same one-step-ahead OD soft-sensor** as EVAL-01 (D-11-06) — one task, two uses.
- Conditions per generator × pipeline:
  - `real_only`: train soft-sensor on `real_windowed_OD[320:]` only (this IS the Phase 10 `real_only_baseline`, R²=-13.35 ± 0.58 — reuse as the anchor).
  - `real + synthetic @ +25% / +50% / +100% / synthetic-only`: augment the real train partition with N synthetic windows where N = {0.25, 0.5, 1.0}×|real_train|, plus a synthetic-only condition.
- **Eval set is always `real_windowed_OD[:320]`** (never mixed, never the train partition).
- Delta table = ΔR², ΔMAE, ΔRMSE vs the `real_only` baseline, per generator → "lift curve per generator" (D-11-07).
- Emit to `revision/results/augmentation.json` in long-form (`metric_name` ∈ {`r2_delta`,`mae_delta`,`rmse_delta`}, an `injection_ratio` field added to rows, `scale="OD"`).

**Pitfall:** the real train partition is small (`real_windowed_OD[320:]` ≈ 64 windows on the 384-window split). The Phase 09.1 FLAG-E note already flagged the 60× synthetic train-set advantage — the augmentation lift is a **lower bound**, not a matched-budget comparison. State this explicitly in `augmentation.json` metadata and the summary (it is the honest framing CONTEXT `<specifics>` mandates).

## Common Pitfalls

### Pitfall 1: TimeGAN `hidden_dim = int(dim/2)` is degenerate for univariate data
**What goes wrong:** The canonical formula assumes multivariate sequences (`dim` = feature count). Here data is univariate (`dim`=1), so `int(1/2)=0` → a zero-width GRU. Blindly copying the formula crashes or produces a trivial model.
**Why it happens:** TimeGAN's reference data is multivariate (stock: 6 features, energy: 28). This project's "sequence" is a length-10 window of a single OD-derived signal.
**How to avoid:** Reframe: `input_size=1`, sequence length=10 (or 9 for one-step). Pin `hidden_dim` to a small documented constant — recommend `H≈5` (≈ `int(WINDOW_LENGTH/2)`) or `H≈WINDOW_LENGTH` per D-11-04's "hidden dim ≈ input_dim". **Record the chosen H + the adaptation rationale in `predictive_discriminative.json` metadata.** This is the single most important judgement call; surfaced in Assumptions Log (A1).
**Warning signs:** GRU constructor error; predictive MAE identical across all generators (model has no capacity).

### Pitfall 2: sklearn assumed present
**What goes wrong:** `from sklearn.metrics import r2_score, mean_absolute_error` → `ModuleNotFoundError` at execution time, after compute already burned.
**Why it happens:** sklearn is ubiquitous in ML tutorials; easy to assume.
**How to avoid:** Verified absent in `qgan_env`. Reuse `r2_score_inline` (Phase 10); `MAE = np.mean(np.abs(y-yhat))`; `RMSE = np.sqrt(np.mean((y-yhat)**2))`. Add these two lines to the existing `train_eval_tstr` return dict.
**Warning signs:** any `import sklearn` in a plan task action.

### Pitfall 3: Pipeline-B OD reconstruction uses a seeded od_start draw
**What goes wrong:** Re-deriving OD from log-returns with a different `od_start` selection produces different OD windows → Phase 11 numbers don't reconcile with Phase 10's `baseline_comparison.json`.
**Why it happens:** `reconstruct_od` Pipeline-B branch draws `od_start` per window via `np.random.default_rng(seed*7919+1).choice(od_starts_pool, ...)`. This exact RNG seeding is load-bearing.
**How to avoid:** Copy `reconstruct_od` verbatim from `_build_baseline_notebook.py:167-210`. Do not "clean it up."
**Warning signs:** Phase 11 quantum|B EMD ≠ Phase 10's 0.0276 ± 0.0046.

### Pitfall 4: Quantum runs have no `data_hash` field
**What goes wrong:** Asserting `config.yaml["data_hash"]` exists on quantum runs → KeyError; or concluding the data differs.
**Why it happens:** Phase 09.1 quantum runs predate the D-10-15 data-hash convention (verified: quantum `config.yaml` has no `data_hash`).
**How to avoid:** Assert hash equality across the **50 baseline configs** + recompute from `load_and_preprocess('data.csv')`. Treat quantum equivalence as by-construction (same loader, same CSV) — exactly as Phase 10's `data_hash_verification.quantum_equivalence` documents.
**Warning signs:** KeyError on `data_hash` for a `transform_ablation/runs/...` config.

### Pitfall 5: Eval/train partition leakage in augmentation
**What goes wrong:** Mixing real windows from `[:320]` into the augmentation train set inflates lift (the model sees eval data at train time).
**Why it happens:** Easy to grab `real_windowed_OD` wholesale when assembling the mixed set.
**How to avoid:** Real portion in EVAL-04 always = `real_windowed_OD[320:]`; eval always = `real_windowed_OD[:320]`. Assert disjointness in the driver.
**Warning signs:** augmentation R² jumps to ~0.99 for the real-only condition (Phase 10 anchor is R²=-13.35 — a positive real-only R² means leakage).

### Pitfall 6: `data.csv` path is repo-root relative
**What goes wrong:** Drivers run from a worktree / different cwd → `load_and_preprocess('data.csv')` fails or reads the wrong file.
**Why it happens:** Phase 10 configs record `csv_path: data.csv` (relative). `data.csv` exists at repo root (`/Users/shawngibford/dev/phd/qGAN/data.csv`, 54437 bytes, dated Aug 2025 — the canonical training CSV; NOT the `fake.csv`/`real.csv` exports).
**How to avoid:** Resolve `data.csv` to an absolute path anchored at repo root in the driver; document it. The recomputed hash must equal `91e447d4624e25b3`.
**Warning signs:** data_hash ≠ `91e447d4624e25b3`.

## Runtime State Inventory

> Phase 11 is evaluation-only — no renames, no migrations, no new persisted generator state. Inventory included for completeness.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | 60 frozen `samples.npy` (50 baseline + 10 quantum), each `(3840,10)` float64 in `[-1,1]`; `inverse_kwargs.npz`; `config.yaml`. All READ-ONLY for Phase 11. | None — consume as-is (D-11-08) |
| Live service config | None — no external services. Local-Mac statevector only; no quantum execution in Phase 11 (samples frozen). | None |
| OS-registered state | None — no schedulers, daemons, or registered tasks. | None |
| Secrets/env vars | None — no secrets, no env-var-driven config. | None |
| Build artifacts | New: `revision/results/{tstr,predictive_discriminative,augmentation}.json` (+ optional notebook). No package re-install (no new deps). `revision/core/__pycache__` unaffected (core untouched). | None — additive only |

**Nothing found in 4 of 5 categories:** verified by inspection of the revision tree, `.planning/config.json`, and CONTEXT (no service/secret/OS-state surface in an evaluation-only phase).

## Validation Architecture

> `.planning/config.json` has `nyquist_validation: false` — Nyquist sampling validation is DISABLED for this run. The section below lists natural correctness oracles the planner should still wire as plain assertions (the objective explicitly asked for these to be flagged).

### Natural correctness oracles (assert even without Nyquist)

| Oracle | Assertion | Why it's a true invariant |
|--------|-----------|---------------------------|
| Data-hash equality | `sha256(load_and_preprocess('data.csv')['OD'].tobytes())[:16] == "91e447d4624e25b3"` and equals all 50 baseline `config.yaml` `data_hash` fields | Cross-phase identical-data proof (D-10-15); a mismatch invalidates every comparison |
| Inverse-transform round-trip | `reconstruct_od` Pipeline B: `inverse_logreturns(forward_logreturns(OD)) ≈ OD` within 1e-8 (ABL-01 already verified the helpers; re-assert at the wrapper level) | Guarantees OD-scale numbers are not corrupted by the inverse |
| Phase 10 reproduction sanity | Phase 11's recomputed quantum\|B OD-EMD ≈ Phase 10's `0.0276 ± 0.0046`; TSTR-lite quantum\|B R² ≈ `0.994` | Proves verbatim reuse of `reconstruct_od`/`train_eval_tstr` didn't drift |
| Sample shape invariant | every loaded `samples.npy` is `(3840, 10)` float64 | A different shape signals a wrong/corrupt artifact |
| Augmentation partition disjointness | `set(real_train_idx) ∩ set(real_eval_idx) == ∅` (eval = `[:320]`, train = `[320:]`) | Prevents the EVAL-04 leakage pitfall |
| `revision/core/` untouched | `git diff --stat revision/core/` is empty after Phase 11 | D-11-10 / D-10-13 hard invariant |

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing `revision/tests/{test_classical.py,test_nonadversarial.py}`) |
| Config file | none detected — tests run via `./qgan_env/bin/python -m pytest revision/tests/` |
| Quick run command | `./qgan_env/bin/python -m pytest revision/tests/ -x -q` |
| Full suite command | `./qgan_env/bin/python -m pytest revision/tests/ -q` |

### Wave 0 Gaps
- [ ] `revision/tests/test_utility.py` — assert `reconstruct_od` output shape `(3840,10)`, data-hash equality, TSTR round-trip vs Phase 10 anchor (optional but recommended; project has a `tests/` convention)
- [ ] No framework install needed — pytest available via `qgan_env`.

*(Nyquist disabled — these are plain assertions / pytest checks, not a Nyquist sampling regime.)*

## Code Examples

### Resolve run dir (quantum vs baseline) — verified contract
```python
# Source: revision/_build_baseline_notebook.py:167-172 (verified on disk)
def _run_base(model_kind, pipeline, seed):
    if model_kind == "quantum":
        return Path(f"revision/results/transform_ablation/runs/{pipeline}/{seed}")
    return Path(f"revision/results/baselines/runs/{model_kind}/{pipeline}/{seed}")
# MODEL_KINDS = ["quantum","wgan_mlp","wgan_cnn","wgan_lstm","vae","ar"]
# PIPELINES   = ["A","B"];  SEEDS = [42,43,44,45,46]   → 6×2×5 = 60 run dirs, all verified present
```

### Inline metrics (sklearn absent) — extend train_eval_tstr return
```python
# Source: revision/_build_baseline_notebook.py:405-440 (verified) + Phase 11 additions
def r2_score_inline(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
# Phase 11 additions for ROADMAP SC-1 (R²/MAE/RMSE):
mae  = float(np.mean(np.abs(y_true - y_pred)))
rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
```

### Faithful TimeGAN predictive net (torch port of the pinned TF1 algorithm)
```python
# Algorithm source: jsyoon0823/TimeGAN/metrics/predictive_metrics.py (CITED)
# input_size=1 (univariate adaptation, Pitfall 1); H pinned & recorded in JSON metadata
class PredictiveGRU(torch.nn.Module):
    def __init__(self, H):           # H ≈ WINDOW_LENGTH or int(WINDOW_LENGTH/2) — RECORD choice
        super().__init__()
        self.gru = torch.nn.GRU(input_size=1, hidden_size=H, num_layers=1, batch_first=True)
        self.fc  = torch.nn.Linear(H, 1)            # sigmoid in TF ref; identity ok for MAE regression
    def forward(self, x):                            # x: (B, T-1, 1)
        out, _ = self.gru(x); return self.fc(out)    # predict next-step sequence
# train: synthetic windows, Adam(lr=1e-3), 5000 iters, batch 128
# test : real windows; score = mean_absolute_error(y_real, y_pred). Lower better.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Phase 10 TSTR-lite (sanity scaffolding, R²/MSE only, 1-layer LSTM-32) | Phase 11 headline TSTR (R²/MAE/RMSE, one consistent chosen arch, same 320-split) | Phase 11 | Comparable numbers; adds MAE/RMSE per ROADMAP SC-1; still references Phase 10 anchor |
| No standardized utility score | Faithful TimeGAN predictive + discriminative (Yoon et al. 2019, canonical repo) | Phase 11 | Rebuttal-grade "standard test" evidence; reference pinned in JSON metadata |
| Fidelity reported single-scale | Dual-scale (`scale: "log_return" | "OD"`) on every metric | Phase 11 EVAL-05 | Directly answers R1-m3 / R1-M3 scale ambiguity |

**Deprecated/outdated:** None — Phase 11 builds on current Phase 10/09.1 contracts (all verified May 2026).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | TimeGAN post-hoc `hidden_dim` for this *univariate length-10* setting should be `H ≈ WINDOW_LENGTH` or `int(WINDOW_LENGTH/2)≈5` (the canonical `int(dim/2)` is degenerate at dim=1) | TimeGAN Score Definitions / Pitfall 1 | MEDIUM — wrong H → under/over-capacity post-hoc net; mitigated by recording the chosen value + rationale in JSON metadata; planner should lock H explicitly (this is a D-11-04 "research item, not user-locked" point) |
| A2 | Orlandi et al. [26] methodology = standard "mixing-ratio lift curve vs real-only baseline" | Orlandi-style Augmentation Design | LOW — D-11-06/07 already lock the concrete design independent of the exact Orlandi text; only the framing citation depends on it |
| A3 | `real_windowed_OD[320:]` ≈ 64 windows (Phase 10 reported `n_train_real=65` for its split) | Pattern 3 / EVAL-04 | LOW — exact count differs by ≤1 depending on the precise 384-window indexing; compute it live, don't hardcode |
| A4 | TF `tf.train.AdamOptimizer()` default LR = 1e-3 (TF1 default) | TimeGAN Score Definitions | LOW — well-documented TF1 default; reproduced as `torch.optim.Adam(lr=1e-3)` |
| A5 | Phase 11 work completes well within local-Mac budget (aggregation over frozen artifacts; no GAN training; longest op is 6×2 TimeGAN nets × 5000/2000 iters on tiny length-10 windows) | (compute) | LOW — Phase 10 VAE was ~16s; these post-hoc nets are smaller. Expect <15 min total. |

## Open Questions

1. **TimeGAN post-hoc `hidden_dim` for univariate length-10 windows**
   - What we know: canonical formula `int(dim/2)`; D-11-04 says "hidden dim ≈ input_dim, ~1–2 layers"; reference uses single-layer GRU.
   - What's unclear: the exact H. `int(1/2)=0` is degenerate; D-11-04 says ≈ input_dim but "input_dim" is ambiguous for a univariate windowed signal (1 feature vs 10-step window).
   - Recommendation: planner locks `H = WINDOW_LENGTH = 10` (matches D-11-04 "≈ input_dim" reading the window as the input) OR `H = 5`; record the value + rationale in `predictive_discriminative.json` metadata; surface in plan as an explicit decision. Either is defensible; consistency (same H for predictive and discriminative) matters more than the exact value.

2. **Soft-sensor architecture: LSTM (Phase 10 precedent) vs 1D-CNN (EVAL-01 "or")**
   - What we know: EVAL-01 allows either; Phase 10 used LSTM-32; D-11 discretion says pick one, used consistently.
   - What's unclear: nothing blocking.
   - Recommendation: reuse the Phase 10 `TSTRLiteLSTM` (1-layer LSTM-32) for direct comparability with the already-published Phase 10 scaffolding table, just add MAE/RMSE. Lowest-risk, most-comparable choice. Document the architecture in `tstr.json`.

3. **Should EVAL-05 dual-scale rows live in `tstr.json`/`augmentation.json` or a separate `fidelity_dualscale.json`?**
   - What we know: ROADMAP SC-4 says "visible as explicit scale fields in JSON outputs"; long-form schema already has a `scale` field.
   - Recommendation: emit a dedicated long-form block (e.g. `fidelity_dualscale.json` or a `fidelity` array in the utility JSON) carrying every `revision.core.eval` metric twice — once `scale="OD"`, once `scale="log_return"` (Pipeline B; Pipeline A is OD-only — log-return scale n/a, emit explicit `"log_return": null` or omit with a documented reason). Planner decides file boundary; the schema is the constraint, not the filename.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| python venv `qgan_env` | all drivers | ✓ | 3.11 | — |
| torch | post-hoc nets, TSTR | ✓ | (qgan_env) | — |
| numpy | I/O, metrics | ✓ | (qgan_env) | — |
| scipy | eval.py (EMD/kurtosis/JSD) | ✓ | (qgan_env) | — |
| statsmodels | eval.py ACF | ✓ | installed | — |
| fastdtw | eval.py DTW | ✓ | installed | — |
| pyyaml | read config.yaml | ✓ | (qgan_env, used by run_baselines) | — |
| pytest | optional Wave-0 tests | ✓ | (qgan_env) | plain assertions in driver |
| **scikit-learn** | (only if naively used for R²/MAE) | **✗** | — | **inline `r2_score_inline` + np MAE/RMSE (Phase 10 precedent — REQUIRED fallback, do not install)** |
| 60 frozen samples.npy | all EVAL-* | ✓ | data_hash 91e447d4624e25b3 | — (hard prerequisite; all 50 baseline + 10 quantum verified on disk) |
| data.csv (repo root) | data-hash + real windows | ✓ | 54437 B, Aug 2025 | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** scikit-learn → inline metric math (this is the established project pattern, not a degraded fallback).

## Sources

### Primary (HIGH confidence)
- Codebase (verified on disk, May 2026): `revision/core/{eval,preprocessing,data,__init__}.py`, `revision/run_baselines.py`, `revision/_build_baseline_notebook.py`, `revision/results/baseline_comparison.json`, all 60 `samples.npy`/`config.yaml`/`inverse_kwargs.npz` run dirs, `revision/results/baselines/sweep_status.json` (50/50 complete), `qgan_env` import probes (sklearn ABSENT; torch/numpy/scipy/statsmodels/fastdtw/pyyaml/pennylane 0.43.0 present).
- `.planning/phases/11-utility-evaluation/11-CONTEXT.md`, `.planning/phases/10-classical-baselines/10-CONTEXT.md`, `.planning/phases/09.1-.../09.1-04-SUMMARY.md`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/config.json`.
- [jsyoon0823/TimeGAN/metrics/predictive_metrics.py](https://github.com/jsyoon0823/TimeGAN/blob/master/metrics/predictive_metrics.py) — single-layer GRU, hidden=int(dim/2), 5000 iters, Adam, batch 128, MAE next-step.
- [jsyoon0823/TimeGAN/metrics/discriminative_metrics.py](https://github.com/jsyoon0823/TimeGAN/blob/master/metrics/discriminative_metrics.py) — single-layer GRU, hidden=int(dim/2), 2000 iters, Adam, batch 128, `|0.5−acc|`, real=1/synth=0.
- [jsyoon0823/TimeGAN/utils.py](https://github.com/jsyoon0823/TimeGAN/blob/master/utils.py) — `train_test_divide`, `train_rate=0.8`.

### Secondary (MEDIUM confidence)
- [jsyoon0823/TimeGAN README](https://github.com/jsyoon0823/TimeGAN) — NeurIPS 2019 reference; `--module gru --hidden_dim 24 --num_layer 3` example (the *generator* config, distinct from post-hoc metric nets).
- [Supplementary Materials: TimeGAN, NeurIPS 2019 (van der Schaar lab)](https://www.vanderschaar-lab.com/papers/NIPS2019_TGAN_Supplementary.pdf) — post-hoc network description.

### Tertiary (LOW confidence)
- Orlandi et al. [26] augmentation methodology — exact paper text not fetched this session; design is locked by D-11-06/07 so dependency is minimal (A2).

## Metadata

**Confidence breakdown:**
- Codebase contracts (reconstruct_od, schema, sample shapes, sklearn-absent, run provenance): HIGH — all verified by direct disk inspection and import probes.
- TimeGAN score definitions: HIGH — pinned to exact canonical source files; the only open item is the univariate `hidden_dim` adaptation (flagged A1, a documented judgement call, not an unknown).
- Augmentation design: HIGH for the locked D-11-06/07 mechanics; LOW only for the Orlandi-paper framing citation (A2).
- Pitfalls: HIGH — derived from verified code behavior and the Phase 09.1/10 documented flags.

**Research date:** 2026-05-17
**Valid until:** 2026-06-16 (30 days — codebase contracts are frozen by D-11-08; TimeGAN reference is a stable 2019 publication)
