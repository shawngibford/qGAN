# Agent 4 — Prose↔Code Consistency Findings

**Audit target:** HEAD `50658a6` (v1.2.1)
**Files audited:** main + supp .tex against `revision/` code
**Scope:** Prose descriptions of methodology, hyperparameters, model definitions verified against actual code.

## Summary
- BLOCK: 0 findings
- FLAG: 2 findings
- NIT: 1 finding

## BLOCK findings
(none)

## FLAG findings

### F-1: Pipeline C described as "Pipeline B followed by inverse Lambert W" — operation order is incorrect in both main and supp

- **Prose location (main):** `main (4) copy.tex` line 291
- **Prose quote:** "*Pipeline~C* (Pipeline~B followed by an inverse Lambert~$W$ heavy-tail correction, the v1.1 published pipeline)"
- **Prose location (supp):** `supp_material.tex` lines 581 + 604
- **Prose quote (supp 581):** "*Pipeline~C* (Pipeline~B followed by inverse Lambert~$W$)"
- **Prose quote (supp 604):** "C: Pipeline~B followed by inverse Lambert~$W$, dropped"
- **Code location:** `revision/core/data.py:269-282` (`load_and_preprocess` — Pipeline C path used by `run_ablation.py:154-164`)
- **Code reality (data.py:269-282):**
  ```
  # Cell 15 — normalize log-delta
  norm_log_delta, mu, sigma = normalize(log_delta)
  # Cell 18 — find optimal Lambert W delta, apply inverse Lambert
  delta = find_optimal_lambert_delta(norm_log_delta.numpy())
  transformed_norm_log_delta = inverse_lambert_w_transform(norm_log_delta, delta)
  # Cell 21 — rescale to [-1, 1]
  min_val = torch.min(transformed_norm_log_delta)
  max_val = torch.max(transformed_norm_log_delta)
  scaled_data = -1.0 + 2.0 * (transformed_norm_log_delta - min_val) / (max_val - min_val)
  ```
- **Discrepancy:** Actual Pipeline C order is `log-returns → standardize → inverse_lambert_w → rescale to [-1,1]`. The Lambert W is inserted **between** the standardize and rescale steps. The prose framing "Pipeline B (= log-return → standardize → rescale) followed by inverse Lambert W" implies the Lambert W is appended **after** rescale-to-[-1,1], which is not what the code does. A reader trying to reproduce Pipeline C by literally chaining "Pipeline B + Lambert W" would produce a different transformation. Note: supp §A.7 ablation Equations 1+2 only define Pipeline B; Pipeline C's chain is never written explicitly.
- **Suggested fix:** Reword as "Pipeline C: log-return → standardize → **inverse Lambert W** → rescale to [-1, 1] (i.e., Pipeline B with an inverse Lambert W step inserted between standardization and rescaling, matching the v1.1 published pipeline)". Apply in main line 291 and supp lines 581, 604.

### F-2: Main text claim "the only component that differs between quantum and classical entrants is the generator" — true for adversarial only

- **Prose location:** `main (4) copy.tex` lines 192-194 and 259-262
- **Prose quote (192-194):** "All adversarial models in this study are trained under a single matched-budget contract so that the only component that differs between the quantum and classical entrants is the generator"
- **Prose quote (259-262):** "A classical critic also makes the comparison against the matched classical WGAN-GP baselines clean: the only component that differs between the quantum and classical entrants is the generator"
- **Code location:** `revision/run_matched2000.py:649-748` (VAE + AR(2) training paths)
- **Code reality:**
  - VAE (`run_matched2000.py:649-731`): uses a **single Adam ELBO loop with lr=1e-3**, **does not use the shared critic**, **does not use gradient penalty**, and `VAEBaseline.sample()` does **not** apply the `*0.1` post-scaling (`nonadversarial.py:105-117`).
  - AR(2) (`run_matched2000.py:746-748`): fit via `np.linalg.lstsq`, **no optimizer / no critic / no epochs at all**.
- **Discrepancy:** The "only component that differs is the generator" statement is true within the *adversarial cluster* (quantum vs. wgan_mlp/cnn/lstm — same critic, same Adam/β/η/n_critic/λ_GP/BS), but the VAE and AR(2) appear in the same nine-model comparison table (Table 2) without an asterisk that distinguishes the four-vs-three adversarial contract from the non-adversarial outliers. Main line 207-209 *does* note "AR(2) is fit in closed form... the VAE uses a single-Adam ELBO loop... and does not consume the shared critic or gradient-penalty terms", which softens this — but the headline claim "the only component that differs is the generator" reads as a global protocol statement and is not qualified at line 194 or 261.
- **Suggested fix:** Add a parenthetical to line 194 and 261: "(within the WGAN-GP adversarial cohort — IQP:SEL, V1/V2/V3, wgan_mlp/cnn/lstm; the VAE and AR(2) non-adversarial baselines use their own native training loops as described above/below)".

## NIT findings

### N-1: Supp figure 6 caption "Four-stage preprocessing pipeline" counts raw OD as a stage

- **Prose location:** `supp_material.tex` line 531
- **Prose quote:** "Four-stage preprocessing pipeline (Pipeline~B, native): raw OD ($n=778$), log-returns ..., standardized ..., and linearly rescaled to $[-1, 1]$"
- **Code location:** `revision/run_ablation.py:134-152` (Pipeline B branch)
- **Code reality:** Pipeline B in code has **three transformations** applied to raw OD: `forward_logreturns` (which combines diff + standardize in a single function — `preprocessing.py:42-59`) followed by linear rescale to [-1, 1] (`run_ablation.py:139`). The supp counts raw OD as "stage 1" to reach four panels.
- **Discrepancy:** Calling raw OD a "stage" of the preprocessing pipeline is loose; raw OD is the **input** to the pipeline, not a preprocessing stage. The same chain is correctly described as 3 stages in the main text line 285: "(log-return $\to$ standardization $\to$ rescale to $[-1, 1]$) as **Pipeline B**". Could mislead a reader who reads supp figure 6 first into thinking Pipeline B has 4 transformations.
- **Suggested fix:** Reword supp line 531 to "Four-panel visualization of the Pipeline~B preprocessing chain: input raw OD ($n=778$), and the three resulting stages — log-returns ..., standardized ..., and linearly rescaled to $[-1, 1]$".

## What was NOT flagged (sanity check)

All of the following high-stakes prose↔code agreements were verified consistent:

- **Pipeline B definition** (main 270-286, supp 542-572): log-returns → standardize → linearly rescale to [-1, 1] using global min/max of the standardized series — matches `run_ablation.py:134-152` exactly, with `forward_logreturns` (`preprocessing.py:42-59`) using `torch.mean` and `torch.std` (default ddof=1). **No Lambert W in Pipeline B in code**, confirmed.
- **n=5 seeds {42, 43, 44, 45, 46}** (main 309-310, supp 305, 378): matches `run_matched2000.py:109` (`SEED_SET = (42, 43, 44, 45, 46)`) and `run_timegan_scores.py:82` (`SEEDS = [42, 43, 44, 45, 46]`).
- **2000-epoch matched budget** (main 194-195, 206): matches `run_matched2000.py:168` (`MATCHED_EPOCHS = 2000`) and `revision/core/__init__.py:20` (`NUM_EPOCHS = 2000`). Strict accept gate at `run_matched2000.py:823` enforces this.
- **55-parameter IQP:SEL** (main 230-232, 323): matches `QuantumGenerator(num_qubits=5, num_layers=3, circuit_id='iqp_sel_55').count_params() = 55` (formula at `quantum.py:104-109`: `5 + 3*15 + 5 = 55`).
- **73-78 parameter classical baselines** (main 196, 381, abstract): matches `WGANMLPGenerator.count_params() = 74`, `WGANCNNGenerator = 73`, `WGANLSTMGenerator = 78` (`classical.py`). Table 2 (main line 381) explicitly labels 74p/73p/78p.
- **250881-param shared critic** (main 196, supp 629): matches `Critic(window_length=10, dropout_rate=0.2)` — `sum(p.numel())` returns exactly **250881**.
- **Hyperparameters** (main 197-201): Adam β₁=0.0/β₂=0.9 (`training.py` Adam init), η_G = 6.9173e-5 / η_C = 1.8046e-5 / n_critic = 9 / λ_GP = 2.16 / BS = 12 — all match `revision/core/__init__.py:11-22` exactly.
- **VAE: 562 parameters** (Table 2 main 381, supp 629): matches `VAEBaseline.count_params() = 562` (176+68+68+80+170 = 562, per docstring `nonadversarial.py:19-22`).
- **VAE degenerate generation regime, NOT posterior collapse** (main 360-362, 391-394, 520-536): correctly uses "degenerate generation regime" — matches handoff §5 directive and `nonadversarial.py` docstring (no "posterior collapse" anywhere in main).
- **AR(2): p=2** (main 207, Table 2, supp 397): matches `ARBaseline(p=2)` instantiation at `run_matched2000.py:748` and `run_baselines.py:387`. `count_params() = p+1 = 3` matches "(3p)" in Table 2.
- **The 9 generators in Table 2** (main 380): `IQP:SEL, V1, V2, V3, WGAN-MLP, WGAN-CNN, WGAN-LSTM, VAE, AR(2)` — exactly matches `MODEL_KINDS` at `run_timegan_scores.py:79-80` and `_MODEL_CHOICES` at `run_matched2000.py:184-187`.
- **V1/V2/V3 specs** (main 202-204, 237-240): V1=4-layer/range/75p, V2=8-layer/range/135p, V3=4-layer/linear/75p — matches `_QUANTUM_ANSATZ` dict at `run_matched2000.py:173-180` exactly.
- **TimeGAN-style discriminative + predictive metrics** (main 581, references to `revision/run_timegan_scores.py`): the implementation at `run_timegan_scores.py:1-90` is a faithful port of jsyoon0823/TimeGAN `metrics/predictive_metrics.py` + `metrics/discriminative_metrics.py` (NeurIPS 2019, master branch) — single-layer GRU + Linear head, Adam lr=1e-3, 5000/2000 iters, batch 128, with the documented `hidden_dim` adaptation for univariate input (locked H=WINDOW_LENGTH=10 with rationale in JSON metadata).
- **Pre-v1.0 historical DTW = 0.6843** (supp 312, 315): correctly disclosed as "frozen pre-v1.0 best-case checkpoint... legacy preprocessing pipeline (Pipeline A; not the matched-budget Pipeline B), under a single-seed best-case selection, and at a different epoch budget" — matches handoff §5 directive.
- **Real-data lag-1 ACF = -0.064** (main 479): matches `cross_model_acf_overlay.json` source value `-0.06411182880401611` (rounded). Source: `revision.core.data.load_and_preprocess()['log_delta']`.
- **384 training windows from 778 raw OD points** (main 274, 283-284): verified — 778 raw OD → 777 log-returns → 384 windows at stride 2 (formula `(777-10)//2 + 1 = 384`).
- **Lambert W functions retained for ablation reproducibility** (`data.py:118-144`, `preprocessing.py:33-36`): per handoff hard prohibition #1, the existence of `lambert_w_transform` / `inverse_lambert_w_transform` in the codebase is correct and intentional; the code documentation (`preprocessing.py:10-22` D-10-05 note) explicitly states this is for ablation reproducibility only.
