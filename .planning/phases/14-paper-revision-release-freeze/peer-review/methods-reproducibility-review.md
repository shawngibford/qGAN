# Peer Review — Methods & Reproducibility

**Reviewer role:** Methods/Reproducibility auditor (external referee perspective)
**Target documents:**
- `docs/methods_full.md` (7 sections + provenance footer)
- `docs/reviewer_response.md`
- Supporting JSON in `results/*.json`

**Verdict (TL;DR):** **CONDITIONAL — methods doc is internally faithful and the executable provenance gate passes, but reproduction of the headline result from a fresh clone is BLOCKED by two missing artifacts (best_checkpoint.pt is .gitignored; requirements.txt uses `>=` not exact pins). Other classical/quantum 2000-epoch runs are reproducible from scratch and the strict_accept gate would re-accept them, but bit-for-bit identity with reported numbers is not guaranteed because `torch.use_deterministic_algorithms(True)` is not set.**

---

## Severity legend
- **BLOCKER** — Stops external reproduction outright.
- **HIGH** — Reproduction is theoretically possible but a referee would have to ignore documentation and fix things themselves.
- **MEDIUM** — Methodological clarity / fairness disclosure issue; not a reproduction blocker.
- **LOW** — Stylistic/citation completeness.

---

## 1. Methods document faithfulness — section-by-section verification

### §1 Dataset — VERIFIED
- `data.csv` actually has 778 rows (verified by `pandas.read_csv` count, file path `/Users/shawngibford/dev/phd/qGAN/data.csv`).
- Methods doc claims `raw_csv_rows = 778`, `log_return_rows = 777`, `window_length = 10`, `window_stride = 2`, `rolling_windows = 384`. Window arithmetic: `(777-10)//2 + 1 = 384` ✓ correct.
- `core/data.py:227-296` (`load_and_preprocess`) implements the documented pipeline order: CSV load → fillna(rolling-10-mean) → dropna → log-delta with dither (seed=42, magnitude 0.005) → normalize → find_optimal_lambert_delta → inverse_lambert_w_transform → rescale to [-1,1] → rolling_window(W=10, stride=2). Order and operations match `methods_full.md §1` exactly.
- Single-campaign caveat (`n_independent_campaigns=1`) is in `methods_full.md:46-47` and `reviewer_response.md:40` — explicit and accurate.

### §2 Models — VERIFIED with one MEDIUM finding

| Model | Methods doc claim | Code verification | Status |
|---|---|---|---|
| `iqp_sel_55_headline` (§2.a) | 55 params, 5 qubits, 3 layers, range entangler, RX_only final rotation, checkpoint epoch 1969 | `canonical_config_lock.json` confirms | OK |
| `iqp_sel_55_repro` (§2.b) | Same arch, no checkpoint, 2000ep fresh train | matches `_QUANTUM_ANSATZ` table + `_REPRO_MODEL` constant in `run_matched2000.py:117-127` | OK |
| `V1/V2/V3` (§2.c-e) | 75/135/75 params; depth 4/8/4; range/range/linear | `v1/v2/v3_config_lock.json` and `_QUANTUM_ANSATZ` (lines 117-124 of `run_matched2000.py`) match | OK |
| `wgan_mlp` (§2.f) | 74 params, single `params_pqc` flat, functional API | `classical.py:58-95` confirms; manual count 5*4+4 + 4*10+10 = 74 ✓ | OK |
| `wgan_cnn` (§2.g) | 73 params, ConvT(1,9,k=6) + Conv(9,1,k=1), functional API | `classical.py:98-141` confirms; manual count (1*9*6+9)+(9*1+1) = 73 ✓ | OK |
| `wgan_lstm` (§2.h) | 78 params, hand-rolled LSTM cell (NOT nn.LSTM) | `classical.py:144-212` confirms gate order i,f,g,o; manual count 48 + 30 = 78 ✓ | OK |
| `vae` (§2.i) | 562 params, ELBO objective | `nonadversarial.py:52-117` confirms; manual count 176+68+68+80+170 = 562 ✓ | **MEDIUM** — see below |
| `ar` (§2.j) | 3 params, order p=2, lstsq closed-form | `nonadversarial.py:120-184` confirms; `_train_ar` in `run_baselines.py:380-409` uses `ARBaseline(p=2)`; `count_params() = p+1 = 3` ✓ | OK |
| `shared_critic` (§2.k) | 250881 params, Conv1d x3 + Linear x2, .double() | `critic.py:19-77` confirms; manual count 704+82048+163968+4128+33 = 250881 ✓ | OK |

**MEDIUM-1: VAE reconstruction loss is MSE, not log-likelihood.**
- Methods doc §2.i renders `L_ELBO = E_q[log p(x|z)] - D_KL(q || p)` (the canonical ELBO).
- Actual code (`run_baselines.py:315`): `recon = torch.nn.functional.mse_loss(x_hat, x)`.
- MSE corresponds to a Gaussian observation model with unit variance, up to a constant scale (`recon ∝ -log p(x|z)` only if `p(x|z) = N(x_hat, I)` and a normalization constant is dropped, AND the average reduction is over the loss elements). Furthermore, `loss = recon + beta * kld` uses raw MSE (averaged per element) added to KLD (averaged per sample); the elementwise vs per-sample reduction does NOT match the standard ELBO formulation exactly. This isn't wrong as a training objective, but the ELBO LaTeX in the methods doc oversimplifies. A referee may ask: "is `recon` summed or averaged across the window dim? Why no `0.5 * (x - x_hat)^2 / sigma^2 + log sigma` factor?"
- **Action:** either (a) replace the LaTeX with the actual code's loss (`L = MSE(x_hat, x) + β·KL(q||p)`, β=1.0) and note "Gaussian-decoder ELBO surrogate with unit variance and elementwise mean reduction" OR (b) change the code to emit a true Gaussian log-likelihood.

### §3 Training — VERIFIED

| Hyperparameter | Methods doc | Code source | Status |
|---|---|---|---|
| Optimizer | Adam | `training.py:296-297` | OK |
| betas | (0.0, 0.9) | `training.py:296-297` | OK |
| lr_critic | 1.8046e-05 | `core/__init__.py:13` | OK |
| lr_generator | 6.9173e-05 | `core/__init__.py:14` | OK |
| n_critic | 9 | `core/__init__.py:11` and `training.py:218` default | OK |
| lambda_gp | 2.16 | `core/__init__.py:12` and `training.py:219` default | OK |
| batch_size | 12 | `core/__init__.py:21` | OK |
| epochs | 2000 | `core/__init__.py:20`, `MATCHED_EPOCHS=2000` in `run_matched2000.py:112` | OK |
| Seeds | [42..46] | `SEED_SET=(42,43,44,45,46)` `run_matched2000.py:109` | OK |
| Early-stopping (reproduction) | OFF | strict_accept rejects `early_stopper` set; `training.py:225` default None | OK |

VAE/AR special-casing (separate Adam lr=1e-3 for VAE, closed-form lstsq for AR) is explicit (§3 paragraph + §2.i/§2.j). Verified in `run_baselines.py:285-358` (VAE) and `run_baselines.py:380-409` (AR).

### §4 Hardware & Software — VERIFIED

Installed versions confirmed by running `./qgan_env/bin/python -c "import pennylane, torch, ..."`:

| Package | Pinned (framework_versions.json) | Actually installed | Status |
|---|---|---|---|
| pennylane | 0.43.0 | 0.43.0 | OK |
| torch | 2.9.0 | 2.9.0 | OK |
| numpy | 2.3.4 | 2.3.4 | OK |
| scipy | 1.16.2 | 1.16.2 | OK |
| matplotlib | 3.10.7 | 3.10.7 | OK |
| PyYAML | 6.0.3 | 6.0.3 | OK |
| Python | 3.11.14 | 3.11.14 | OK |

`dtype_params` / `dtype_samples` split is verified:
- `core/models/classical.py:78` — `torch.randn(self._n, dtype=torch.float32) * _INIT_SCALE` ✓
- `core/models/critic.py:67` — `self.net = self.net.double()` ✓
- `core/training.py:268` — `compute_dtype = torch.float32 if device.type == "mps" else torch.float64` ✓
- `core/training.py:347` — `generated_samples = generated_samples.to(compute_dtype) * 0.1` ✓

### §5 Reproducibility — VERIFIED line numbers
- `training.py:245` — `torch.manual_seed(seed)` ✓ (exact line)
- `training.py:246` — `np.random.seed(seed)` ✓
- `training.py:247` — `random.seed(seed)` ✓
- `training.py:248-249` — `if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)` ✓
- `data_hash = "91e447d4624e25b3"` matches `EXPECTED_DATA_HASH` constant in `run_matched2000.py:106`. The hash is sha256 of the OD numpy array bytes (`run_matched2000.py:242-252`).
- The rerun command template in §5.2 IS sliced from `run_matched2000.py:1-69` (the module docstring ends at line 69; methods doc says "lines 1-80" but the docstring is actually only 1-69 — minor LOW-severity drift, the template content itself is verbatim).

### §6 Contradictions — VERIFIED
- (a) `default_75` vs `iqp_sel_55`: explanation matches `default_75_config_lock.json` (num_layers=4, RX+RY, n_params=75) vs `canonical_config_lock.json` (num_layers=3, RX_only, n_params=55, checkpoint_epoch=1969). Code references `core/__init__.py:17-18` (`NUM_QUBITS=5`, `NUM_LAYERS=4`) which is the `default_75` baseline ✓.
- (b) `dtype_params` vs `dtype_samples`: explanation matches code at the cited line numbers (training.py:259-268, 347; classical.py:78; critic.py:67) ✓.

### §7 Provenance footer — VERIFIED
- Ran `./qgan_env/bin/python verify_number_provenance.py --target docs/methods_full.md` → **PASS — 57 distinct numeric literals all resolve to results/*.json**.

---

## 2. Reviewer reproduction walkthrough

Pretending to be an external referee who clones the repo at HEAD and tries to reproduce.

### Step 1 — Clone repo, check data
```
git clone <repo> qgan && cd qgan
git ls-files data.csv   # tracked ✓
wc -l data.csv          # 778 ✓
```
**Status:** PASS. Data file is in repo.

### Step 2 — Install environment
```
pip install -r requirements.txt
```
**Status:** **BLOCKER-1**. `requirements.txt` uses `>=` constraints:
```
pennylane>=0.32.0
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
```
A reviewer running this in 2026 will get the latest compatible versions (probably pennylane>=0.44, torch>=2.10, etc.), NOT the pinned versions documented in `framework_versions.json` (pennylane=0.43.0, torch=2.9.0, numpy=2.3.4, scipy=1.16.2, matplotlib=3.10.7, PyYAML=6.0.3). The methods doc admits this on line 278-280 ("`revision/requirements.txt` carries the `>=` constraint set; the table above is the exact installed pin"), but a referee has no out-of-band file with `==` pins.

**Fix:** ship a second file `revision/requirements-lock.txt` or `requirements-pinned.txt` with `==` pins generated from `pip freeze` or matching `framework_versions.json`.

### Step 3 — Try to load the headline checkpoint
```
ls best_checkpoint.pt          # 6 MB, exists in working tree but...
git ls-files | grep best_checkpoint   # EMPTY — not tracked!
```
**Status:** **BLOCKER-2**. The headline result (`iqp_sel_55_headline`, §2.a, "frozen-checkpoint paper headline") relies on `best_checkpoint.pt` which is .gitignored (verified via `.gitignore:37` `*.pt`, and via the explicit comments in `run_canonical_headline.py:34` and `run_recover_canonical.py:43`: "`best_checkpoint.pt` and `qgan_env` are .gitignored, so in a git…"). `run_canonical_headline.py:380-383` even asserts a hard sha256 lock on the checkpoint file. **A fresh clone cannot reproduce the headline number** because there is no committed source for the trained 1969-epoch checkpoint.

**Mitigation in code:** `run_recover_canonical.py` exists and has logic to search for the checkpoint, but the file must come from outside the repo. The methods doc doesn't surface this — §2.a says "Checkpoint epoch | 1969 | `canonical_config_lock.json` checkpoint_epoch" without telling the reader the checkpoint file itself is not in version control.

**Fix:** either (a) commit `best_checkpoint.pt` via git-lfs / a release artifact and link from `methods_full.md §2.a`, OR (b) explicitly tell readers in §2.a that the headline can be regenerated via `run_canonical_headline.py` from `best_checkpoint.pt` which must be obtained separately (and add an external location e.g. Zenodo DOI or release asset URL).

### Step 4 — Try to run a fresh 2000-epoch training
```
./qgan_env/bin/python -m revision.run_matched2000 --model V1 --seed 42 --epochs 2000
```
**Status:** PASS (assuming the pinned env from Step 2 was sorted). The `argparse` in `run_matched2000.py:807-840` accepts `--model`, `--seed`, `--epochs`, `--out-root`, `--csv-path`, `--accept` — matches the command in `methods_full.md §5.2` verbatim.

`_MODEL_CHOICES` (line 128) includes all 9 names (`iqp_sel_55_repro`, V1-V3, wgan_*, vae, ar). The methods doc says "iqp_sel_55_repro|V1|V2|V3|wgan_mlp|wgan_cnn|wgan_lstm|vae|ar" — matches.

### Step 5 — Strict accept gate
```
./qgan_env/bin/python -m revision.run_matched2000 --accept --model V1 --seed 42
```
**Status:** PASS *conditional on* the reviewer's data.csv producing the same sha256 hash (=`91e447d4624e25b3`). Since `data.csv` is in git, the file bytes are identical and the hash will match (verified in code path `_compute_data_hash` at `run_matched2000.py:242-252`).

### Step 6 — Reproduce the exact reported numbers
**Status:** **HIGH-1**. The methods doc §5.1 claims "same seed produces bit-identical training trajectories on the same device/dtype path". This is NOT guaranteed:
- `torch.use_deterministic_algorithms(True)` is NOT called anywhere (verified by grep across `revision/`).
- `torch.backends.cudnn.deterministic` / `torch.backends.cudnn.benchmark` not pinned.
- No `PYTHONHASHSEED` setting.
- Adam optimizer accumulates state across `n_critic*epochs = 18000` gradient steps; floating-point reduction order in `nn.Conv1d` autograd is implementation-dependent across torch versions and BLAS backends.

In practice, on the CPU/float64 path most ops *are* numerically deterministic and the same `pip freeze` should give the same numbers. But a referee on a different CPU vendor (Intel vs Apple Silicon) running with a different BLAS (MKL vs Accelerate) MAY see drift in the last few significant digits of EMD/moments. The methods doc should either (a) drop the "bit-identical" claim and replace with "trajectories agree to ~1e-6 EMD", OR (b) actually invoke `torch.use_deterministic_algorithms(True)` and `os.environ['PYTHONHASHSEED'] = '0'` in `train_wgan_gp`.

---

## 3. Determinism claim audit

| Determinism component | Status |
|---|---|
| Python dict iteration order (3.7+ guarantees insertion order) | OK — code does not use `set()` over training-affecting collections |
| `torch.manual_seed(seed)` | SET (training.py:245) |
| `np.random.seed(seed)` | SET (training.py:246) |
| `random.seed(seed)` | SET (training.py:247) |
| `torch.cuda.manual_seed_all` | SET conditionally (training.py:248-249) |
| `torch.use_deterministic_algorithms(True)` | **NOT SET** anywhere in revision/ |
| `torch.backends.cudnn.deterministic = True` | Not set (irrelevant since runs are CPU) |
| `PYTHONHASHSEED` env var | Not set (irrelevant for non-set/dict-key-dependent code) |
| MPS device fallback | **IF** Apple Silicon and `mps.is_available()`, `compute_dtype` flips float64→float32 (training.py:268) — different dtype path, different numerics. The methods doc admits this in §4.1/§6(b) but it is NOT a guaranteed-same-device contract: if a reviewer accidentally runs on MPS instead of CPU, results will silently differ. The `_device_manifest()` hard-assert at `run_matched2000.py:255-274` enforces sample-generation on CPU/float64 but not the *training* device. |

**Severity:** HIGH for the "bit-identical" wording but the practical outcome (~1e-6 EMD agreement on same CPU/BLAS) is fine.

---

## 4. Single-campaign disclosure

**Status:** ADEQUATE. The single-campaign limitation is disclosed in:
- `methods_full.md:46-47`: "The single-campaign limitation means all reported metric variance is over training-seed variation, not over independent experimental campaigns."
- `reviewer_response.md:40`: "single-variable, single-campaign proof-of-concept"
- `paper_blocks_framing.md:125`: "empirical validation on a single real-world photobioreactor cultivation campaign"

I did NOT find any claim in `methods_full.md` or `reviewer_response.md` that accidentally implies data variance is being captured.

---

## 5. Classical architecture extraction — re-verification

Manual parameter counts (independent of `classical_architectures.json`):

| Model | Manual count | Reported total_params | Match |
|---|---|---|---|
| wgan_mlp | 5*4+4 + 4*10+10 = 74 | 74 | ✓ |
| wgan_cnn | (1*9*6+9) + (9*1*1+1) = 73 | 73 | ✓ |
| wgan_lstm | (8*2)+(8*2)+8+8 + (10*2+10) = 78 | 78 | ✓ |
| vae | 176+68+68+80+170 = 562 | 562 | ✓ |
| ar | p + 1 = 2 + 1 = 3 | 3 | ✓ |
| shared_critic | 704+82048+163968+4128+33 = 250881 | 250881 | ✓ |

Latent/hidden/window dimensions in JSON (latent_dim=4, hidden_dim=16, window=10) match the constants in `core/models/nonadversarial.py:63-65` (`LATENT_DIM=4`, `HIDDEN=16`, `WINDOW=10`).

AR(p) order: JSON says `order_p=2`; code (`run_baselines.py:387`) instantiates `ARBaseline(p=2)`. Match.

---

## 6. Parameter-matched-comparison fairness

**Status:** **MEDIUM-2**. The headline numerical claim is "quantum at 55 params matches classical WGAN at 73-78 params", but:

- The **shared critic** (250881 params, `methods_full.md §2.k`) is excluded from the count for BOTH sides of the comparison. The generator-only matching is what's compared.
- The **VAE** at 562 params is 8-10x heavier than the quantum generator. Methods doc admits this ("NOT parameter-matched to the quantum generator (D-10-03)" — `nonadversarial.py:11-13`) but the comparison table still lists it alongside the matched WGAN baselines.

**Fairness assessment:**
- Quantum-vs-classical generator parameter matching IS fair on the generator side because both share the same critic, optimizer, training budget, data, and seeds. The shared-critic disclosure exists in `methods_full.md §2.k:211-219`.
- VAE/AR are NOT parameter-matched and are labeled as such (`family = non-adversarial`). The methods doc could be clearer about this distinction in §2.i (the §2.i header just says "non-adversarial baseline" without explicitly noting "NOT param-matched to quantum"). The `paper_blocks_framing.md` does use the "parameter-matched comparison" framing carefully when comparing quantum vs classical-WGAN only.

**Recommendation:** add a single sentence to `methods_full.md §2.i`: "Unlike the WGAN-GP baselines, the VAE is intentionally NOT parameter-matched to the quantum generator (562 vs 55) — it is included as the 'smallest deep VAE that trains stably' on length-10 windows; the comparison is reported as a non-adversarial reference, not as a parameter-controlled head-to-head."

---

## 7. Loss function citations

**Status:** **LOW-1**. `methods_full.md` renders the WGAN-GP, ELBO, and AR(p) equations verbatim from the JSON corpus but does NOT cite the original papers (Gulrajani et al. 2017 for WGAN-GP, Kingma & Welling 2013 for VAE/ELBO, Hamilton or Box-Jenkins for AR(p)). The methods doc is a numbers-provenance document, so this is intentional and OK.

The companion `docs/paper_blocks_refs_methods.md` cites `\cite{wang2018esrganenhancedsuperresolutiongenerative, akkem2024comprehensive}` and `\cite{yoon2019TimeGAN}` for GAN/VAE generally but does NOT cite Gulrajani 2017 for WGAN-GP or Kingma & Welling 2013 for ELBO.

The actual paper LaTeX (`supp_material.tex` and `main (4) copy.tex`) DOES cite `\cite{gulrajani2017improved}`, `\cite{arjovsky17a}`, `\cite{Arjovsky2017}` (lines 78, 83, 85, 106, 405, 420 in supp_material.tex; line 131 in main). So WGAN-GP citation is in the paper. **Missing: Kingma & Welling 2013 (VAE), and a foundational AR(p) reference (Box-Jenkins 1970 / Hamilton 1994).**

**Recommendation:** add `\cite{kingma2013auto}` to wherever VAE is introduced in the paper, and `\cite{hamilton1994time}` or `\cite{box2015time}` for AR(p). (These are LOW — they live in the LaTeX paper, not in methods_full.md.)

---

## 8. Reproduction walkthrough — final verdict

| Step | Can a referee complete? |
|---|---|
| Clone repo | YES |
| Read methods doc | YES (provenance gate passes) |
| Reproduce dataset (data.csv tracked) | YES |
| Get pinned env | **NO — BLOCKER-1** (only `>=` constraints in requirements.txt) |
| Reproduce headline (`iqp_sel_55_headline`) | **NO — BLOCKER-2** (best_checkpoint.pt is .gitignored) |
| Reproduce matched-2000 runs from scratch | YES (with workaround for BLOCKER-1) |
| Pass `--accept` strict gate on their reproduction | YES (data_hash will match because data.csv is in git) |
| Reproduce reported numbers bit-identically | NO — only to ~1e-6 EMD on same CPU/BLAS (HIGH-1) |

**Final assessment:** An external researcher CAN reproduce the matched-2000 sweep results (the V1/V2/V3/wgan_mlp/wgan_cnn/wgan_lstm/vae/ar/iqp_sel_55_repro 45-run matrix) from scratch with this codebase, **PROVIDED** they manually install the exact pinned package versions documented in `framework_versions.json`. They CANNOT reproduce the headline frozen-checkpoint result (`iqp_sel_55_headline` at epoch 1969) without obtaining `best_checkpoint.pt` from an out-of-band source.

---

## Summary table of findings

| # | Severity | Finding | Location |
|---|---|---|---|
| 1 | BLOCKER | `requirements.txt` uses `>=` not `==` pins; reviewer cannot recreate the pinned environment from the repo alone | `/Users/shawngibford/dev/phd/qGAN/requirements.txt` vs `results/framework_versions.json` |
| 2 | BLOCKER | `best_checkpoint.pt` (the headline frozen checkpoint) is .gitignored and not in version control; methods_full.md §2.a does not surface this | `.gitignore:37`, `run_canonical_headline.py:34`, `run_recover_canonical.py:43` |
| 3 | HIGH | "bit-identical training trajectories" claim is too strong; `torch.use_deterministic_algorithms(True)` is not set; results agree only to ~1e-6 on same CPU/BLAS | `docs/methods_full.md:316-318` vs absence of `use_deterministic_algorithms` in `core/training.py` |
| 4 | MEDIUM | VAE training uses raw `mse_loss` for reconstruction but methods doc renders the canonical Gaussian-log-likelihood ELBO LaTeX; the equation oversimplifies the actual code | `methods_full.md §2.i` LaTeX vs `run_baselines.py:315` |
| 5 | MEDIUM | VAE param count (562) is 8-10x the quantum generator (55); `methods_full.md §2.i` does not explicitly say "NOT parameter-matched" the way `nonadversarial.py:11-13` does | `methods_full.md §2.i:171-191` |
| 6 | LOW | Methods doc §5.2 says the rerun template is sliced from "lines 1-80" of `run_matched2000.py`, but the actual module docstring is lines 1-69 | `methods_full.md:322` vs `run_matched2000.py:1-69` |
| 7 | LOW | `paper_blocks_refs_methods.md` lacks an explicit Kingma & Welling 2013 (VAE) citation and a Box-Jenkins / Hamilton AR(p) foundational reference, although Gulrajani 2017 IS cited in the LaTeX paper | `docs/paper_blocks_refs_methods.md` |

---

## Recommended pre-submission fixes (priority order)

1. **(BLOCKER-1 fix)** Ship `requirements-pinned.txt` (or rename existing requirements.txt) with `==` pins matching `framework_versions.json`. Reference it from `methods_full.md §4.1` so reviewers see it.
2. **(BLOCKER-2 fix)** Either (a) commit `best_checkpoint.pt` via git-lfs / release tarball, OR (b) add an explicit "**Frozen checkpoint location**" line in `methods_full.md §2.a` pointing reviewers to a Zenodo/Figshare DOI for the checkpoint binary.
3. **(HIGH-1 fix)** Soften the §5.1 "bit-identical" wording to "trajectories agree to ~1e-6 EMD on the same CPU+BLAS+pinned-pip-freeze stack; full bit-determinism would require `torch.use_deterministic_algorithms(True)` which is not set in this run" — OR add the `torch.use_deterministic_algorithms(True, warn_only=True)` call to `train_wgan_gp`.
4. **(MEDIUM-1 fix)** Either reconcile the §2.i ELBO LaTeX with the MSE-based reconstruction loss actually trained, or change the code to compute a true Gaussian log-likelihood.
5. **(MEDIUM-2 fix)** Add one sentence to §2.i explaining the VAE is not parameter-matched (562 vs 55).
6. **(LOW)** Correct the §5.2 "lines 1-80" → "lines 1-69" line range.
7. **(LOW)** Add Kingma & Welling and Box-Jenkins / Hamilton citations to the paper LaTeX.

---

## Files cited
- `/Users/shawngibford/dev/phd/qGAN/docs/methods_full.md`
- `/Users/shawngibford/dev/phd/qGAN/docs/reviewer_response.md`
- `/Users/shawngibford/dev/phd/qGAN/docs/paper_blocks_framing.md`
- `/Users/shawngibford/dev/phd/qGAN/docs/paper_blocks_refs_methods.md`
- `/Users/shawngibford/dev/phd/qGAN/core/data.py`
- `/Users/shawngibford/dev/phd/qGAN/core/training.py`
- `/Users/shawngibford/dev/phd/qGAN/core/models/classical.py`
- `/Users/shawngibford/dev/phd/qGAN/core/models/nonadversarial.py`
- `/Users/shawngibford/dev/phd/qGAN/core/models/critic.py`
- `/Users/shawngibford/dev/phd/qGAN/core/__init__.py`
- `/Users/shawngibford/dev/phd/qGAN/run_matched2000.py`
- `/Users/shawngibford/dev/phd/qGAN/run_baselines.py`
- `/Users/shawngibford/dev/phd/qGAN/run_canonical_headline.py`
- `/Users/shawngibford/dev/phd/qGAN/run_methods_full.py`
- `/Users/shawngibford/dev/phd/qGAN/results/methods_full.json`
- `/Users/shawngibford/dev/phd/qGAN/results/model_info.json`
- `/Users/shawngibford/dev/phd/qGAN/results/framework_versions.json`
- `/Users/shawngibford/dev/phd/qGAN/results/classical_architectures.json`
- `/Users/shawngibford/dev/phd/qGAN/requirements.txt`
- `/Users/shawngibford/dev/phd/qGAN/.gitignore`
