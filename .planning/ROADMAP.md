# Roadmap: qGAN Post-HPO Improvements

## Milestones

- ✅ **v1.0 qGAN Code Review Remediation** -- Phases 1-3 (shipped 2026-03-07)
- **v1.1 Post-HPO Improvements** -- Phases 4-7 (in progress)

## Phases

<details>
<summary>v1.0 qGAN Code Review Remediation (Phases 1-3) -- SHIPPED 2026-03-07</summary>

- [x] Phase 1: Foundation and Correctness Infrastructure (3/3 plans) -- completed 2026-03-01
- [x] Phase 2: WGAN-GP Correctness and Quantum Circuit Redesign (4/4 plans) -- completed 2026-03-05
- [x] Phase 3: Post-Processing Consistency and Cleanup (2/2 plans) -- completed 2026-03-07

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

### v1.1 Post-HPO Improvements

**Milestone Goal:** Fix regressions from conditioning work and add high-impact improvements to address variance collapse (fake std 0.0104 vs real 0.0218)

**Phase Numbering:**
- Integer phases (4, 5, 6, 7): Planned milestone work
- Decimal phases (4.1, 5.1): Urgent insertions (marked with INSERTED)

- [ ] **Phase 4: Code Regression Fixes** - Restore correct noise range, fix par_zeros eval bug, eliminate mu/sigma shadowing
- [ ] **Phase 5: Backprop and Broadcasting** - Switch diff_method to backprop and restore batched QNode calls for ~12x speedup
- [ ] **Phase 6: Spectral Loss** - Add differentiable PSD loss to give generator explicit frequency-domain gradient signal
- [ ] **Phase 7: Conditioning Verification** - Verify PAR_LIGHT conditioning modulates output and make dropout configurable

## Phase Details

### Phase 4: Code Regression Fixes
**Goal**: Training code produces correct results with trustworthy metrics -- noise range matches circuit design, evaluation reflects conditioned generation, and plotting cells are safe for re-execution
**Depends on**: Phase 3 (v1.0 complete)
**Requirements**: REG-01, REG-04, REG-05
**Success Criteria** (what must be TRUE):
  1. Running a training epoch uses noise sampled from [0, 4pi] in all three locations (critic training, generator training, evaluation) -- verified by inspecting generated noise tensor ranges
  2. Evaluation cell generates fake samples using real PAR_LIGHT values from the dataset, not zeros -- EMD and moment statistics reflect conditioned generation
  3. Plotting cells can be re-executed without mu/sigma variable shadowing corrupting distribution overlays
  4. A 200-epoch validation run produces EMD within 2x of HPO baseline (best_emd=0.001137) -- confirming HPO hyperparameters transfer to corrected code
**Plans:** 2 plans

Plans:
- [ ] 04-01-PLAN.md -- Apply all code regression fixes (noise range, par_zeros, mu/sigma, ACF removal, HPO hyperparameters)
- [ ] 04-02-PLAN.md -- Execute 200-epoch validation run and capture baseline metrics to JSON

### Phase 5: Backprop and Broadcasting
**Goal**: Training runs ~12x faster through batched quantum circuit execution, unblocking efficient iteration for spectral loss tuning and conditioning experiments
**Depends on**: Phase 4 (noise range must be correct before broadcasting it)
**Requirements**: REG-02, REG-03
**Success Criteria** (what must be TRUE):
  1. QNode uses diff_method='backprop' on default.qubit with shots=None
  2. All three training loops (critic, generator, evaluation) use a single batched QNode call per step instead of per-sample Python loops
  3. Batched QNode output matches sequential per-sample output within 1e-6 tolerance -- verified by element-wise comparison on a test batch
  4. Epoch wall-clock time is less than 30% of pre-broadcasting time (measured over 10 consecutive epochs)
**Plans:** 2 plans

Plans:
- [ ] 05-01-PLAN.md -- Switch QNode to backprop and convert all four per-sample loops to batched calls
- [ ] 05-02-PLAN.md -- Validate equivalence, reproducibility, and SC4 timing gate (one-time cells, then delete)

### Phase 6: Spectral Loss
**Goal**: Generator receives explicit gradient signal penalizing wrong frequency content, directly addressing the root cause of variance collapse where the generator learns mean drift but not volatility structure
**Depends on**: Phase 5 (broadcasting speedup enables practical PSD loss tuning)
**Requirements**: SPEC-01, SPEC-02, SPEC-03
**Success Criteria** (what must be TRUE):
  1. Generator loss includes a log-PSD MSE term computed via torch.fft.rfft that is fully differentiable (gradients flow back through PSD computation to generator parameters)
  2. lambda_psd is exposed as a configurable hyperparameter with a sensible default, and training logs report PSD loss as a separate component alongside Wasserstein and ACF losses
  3. PSD loss is computed on the same batch of real/fake windows used for the WGAN loss (no separate forward pass or data sampling)
  4. After training with PSD loss enabled, fake sample standard deviation trends closer to real std (0.0218) compared to the Phase 5 baseline without PSD loss
  5. No single loss component (Wasserstein, ACF, PSD) exceeds 10x another at training equilibrium -- loss balance is maintained
**Plans**: TBD

### Phase 7: Conditioning Verification
**Goal**: Empirical evidence determines whether PAR_LIGHT conditioning actually modulates generator output -- a thesis-critical question that has never been honestly measured due to the par_zeros bug fixed in Phase 4
**Depends on**: Phase 6 (requires a well-trained model with spectral loss for meaningful conditioning measurement)
**Requirements**: COND-01, COND-02, COND-03
**Success Criteria** (what must be TRUE):
  1. An intervention test cell generates samples at PAR_LIGHT=0 vs PAR_LIGHT=1 and reports a KS test statistic with p-value -- providing a binary answer on whether conditioning is effective (p < 0.05)
  2. A sweep test cell generates samples across PAR_LIGHT grid [0, 0.2, 0.4, 0.6, 0.8, 1.0] and displays summary statistics (mean, std, kurtosis) per level -- showing whether output varies monotonically or systematically with PAR_LIGHT
  3. Critic dropout rate is exposed as a configurable hyperparameter (default 0.2) that can be adjusted without code changes
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 4 -> 5 -> 6 -> 7

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation and Correctness Infrastructure | v1.0 | 3/3 | Complete | 2026-03-01 |
| 2. WGAN-GP Correctness and Quantum Circuit Redesign | v1.0 | 4/4 | Complete | 2026-03-05 |
| 3. Post-Processing Consistency and Cleanup | v1.0 | 2/2 | Complete | 2026-03-07 |
| 4. Code Regression Fixes | v1.1 | 2/2 | Complete | 2026-03-13 |
| 5. Backprop and Broadcasting | v1.1 | 0/2 | Planning complete | - |
| 6. Spectral Loss | v1.1 | 0/? | Not started | - |
| 7. Conditioning Verification | v1.1 | 0/? | Not started | - |
