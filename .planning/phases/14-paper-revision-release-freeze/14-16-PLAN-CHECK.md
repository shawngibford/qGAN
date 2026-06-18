# 14-16-PLAN-CHECK — Pre-execution Verification

## Verdict: **NEEDS REVISION**

The plan is structurally sound on the two CRITICAL bug fixes (T1 R3-CR-2 + T2 R3-CR-1) and the byte-freeze invariants (D-14-22, D-14-16, D-14-18) are preserved. However, the plan ships a manuscript with a known sister-bug intact, citations to non-existent text, and a parametric-efficiency equivalence table whose param counts contradict `model_info.json`. Three of these are blockers; the rest are warnings.

## Per-outcome trace (1–6)

| # | Outcome | Plan coverage | Status |
|---|---------|---------------|--------|
| 1 | R3-CR-2 fixed at root cause; standardized vs standardized on log-return EMD; aggregates match Agent 2 anchors | T1 `<action>` Step B + `<verify>` directional anchors (AR 0.003, V1 0.0145, VAE 0.0163; quantum-beats-WGAN); `must_haves.truths` item 6 + 7; `key_links` 1 | **COVERED** for the `run_matched2000_dualscale.py` site |
| 2 | R3-CR-1 fixed at root cause; shared-edges; `fake_in_range_mass` disclosed; ranking inversion in `distribution_emd.json` v2 | T2 `<action>` Steps A-H + `<verify>` schema bump + ranking-inversion assertion; `must_haves.truths` items 8, 9, 10; `key_links` 2 | **COVERED** for `compute_histogram_density_emd` (`:94-141`) only |
| 3 | Strong-claim parametric-efficiency-equivalence framing in `reviewer_response.md` R1-M1 | T3 Step A rewrites R1-M1 with the verbatim Welch claim; Step B appends `## Parametric-efficiency equivalence` subsection; `must_haves.truths` item 11; `key_links` 3 | **PARTIALLY COVERED** — text is specified but the cited "follow-up work" hedge does not exist in the file (see Issue 4) and the param-count table is wrong (see Issue 3) |
| 4 | All 10 paper-facing docs PASS v2.1 gate; no gate edit | T4 Step F enumerates the 10-doc loop; T4 `<verify>` runs the gate; T3 Step H runs the gate on the three updated docs; `must_haves.truths` item 14; `key_links` 6 | **COVERED structurally** but the gate is content-agnostic and will PASS on coincidental ε-matches (see Issue 6 — soft warning) |
| 5 | `core/` byte-untouched (D-14-22) across all 5 tasks | Every task `<verify>` asserts `[ -z "$(git diff --stat core/)" ]`; `must_haves.truths` item 3; `<threat_model>` T-14-82 + T-14-94 | **COVERED** |
| 6 | Only 14-07 remains outstanding for Phase 14 after 14-16 merges | T5 Step C + `must_haves.truths` items 16, 17; `<acceptance_criteria>` for T5 | **COVERED** |

## R3-HI-1 verdict: **(b) WRONG PUNT** — must fold sibling fix into T2

### Evidence from `peer-review-r3/code-review-r3.md` §H3 (lines 262-339)

R3-HI-1 is **a single finding** that explicitly names **two files** as edit sites:

```
Files:
- run_matched2000_dualscale.py:368-372  (the _log_return_rows emit)
- run_distribution_emd.py:144-153 (_real_references) +
  run_distribution_emd.py:156-169 (_fake_log_return_flat)
```

Agent 5's recommended fix at lines 320-330 explicitly says:

> change `real_log_delta` to `norm_log_delta` in the real reference so both sides are on the standardized scale. The LATTER is a smaller code change (single-line edit to `build_real_references`); the FORMER is more semantically correct. **Same fix needed in `run_distribution_emd.py:_real_references` / `_fake_log_return_flat`.**

Agent 5's recommended-action section (line 823-829) bundles both into pre-tag hot-fix #1:

> **R3-HI-1 (scale mismatch fix):** edit `run_matched2000_dualscale.py:build_real_references` AND `run_distribution_emd.py:_real_references` to use `norm_log_delta`...

### Why this is a wrong-to-punt (not right-to-punt)

1. **Same file already being edited.** T2 already opens `scripts/run_distribution_emd.py`. The sibling-call sites are at `:144-169`, ~30 lines below the `:94-141` block T2 replaces. The marginal cost is on the order of 20-40 lines of edit plus an analogous `fake_in_range_mass`-style disclosure stat for the log-return-scale rows.

2. **Manuscript ships with one corrected metric variant and one still-broken metric variant on the same JSON.** Post-T2, `distribution_emd.json` v2 carries:
   - OD-scale rows: correctly reformulated (R3-CR-1 fixed)
   - log-return-scale rows: STILL on the broken scale-mismatched comparison (R3-HI-1/R3-CR-2 inherited)
   
   `statistical-honesty-r3.md` §3d explicitly uses HD-LR-EMD numbers to support the framing that "quantum looks worst on this metric". If those numbers are inherited-broken, the §3d framing the plan cites is itself unstable.

3. **The plan's `must_haves.truths` and T2 docstring both EXPLICITLY acknowledge** the bug remains: "the log-return-scale rows in `distribution_emd.json` v2 inherit the same scale mismatch as the pre-T1 dualscale driver but are NOT corrected by this plan — the OD-scale rows are the primary deliverable of v2." This is an admission that the deliverable ships with a known broken column.

4. **SYNTHESIS.md is ambiguous (does not explicitly punt).** SYNTHESIS.md Path 1 says "Fix R3-CR-2 (log-return scale mismatch in `run_matched2000_dualscale.py`) — single-line edit using `norm_log_delta`." It does NOT name the sister site in `run_distribution_emd.py:_real_references`. The plan reads this silence as a punt, but the underlying source (code-review-r3.md, the more granular agent report) names BOTH sites under the SAME finding ID (R3-HI-1) and asks for them BOTH to be fixed in one pre-tag hot-fix. The synthesis is the executive summary; the code-review is the engineering ground-truth, and they conflict.

### Recommended T2 expansion text

Add the following step to T2 between current Steps E and F:

> **Step E2 — Apply analogous standardization fix to log-return-scale rows in distribution_emd:**
>
> Locate `_real_references` at `:144-153` and `_fake_log_return_flat` at `:156-169`. Apply the same standardization fix as T1 to the log-return-scale path: read per-seed `mu, sigma` from `inverse_kwargs.npz` and either (a) replace `real_log_delta` with `norm_log_delta = (real_log_delta - mu) / sigma`, or (b) un-standardize the fake side via `r_norm * sigma + mu`. Choose option (a) to mirror T1.
>
> Update `_model_seed_rows` log-return-scale call site to consume the standardized real reference. The OD-scale path is byte-untouched.
>
> Anticipated marginal effort: ~20 lines of edit + an analogous `fake_in_range_mass`-style disclosure stat for the log-return-scale rows (the WGAN-CNN 94%-out-of-range disclosure must also surface in the LR-scale aggregate, where the original §3d "quantum worst" framing comes from).

This expands T2 from one fix to two fixes in the same file. Bump T2 `<acceptance_criteria>` with: "log-return-scale rows in `distribution_emd.json` v2 are computed against standardized real reference; LR-scale `fake_in_range_mass_mean` populated; the `*_LR_no_longer_inherited_R3-HI-1` disclosure landed in module docstring."

Also delete the `truths` item and T2 Step E docstring extension claims that R3-HI-1 is "outside the locked Path 1 scope" — that claim is the source of the wrong-punt.

---

## Specific revisions required (line-edit-level)

### Issue 1 (BLOCKER): R3-HI-1 sister fix must be folded into T2

**Plan section:** T2 `<read_first>` (line ~662), T2 `<action>` Step E docstring extension (line ~826-832), `must_haves.truths` items 9 + 10 + 19, `<threat_model>` T-14-81 + T-14-93.

**Quoted truth to remove (or rewrite):**
```
"Files explicitly NOT modified by this plan: core/ (D-14-22 byte-freeze...); ... 
 the analogous scale mismatch in `run_distribution_emd.py:_real_references` + 
 `_fake_log_return_flat` at `:144-169` (R3-HI-1 per `code-review-r3.md`) is OUTSIDE 
 the user-locked Path 1 scope of Plan 14-16 and is a known follow-up..."
```

**Change to:** delete the R3-HI-1 punt entirely and add a coverage-truth statement: "R3-HI-1 sister site in `run_distribution_emd.py:_real_references` + `_fake_log_return_flat` at `:144-169` ALSO FIXED by T2 with the same standardization recipe; both LR-scale call sites in both files now use `norm_log_delta`."

**Add T2 Step E2** (text in the R3-HI-1 verdict section above) and bump T2 acceptance criteria accordingly.

---

### Issue 2 (BLOCKER): Parametric-efficiency equivalence table param counts are wrong

**Plan section:** T3 `<action>` Step B, the equivalence table at lines ~969-975.

**Quoted block to fix:**
```
| classical baseline | generator parameter count | Welch t-test p | Cohen's d |
| wgan_mlp | 74 | 0.688 | +0.26 |
| wgan_lstm | 562 | 0.836 | -0.14 |
| wgan_cnn | ~10^5 | 0.365 | -0.64 |
| vae | ~10^4 | 0.664 | +0.29 |
| ar | 3 (closed-form) | 0.627 | -0.32 |
```

**Ground truth from `results/model_info.json`:**
- wgan_mlp: parameter_count = 74 ✓
- wgan_cnn: parameter_count = 73  (NOT ~10^5)
- wgan_lstm: parameter_count = 78  (NOT 562)
- vae: parameter_count = 562  (NOT ~10^4)

The plan inverted wgan_lstm ↔ vae and invented `~10^4` and `~10^5` values. The strong-claim text says "10^4-10^5 classical generator parameters" but no classical generator has 10^4 params; the closest is vae=562.

**The "10^4-10^5" framing likely refers to the adversarial budget (generator + shared 250,881-param critic) from `total_adversarial_param_budget.json`, which is ~250,936 (10^5) for every adversarial model. If so, the framing must say "adversarial parameter budget" not "classical generator parameters" — and 10^4 still has no source.**

**Change to (line-edit):**
1. In R1-M1 rewrite (line ~938) and in `must_haves.truths` item 11: replace "10^4-10^5 classical generator parameters" with **either** "74-562 classical generator parameters (or 10^5 including the shared critic)" **or** the specific honest framing from `statistical-honesty-r3.md` §7-Option A line 288: "55 quantum parameters vs ~10⁴–10⁵ classical parameters at comparable distributional fidelity" — but **only if the JSONs actually support 10^4-10^5**; verify against `total_adversarial_param_budget.json` (`250881 + generator` ≈ 250,956 for all adversarial models, single power of ten, NOT 10^4-10^5).
2. In the table (line 969-975): fix wgan_cnn=73, wgan_lstm=78, vae=562. Drop "~10^4" and "~10^5" placeholders.

If kept as adversarial-budget framing, also add `total_adversarial_param_budget.json` to the cited artifact list at line ~991.

---

### Issue 3 (BLOCKER): Plan repeatedly cites a "follow-up work" hedge that does not exist in `docs/reviewer_response.md`

**Plan section:** `<objective>` line 187-188, T3 `<action>` Step A line 927-928, `<contract_changes>` line 290-292, `<success_criteria>` line 1657-1659.

**Quoted text:**
```
"The current R1-M1 response hedges with 'this is being addressed in follow-up work'."
"replace the existing 'this is being addressed in follow-up work' hedge..."
```

**Verification:** `grep -i 'follow-up work\|being addressed' docs/reviewer_response.md` returns **zero matches**. The actual R1-M1 row (line 36 of the file) is a table-cell containing: *"Added matched-parameter classical WGAN-GP (MLP/CNN/LSTM critics) and a non-adversarial VAE + AR baseline, all at matched 2000-epoch budget, identical critic/optimizer/seed set; parameter-count-controlled comparison table"*. There is no hedge; there is no follow-up-work language. The current row simply doesn't make an equivalence claim — but it doesn't deny one either.

**Consequence:** T3 Step A's "drop the prior 'this is being addressed in follow-up work' hedge" cannot be executed because that hedge is not in the file. Executor will either (a) fail to find the string and abort, or (b) interpret loosely and rewrite the table row, which is currently in markdown-table form, into a paragraph — without specifying how the surrounding table structure should adapt.

**Change to (line-edit):**
1. Replace all four "follow-up work hedge" mentions with accurate description: "The current R1-M1 row makes no equivalence claim and reads as a generic 'we added baselines' summary; T3 expands this to include the strong-claim Welch-grounded equivalence paragraph."
2. T3 Step A must specify: "The R1-M1 row remains a one-line table cell preserving the existing 'Change made' text; the new strong-claim paragraph is APPENDED to the same row as a follow-on paragraph (or, alternatively, a NEW `## Reviewer 1 — Major Issues — supplementary (post-r3 corrected metrics)` H2 section is inserted between line 41 and line 42 of the file)."
3. Choose ONE structural strategy (paragraph-in-cell vs new section) explicitly. Do not leave executor to choose.

---

### Issue 4 (WARNING): Cohen's d range "-3 to -5" understates wgan_cnn

**Plan section:** R1-M1 strong-claim wording (line 941), `must_haves.truths` item 17 (line 48), `<success_criteria>` line 1662.

**Quoted:** `"d ≈ -3 to -5"`

**Ground truth from `statistical-honesty-r3.md` §3b:**
- iqp vs wgan_mlp: d = -5.22
- iqp vs wgan_cnn: d = **-2.63**  ← below -3
- iqp vs wgan_lstm: d = -2.97  ← also marginal vs -3
- §7-Option A line 288 phrases this as "Cohen d ≤ −2.6"

**Change to:** replace `d ≈ -3 to -5` with `d ≤ -2.6` (matches §7-Option A verbatim) OR `d ≈ -2.6 to -5.2` (matches the empirical range). The current `-3 to -5` excludes wgan_cnn (d=-2.63) from the cited range, which is the very pair that anchors the `p ≤ 0.014` ceiling.

---

### Issue 5 (WARNING): Welch p-values + Cohen's d are not stored in any JSON the v2.1 walker walks

**Plan section:** `<threat_model>` T-14-86, `must_haves.truths` item 14 + 17, `key_links` 6.

**Issue:** The strong-claim numbers (`p > 0.36`, `|d| ≤ 0.65`, `p ≤ 0.014`, `d ≈ -2.6 to -5`, `n=5`) live ONLY in `peer-review-r3/statistical-honesty-r3.md`, which is NOT under `results/*.json`. The v2.1 walker only crawls `results/*.json`. Therefore these literals will resolve **via ε-neighborhood coincidental matches** to OTHER, unrelated JSON values (e.g., `0.36` matches `0.36042169522527` in `matched2000_dualscale.json#rows[*].value` — an EMD value, not a Welch p-value).

The gate PASSES but the resolution is **semantically meaningless** — a defense-in-depth gap that the gate itself cannot detect by design.

**Recommended remediation (warning, not blocker because the gate technically passes):** add a T3 sub-task to emit a new JSON `results/welch_tests.json` mirroring `statistical-honesty-r3.md` §3a + §3b tables (per-pair Welch t, p, Cohen's d, MWU p, n=5). This makes the strong-claim literals genuinely resolvable to a JSON source rather than coincidentally-matched to unrelated values. The schema is straightforward (per-pair records keyed by `{quantum: <m>, classical: <m>, metric: 'OD-EMD' | 'LR-EMD'}`).

If skipped: when a reviewer asks "where did 0.014 come from?", the answer is `statistical-honesty-r3.md` which is in `.planning/`, not in the audited corpus — a provenance audit gap.

---

### Issue 6 (WARNING): T3 mixes 3 separate file rewrites (1 emitter + 3 docs) in one task

**Plan section:** T3 `<files>` line 906.

T3 modifies: `scripts/run_model_info.py`, `docs/reviewer_response.md`, `docs/methods_full.md`, `docs/reconciliation_note.md` — that's 1 Python emitter plus 3 paper-facing docs (4 files total). 14-13/14/15 historically split this kind of work across 2 tasks. The plan justifies the consolidation under "single coherent edit" but T3 spans:
- R1-M1 rewrite + new subsection in `reviewer_response.md` (manual edits)
- Two new paragraphs in `methods_full.md` (manual edits)
- C-3 sentence extension in `reconciliation_note.md` (emitted via `scripts/run_model_info.py`?)
- `scripts/run_model_info.py` table-emission update

The atomicity contract is at risk: if `scripts/run_model_info.py` edit lands but the manual reviewer_response.md edit fails or vice versa, the commit is incoherent. Compare with 14-15 which split similar work across separate tasks.

**Recommendation (warning):** split T3 into T3a (`scripts/run_model_info.py` + `reconciliation_note.md` regeneration via emit) and T3b (manual edits to `reviewer_response.md` + `methods_full.md`). Each gets its own atomic commit. Total task count goes from 5 to 6 — still within the 2-3-target threshold per dimension when measured per-task-scope.

This is a warning, not blocker, because the plan explicitly enumerates all edits step-by-step. But the failure mode (incoherent mid-task state) is real and the 14-13/14/15 precedent splits this.

---

### Issue 7 (WARNING): T1's `--target` re-run of v2.1 gate is in T3+T4, not in T1

**Plan section:** T1 `<verify>` block.

T1's verify gate validates the corrected JSON has expected aggregates but does NOT run the v2.1 gate to confirm the OLD reviewer_response.md / reconciliation_note.md literals (which referenced uncorrected aggregates) still resolve. If T1 changes a value that was previously cited by an existing doc, the gate could fail mid-stream until T3 lands.

**Mitigation:** T1 should additionally invoke `verify_number_provenance.py --target reconciliation_note.md` before commit to confirm the OD-byte-identical invariant didn't break any existing OD-EMD citation. (LR-EMD literals are expected to become unresolvable until T3 updates the docs — this is the inter-task incoherence cost of splitting T1 from T3.)

**Recommended fix:** T1 `<verify>` adds a gated check: existing OD-EMD-citing docs (e.g., `reconciliation_note.md` OD headline table from 14-13) still gate-pass; LR-EMD-citing docs are EXPECTED to fail until T3 (record this as a known intermediate state).

---

## Summary

3 BLOCKERS (R3-HI-1 wrong-punt, param-count table errors, non-existent "follow-up work hedge" citation) + 4 WARNINGS. The R3-HI-1 punt is the most consequential: shipping a manuscript whose `distribution_emd.json` v2 carries one corrected metric variant alongside one inherited-broken variant in the same file is exactly the kind of half-fix a reviewer who reads `peer-review-r3/code-review-r3.md` will spot — and the bug is named under the same finding ID Agent 5 already wrote up for fixing. The marginal cost (~20 lines + analogous disclosure stat) is small enough that punting cannot be justified on scope grounds.

Revise the plan to (1) fold the R3-HI-1 sister fix into T2 Step E2, (2) correct the param-count table to match `model_info.json` and pick honest framing for the "10^4-10^5" claim (or drop it), (3) replace the "follow-up work hedge" citation with the actual current R1-M1 row text and specify the structural strategy (paragraph-in-cell vs new section), (4) widen the Cohen's d range to `≤ -2.6` matching §7-Option A.

---

## Re-check (after planner revision)

**Verdict:** **NEEDS REVISION** — 1 BLOCKER + 4 WARNINGS remain.

The planner closed the structural intent of B1, B2, B3, W1, W2, W3, W4 in the task definitions, but the revision introduced (a) one fresh BLOCKER (gate-resolution failure for hand-typed WGAN adversarial totals — a B2-equivalent regression), and (b) systematic stale references in the `must_haves` frontmatter and artifact-attribution lines pointing to the old T1-T5 numbering instead of the new T1-T7. The R3-HI-1 sister-fix is genuinely landed in T2 Step E2. The Welch aggregator T3 is genuinely landed. The W3 task split into T4/T5 is genuinely atomic. The W4 preflight check in T1 is genuinely landed.

### Per-focus-area findings

#### 1. B1 closure (R3-HI-1 sister-fix into T2 Step E2) — **CLOSED**

T2 Step E2 (lines 863-955) applies `norm_log_delta = (log_delta - mu) / sigma` to `_real_references` at `:144-153`; updates `_model_seed_rows` to consume `real_refs["norm_log_delta"]` for the LR-scale path; preserves OD-scale path byte-untouched; extends module docstring with the R3-HI-1 sister-fix paragraph. T2 verify gate (line 1013) asserts LR-scale aggregates are at corrected magnitude (`quantum LR-EMD < 0.1`, `AR < 0.05`). The `distribution_emd.json` v2 artifact contract no longer carries an inherited-broken LR-scale row. **B1 fully closed at the task-definition level.**

#### 2. B2 closure (corrected param counts + adversarial framing) — **PARTIALLY CLOSED — 1 BLOCKER**

- Generator parameter counts (74 / 73 / 78 / 562 / 3) are correct against `results/model_info.json` — verified.
- The "10^4-10^5" old framing is fully removed (grep returns zero hits).
- The strong-claim prose uses `73-562 generator parameters AND the full ~2.5x10^5-parameter adversarial budget (generator + 250,881-parameter shared critic)` — this is the honest framing.
- **NEW BLOCKER:** The per-baseline equivalence table inserted by T4 (lines 1244-1250) hand-types `250,955` (wgan_mlp), `250,954` (wgan_cnn), `250,959` (wgan_lstm) as the WGAN adversarial totals. These values **do NOT exist in any JSON in the resolution corpus**. Inspection of `results/total_adversarial_param_budget.json` shows the WGAN entries carry only `shared_critic_n_params: 250881` — they do NOT have a pre-computed `total_adversarial_param_budget` field (only quantum models do: 250936, 250956, 251016). Combined with the v2.1 gate's `_NUM` regex `[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?` not handling commas, `250,955` tokenizes as TWO tokens (`250`, `955`); `955` has no JSON anchor; integer ε-tolerance is 0.5; the gate will FAIL to resolve these tokens. **T4's verify step `./qgan_env/bin/python scripts/verify_number_provenance.py --target docs/reviewer_response.md` will reject this output.**
- Additional gate-resolution risk: `250,881` (cited twice in the T4 H2 prose) tokenizes as `250` + `881`; `881` has no JSON anchor (the corpus carries `250881` as a single token). Existing pre-14-16 docs cite `250881` without comma (see `methods_full.md:244`, `:250`, `:255`, `:435` and `reviewer_response.md:201`) — those resolve because they're a single token. The plan's introduction of the comma-separated form is a regression.

**Fix path:** Either (a) drop the WGAN-total column from the T4 H2 table and replace with prose `"each WGAN adversarial budget = generator + 250881 (shared critic) per total_adversarial_param_budget.json#shared_critic_n_params"`, OR (b) extend T3's `welch_pairwise.json` schema (or extend `total_adversarial_param_budget.json`) to emit explicit `wgan_mlp_total: 250955`, `wgan_cnn_total: 250954`, `wgan_lstm_total: 250959` as JSON leaves so the gate can resolve them. And replace all `250,XXX` comma-separated literals in the doc body with `250XXX` no-comma form to match the existing audit-resolvable convention.

#### 3. B3 closure (R1-M1 row preserved + new H2 inserted) — **CLOSED at task-definition level, BUT stale narrative reference**

T4 Steps A + B explicitly preserve the R1-M1 row verbatim and insert a NEW H2 between the Marginal-convergence and Completeness-sweep sections at a specified insertion point. T4 verify gate asserts `'Added matched-parameter classical WGAN-GP' in rr` confirming the row is unchanged. **B3 task-definition CLOSED.**

**Stale reference (WARNING — not blocking):** Line 308-309 in `<contract_changes>` still reads: *"The current R1-M1 response in `reviewer_response.md` hedges with 'this is being addressed in follow-up work'. T3 rewrites it to assert the explicit Welch-grounded equivalence claim."* This is the exact non-existent hedge that B3 flagged, plus mis-attributes the rewrite to T3 (it's T4) AND mis-frames the action as a rewrite (it's a preservation + insertion). The contract_changes preamble was not updated to match the closure. Line 1939 (trust boundary table) also says "the R1-M1 rewrite" which is stale.

#### 4. W2 architectural check (welch_pairwise.json + T1→T3→T4 chain) — **PARTIALLY CLOSED — 1 WARNING**

- T3 reads only post-T1 aggregates (`matched2000_dualscale.json` + `model_info.json`) — no raw seed access. **Confirmed clean.**
- T3 → T4 depends_on chain is acyclic (sequential auto tasks). **Confirmed.**
- Schema captures the 4 strong-claim summary anchors as JSON-resolvable fields: `summaries.OD_floor_welch_p_quantum_vs_classical`, `summaries.OD_ceiling_abs_cohen_d_quantum_vs_classical`, `summaries.log_return_ceiling_welch_p_quantum_vs_wgan`, `summaries.log_return_extremum_cohen_d_quantum_vs_wgan` — **the KEY-paths exist.**

**WARNING — JSON leaf values vs literal threshold values mismatch:** The new H2 cites threshold literals `p > 0.36`, `|d| ≤ 0.65`, `p ≤ 0.014`, `d ≤ -2.6`. The `summaries` block stores the actual COMPUTED values, which are strictly greater (or less) than the thresholds — e.g., `OD_floor_welch_p_quantum_vs_classical` might be `0.376` (passes the `> 0.36` claim) but the literal `0.36` does NOT resolve to `0.376` under the gate's float ε-neighborhood (tol=0.005 for `0.36`, computed value ≥ 0.36 + tolerance margin). Same problem for `0.014`, `-2.6`. The literals are threshold ANCHORS, not data points; the gate is content-agnostic and resolves only numeric literals against JSON leaves. T4's `verify_number_provenance.py --target reviewer_response.md` may FAIL on these threshold literals unless T3 also emits the threshold values as explicit JSON leaves (e.g., `strong_claim_thresholds: {floor_p_OD: 0.36, ceiling_abs_d_OD: 0.65, ceiling_p_LR: 0.014, extremum_d_LR: -2.6}`).

**Fix path:** Add to T3's `welch_pairwise.json` schema a top-level `strong_claim_thresholds` field with the exact literal values `{"floor_welch_p_OD": 0.36, "ceiling_abs_cohen_d_OD": 0.65, "ceiling_welch_p_LR_vs_wgan": 0.014, "extremum_cohen_d_LR_vs_wgan": -2.6}`. T3 acceptance criteria should assert these literals are present as JSON leaves. Otherwise the v2.1 gate is solving the wrong resolution problem — it resolves the THRESHOLD literals via coincidental ε-match against unrelated EMD values (the exact failure mode planner-checker W2 was supposed to close).

#### 5. W3 closure (T4/T5 atomic + independent) — **CLOSED**

T4 `<files>` = `docs/reviewer_response.md` only (single-file atomic edit). T5 `<files>` = `run_model_info.py + methods_full.md + reconciliation_note.md` (one emitter + the docs it regenerates — a coherent single emit cycle). T4 does not read T5 outputs; T5 does not read T4 outputs. Neither depends on the other's commit state for gate-pass. T5 verify gate runs the v2.1 gate against methods_full.md + reconciliation_note.md only — independent of T4's reviewer_response.md. **W3 fully closed.**

#### 6. Task renumbering cross-check — **NOT FULLY MIGRATED — WARNING**

The PLAN's `<task>` definitions (T1-T7) are correctly numbered and scoped. However the frontmatter `must_haves.truths`, `must_haves.artifacts`, `must_haves.key_links`, threat_model entries, `<contract_changes>`, and `<interfaces>` blocks carry multiple stale references to the old T1-T5 numbering. None of these block execution (the executor reads `<task>` definitions), but they create cross-task audit ambiguity. Stale references found at lines:

- **Line 36** (truth, D-14-16 closure narrative): *"after T3 lands"* — the gate runs against the updated docs after **T5** lands (not T3, which is the welch aggregator); T4 runs the gate against `reviewer_response.md` first.
- **Line 41** (truth, R3-CR-1 schema): *"C-3 disclosure paragraph extension that T3 writes into reconciliation_note.md"* — T5 writes the C-3 extension.
- **Line 44** (truth, R1-M1 / new H2): *"After T3, `docs/reviewer_response.md` R1-M1 row..."* — should read "After T4".
- **Line 47** (truth, cross_model_emd re-render): *"After T4, `figures/cross_model_emd.{png,pdf,json}` is RE-RENDERED"* — figure re-render is T6, not T4.
- **Line 49** (truth, v2.1 gate run on 10 docs): *"After T4, v2.1 gate runs PASSING against all 10 paper-facing docs"* — the 10-doc gate run is T6 (T4 runs only against reviewer_response.md).
- **Line 74** (artifact `scripts/run_model_info.py`): *"Updated (Task 3) to surface the corrected aggregates"* — should be Task 5. Also contradicts T4 by suggesting "R1-M1 response rewrite + Parametric-efficiency subsection logic MAY live in this emitter" (T4 task-definition is a manual edit, not an emit).
- **Line 83** (artifact `methods_full.md`): *"Re-emitted (Task 3)"* — should be Task 5.
- **Line 92** (artifact `cross_model_emd.png`): *"Re-rendered (Task 4)"* — should be Task 6.
- **Line 134** (key_link via): *"Task 4 re-renders the cross-model EMD bar figure"* — should be Task 6.
- **Line 137** (key_link to): *"v2.1 gate (D-14-16 byte-frozen, T4 + T5 invocations)"* — should include T6 (the 10-doc gate run is T6).
- **Line 693, 728, 969** (T2 read_first + Step A + Step G): *"C-3 disclosure extension that T3 writes"* — should be T5.
- **Line 1939** (trust boundary table): *"the R1-M1 rewrite + the new Parametric-efficiency subsection"* — R1-M1 is preserved, not rewritten.
- **Line 308-309** (`<contract_changes>`): *"current R1-M1 response in `reviewer_response.md` hedges with 'this is being addressed in follow-up work'. T3 rewrites it to assert..."* — the hedge does not exist (B3 finding); the work is T4 (not T3); the action is a preservation + insertion (not a rewrite). **This paragraph directly contradicts the B3 closure narrative at lines 2209-2216.**

#### 7. Sanity sweep — **WARNING**

- **Hand-typed unsourced numbers:** `250,955`, `250,954`, `250,959` (WGAN adversarial totals in T4's H2 table) have no JSON anchor — captured as Issue #2 BLOCKER above.
- **Comma-separated 250,881 form:** introduces tokenizer regression vs existing `250881` (no comma) usage in already-audit-passing docs — captured as Issue #2 BLOCKER above.
- **Byte-freeze invariant assertions per task:**
  - D-14-22 (`core/` byte-freeze): asserted in **all 7 task verify gates** (T1 line 670, T2 1013, T3 1157, T4 1310, T5 1456, T6 1590, T7 1913). ✓
  - D-14-16 (gate byte-freeze): asserted in T3, T4, T5, T7 verify gates. **T1, T2, T6 do not explicitly assert `git diff verify_number_provenance.py` is empty.** Minor coverage gap — T1 + T2 don't touch the gate (low risk); T6 runs the gate but doesn't assert it's unmodified post-run (defensible since `scripts/verify_number_provenance.py` is a read-only invocation). Not a blocker; flag as soft warning.
  - D-14-13, D-14-18: documented in contract_changes preamble; not per-task verify gates. Inherently preserved by no-LaTeX-edit and no-strict-accept-edit scopes.

### Summary of remaining issues

| Severity | Issue | Where | Fix path |
|---|---|---|---|
| **BLOCKER** | T4 H2 table hand-types WGAN adversarial totals `250,955` / `250,954` / `250,959` not in any JSON; comma-separated `250,881` tokenizes incompatibly. v2.1 gate will reject `reviewer_response.md`. | T4 Step B (lines 1244-1250), the prose at line 1253-1255. | Either drop the WGAN-total column (use prose "generator + 250881 shared critic" no comma) OR extend T3 to emit WGAN adversarial totals as JSON leaves in welch_pairwise.json (e.g., a `model_parameter_counts` block). Replace all `250,881` with `250881` to match existing audit-resolvable convention. |
| **WARNING** | Threshold literals `0.36`, `0.65`, `0.014`, `-2.6` in T4's H2 cite thresholds, not stored computed values; v2.1 gate cannot resolve them against welch_pairwise.json summaries (which store actual ≥0.36 / ≤0.65 etc. values). | T4 Step B strong-claim assertion; T3 schema spec at lines 1095-1100. | Extend T3 schema with explicit `strong_claim_thresholds: {floor_welch_p_OD: 0.36, ceiling_abs_cohen_d_OD: 0.65, ceiling_welch_p_LR_vs_wgan: 0.014, extremum_cohen_d_LR_vs_wgan: -2.6}` JSON leaf field; add T3 acceptance criteria asserting these leaves exist. |
| **WARNING** | Contract_changes line 308-309 still says R1-M1 "hedges with 'this is being addressed in follow-up work'" and that "T3 rewrites it" — directly contradicts B3 closure narrative. | `<contract_changes>` block at line 307-313. | Replace with the closed narrative: "The current R1-M1 row makes no equivalence claim; T4 PRESERVES it verbatim and INSERTS a new H2 section asserting the strong claim." Update line 1939 trust-boundary table similarly. |
| **WARNING** | Frontmatter `must_haves` blocks and artifact attributions carry 11 stale T-number references (T3/T4 → should be T4/T5/T6) — see line list in §6 above. | Lines 36, 41, 44, 47, 49, 74, 83, 92, 134, 137, 693, 728, 969, 1939. | Mechanical search-and-replace to align with the new T1-T7 scheme. |
| **WARNING** | T1, T2, T6 verify gates do not assert `git diff verify_number_provenance.py` is empty (other tasks do). Minor D-14-16 coverage gap. | T1 line 670, T2 line 1013, T6 line 1590. | Add `&& [ -z "$(git diff verify_number_provenance.py)" ]` to those three verify gates for symmetry. |

### Recommendation

**RETURN TO PLANNER for one more revision cycle.** The B2 regression is a hard execution blocker (T4's verify step will fail the v2.1 gate on hand-typed WGAN adversarial totals + comma-separated `250,881`). The W2 threshold-literal architectural gap is a secondary concern that may also block T4's gate, depending on whether the gate's ε-neighborhood matching happens to coincidentally resolve the thresholds against unrelated JSON values (which is exactly the failure mode W2 was supposed to close — so any "it passes anyway" outcome would be hollow).

The remaining warnings (stale T-numbering, contract_changes drift, D-14-16 per-task gate-byte-freeze coverage) are cosmetic but should be cleaned up for audit hygiene since the SUMMARY itself will reference these locations and any reviewer reading the plan will see the internal contradictions.

After the next revision:
- BLOCKER fix: either drop hand-typed WGAN totals or emit them as JSON leaves; normalize `250,881` → `250881`.
- WARNING fix: emit threshold literals as JSON leaves in `welch_pairwise.json`.
- WARNING fix: rewrite contract_changes paragraph at line 307-313 to match B3 closure.
- WARNING fix: mechanical migration of 11 stale T-references.

If the planner closes the BLOCKER + WARNING #2 (threshold-literal anchoring), the plan can proceed to execution even with the cosmetic stale references intact (no execution-blocking impact).

## Re-check 2 (post-revision-2)

**Verdict:** NEEDS REVISION (cosmetic blocker only — execution-safety blockers all closed)

### Closure status per item

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | B2 regression: no comma-formatted `250,xxx` literals; `250881` anchored to `total_adversarial_param_budget.json#shared_critic_n_params` | **CLOSED** | `grep -n '250,' PLAN.md` → 0 hits. All 27 occurrences of the literal are bare `250881` (e.g., lines 51, 190, 321, 1261, 1310, 1323-1324, 1331-1333, 1339-1340, 1388-1393, 1813, 1890, 2114, 2207, 2217-2218, 2255, 2297-2298). The audit-passing form is now enforced and explicitly cited (line 1407: "no comma in the literal — matches the existing audit-passing form in methods_full.md:244,250,255"). The strong-claim table cites `welch_pairwise.json` lookups for Welch/d + `total_adversarial_param_budget.json#shared_critic_n_params` for `250881` (line 1340). Per-WGAN adversarial-total enumeration replaced with prose ("generator + 250881 shared critic" per row) at lines 1331-1333. |
| 2 | W2 threshold-vs-computed: `strong_claim_thresholds` block as first-class JSON leaves; emitter asserts thresholds; T3 verify gate has leaf-equality assertions | **CLOSED** | T3 artifact `welch_pairwise.json` at line 71 names the four threshold leaves explicitly. T3 schema spec at lines 1112-1118 declares `strong_claim_thresholds: {floor_welch_p_OD: 0.36, ceiling_abs_cohen_d_OD: 0.65, ceiling_welch_p_LR_vs_wgan: 0.014, extremum_cohen_d_LR_vs_wgan: -2.6}`. T3 Step B2 (lines 1152-1184) emits the block AND adds pre-write `assert summaries["X"] > payload["strong_claim_thresholds"]["X"]` direction checks for all 4 thresholds. T3 acceptance criteria (lines 1202-1208) require exact-leaf equality. T3 `<automated>` (line 1234) has the four `(thr.get('floor_welch_p_OD') == 0.36)` exact-match assertions. |
| 3 | B3 narrative: no "follow-up work" or "T3 rewrites R1-M1" attribution; consistent "R1-M1 PRESERVED + new H2 INSERTED" framing | **CLOSED** | `grep -niE 'follow-up work\|T3 rewrites R1-M1'` → 0 hits. All 4 mention sites converge on the closed framing: line 44 ("LEFT VERBATIM ... a NEW H2 section ... is INSERTED"), line 80 ("LEFT VERBATIM. A new H2 section ... is INSERTED"), line 313 ("PRESERVES that row verbatim and INSERTS a new H2 section"), lines 1968 / 1881 / 2031 / 2047 / 2103 / 2202 / 2304 (all repeat "PRESERVED verbatim + NEW H2 INSERTED"). Contract_changes block at lines 307-323 is now the corrected version. |
| 4 | Task-number renumbering: all 14 stale T-references updated to 7-task scheme | **NOT CLOSED** | The planner closed some references but introduced more new ones. Stale T-N references that still misalign with the authoritative 7-task numbering (T1=dualscale fix, T2=distribution_emd fix, T3=welch aggregator, T4=reviewer_response, T5=methods_full + reconciliation_note + run_model_info, T6=figure re-render + gate, T7=SUMMARY + peer_review_remediation + completeness_sweep_manifest) — see authoritative `<name>` headers at lines 528, 696, 1045, 1254, 1417, 1563, 1696. Stale references still present: **must_haves.truths**: line 45 ("After T3, methods_full.md..." — should be T5), line 46 ("After T3, reconciliation_note.md..." — should be T5), line 50 ("After T5, 14-16-SUMMARY.md exists..." — should be T7). **must_haves.key_links**: line 125 ("reviewer_response.md ... (T3 output)" — should be T4), line 129 ("reconciliation_note.md ... column 3 (T3 output)" — should be T5), line 133 ("cross_model_emd ... (re-rendered, T4)" — should be T6). **Objective prose**: line 224 ("methods_full.md with corrected numbers ... (T3)" — should be T4+T5), line 227 ("v2.1 gate run on all 10 paper-facing docs (T4)" — should be T6), line 230 ("completeness_sweep_manifest.md (T5)" — should be T7). **Interfaces block**: line 396 ("run_model_info.py (READ-FIRST + EDIT in T3)" — should be T5), line 405 ("T3 read_first") — should be T5, line 429 ("methods_full.md (READ-FIRST + EDIT in T3)" — should be T5), line 437 ("reconciliation_note.md (READ-FIRST + EDIT in T3)" — should be T5), line 444 ("run_figure_suite.py (READ-FIRST in T4...)" — should be T6), line 448 ("T4 invokes the existing render path") — should be T6, line 453 ("T4 only appends a plan_14_16_verification field") — should be T6, line 457 ("cross_model_emd (RE-RENDERED in T4)" — should be T6), line 463 ("CONDITIONALLY TOUCHED in T4)" — should be T6), line 468 ("T4 invokes it against all 10 paper-facing docs") — should be T6, line 471 ("peer_review_remediation.md (READ-FIRST + EDIT in T5)" — should be T7), line 478 ("completeness_sweep_manifest.md (READ-FIRST + EDIT in T5)" — should be T7), line 501 ("**T3:** run_model_info.py:220-302") — should be T5, line 509 ("**T4:** run_figure_suite.py") — should be T6. **Frontmatter `depends_on` chain inferred**: lines 36, 49 ("After T6, v2.1 gate ... runs PASSING") are correctly T6, but line 49 also says the gate runs against welch_pairwise.json "produced by T3" — that's correct. Total stale references identified: **~22 surviving stale T-N references** misaligned with the authoritative 7-task scheme (more than the 14 the prior pass identified — the renumbering pass appears to have been partial). |
| 5 | D-14-16 symmetry: all 7 task verify gates assert `git diff verify_number_provenance.py` empty | **CLOSED** | `grep -n 'git diff verify_number_provenance.py' PLAN.md` returns hits in all 7 task `<automated>` blocks: T1 line 681, T2 line 1024, T3 line 1234, T4 line 1402, T5 line 1548, T6 line 1682, T7 line 2005. Each task also lists this in its `<done>` block (lines 689, 1039, 1248, 1411, 1557, 1690, 2014). Line 2014 explicitly states "D-14-16 v2.1 byte-freeze preserved across all 7 tasks". |

### Severity calibration

- **BLOCKER vs WARNING for item 4:** The stale T-N references are **cosmetic, not execution-blocking**. The executor receives the plan as a single document and reads `<task>` blocks sequentially with their authoritative `<name>` headers (lines 528, 696, 1045, 1254, 1417, 1563, 1696); the must_haves/interfaces/prose mismatches do not change which file gets edited (the `<files>` and `<action>` blocks within each task are correct). The risk is **audit-trail confusion**: a future reviewer reading the must_haves block will see "After T3, methods_full.md..." and not find methods_full.md in T3 (it's in T5). For a 7-task plan with high-stakes forensic remediation that needs to be defensible to journal reviewers, this is a meaningful audit hygiene blocker — but it does not change the executed behavior.

- **Per the prior re-check's own framing** ("the remaining warnings are cosmetic but should be cleaned up for audit hygiene since the SUMMARY itself will reference these locations and any reviewer reading the plan will see the internal contradictions") — that calibration still holds. Item 4 is a WARNING, not a BLOCKER.

### Recommendation

**APPROVED WITH ONE WARNING** — the plan is execution-safe. All four execution-blocking items (B2 numeric literals, W2 threshold provenance, B3 narrative consistency, D-14-16 symmetry) are CLOSED. Item 4 (task-number renumbering) is partially closed with ~22 stale references surviving in audit-facing locations (must_haves, interfaces, prose). The stale references do NOT affect which files get edited by which task — the authoritative `<name>` + `<files>` + `<action>` blocks per task are internally consistent. Recommend either:

- **Option A (preferred):** Planner does one final mechanical search-and-replace pass on stale T-N refs in lines 45, 46, 50, 125, 129, 133, 224, 227, 230, 396, 405, 429, 437, 444, 448, 453, 457, 463, 468, 471, 478, 501, 509 (per the explicit list in the table row above), then plan ships.

- **Option B (acceptable):** Plan ships as-is with item 4 flagged in the 14-16-SUMMARY's Self-Check as a known audit-hygiene caveat ("must_haves and interfaces blocks reference some tasks by their pre-revision numbering; authoritative task scope is per `<name>` headers in PLAN.md").

Either path is execution-safe. Plan is unblocked from the executor's perspective; only the audit-trail hygiene is degraded.
