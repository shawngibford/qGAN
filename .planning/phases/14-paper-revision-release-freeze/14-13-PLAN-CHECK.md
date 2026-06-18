# 14-13 Plan Check Report

**Plan:** `.planning/phases/14-paper-revision-release-freeze/14-13-PLAN.md` (1,647 lines, 7 atomic tasks, Wave 11)
**Approved design:** `~/.claude/plans/we-need-circuit-diagrams-inherited-waterfall.md`
**Peer-review source:** 5 reports under `.planning/phases/14-paper-revision-release-freeze/peer-review/`
**Reviewer date:** 2026-05-20

---

## Verdict

**PASS-WITH-FIXES**

Plan is structurally sound, faithfully translates the approved design, and covers 27 of 28 actionable findings (HI-9 explicitly OOS). All locked decisions (CR-4 disclose+future-gate, `best_checkpoint.pt` direct-commit, D-14-16 lift Task 2 only, D-14-22 preserved, D-14-13 extended) are honored exactly as specified. File:line citations spot-check 100% accurate against the live repo. ROADMAP edits match the actual repo state. `core/` byte-freeze enforced by per-task gate (`[ -z "$(git diff --stat core/)" ]`).

However, **5 fixable issues** would either confuse reviewers using the remediation index (HIGH) or cause spurious verify failures on otherwise-correct work (HIGH/MEDIUM). None of these threaten the 7-task atomic-commit boundary or D-14-22; they are surface-level corrections the planner should apply before T7 executes.

---

## Findings by severity

### HIGH

#### H-CHK-1 — Misattribution of C-1/C-2/C-3 findings to quantum-review.md (actually math-review.md)

**Plan lines:** 628, 1404–1409 (peer_review_remediation.md template's `## quantum-review.md` table)
**Conflicts with:** `peer-review/math-review.md:74` (C-1), `:136` (C-2), `:176` (C-3) — these three CRITICAL findings are **only** present in math-review.md. The quantum-review.md contains zero `C-N` findings; its enumerated items are MINOR (IQP nomenclature, "data re-uploading" docstring) plus a MEDIUM (barren-plateau probe for V2 absent at §10).

**Impact:** The reviewer-facing `peer_review_remediation.md` (the Task 7 deliverable that maps every finding to a commit SHA) would direct a reviewer to quantum-review.md to read the C-1 / C-2 / C-3 citations — but those reviewers would find nothing there. The misattribution propagates from the read_first list (line 628) into the remediation-index table template (lines 1404–1409). The plan acknowledges that the reviewers triangulated the same root causes; that's true (PROV-CRIT-1 in provenance-review.md, CR-2 in code-review.md, C-1/C-2/C-3 in math-review.md), but the index must list the actual originating report per finding ID.

**Fix:** Move the C-1 / C-2 / C-3 rows from the `## quantum-review.md` section to the `## math-review.md` section (which currently only carries H-1 / H-2 / H-3 / M-2 / M-3 / M-4). The `## quantum-review.md` section can then either be removed (no actionable findings from that reviewer) or list the three MINOR/MEDIUM items the plan does NOT address (IQP-nomenclature footnote, quantum.py:1 docstring misnomer, barren-plateau acknowledgement for V2) under "Acknowledged but out-of-scope".

#### H-CHK-2 — 4 of 9 timeseries figure paths in plan don't match disk reality

**Plan lines:** 40–48 (frontmatter `files_modified`), 974 (Task 5 `<files>`), 1095 (Task 5 determinism verify model list)
**Conflicts with:** disk listing `ls figures/timeseries_*.png` returns: `ar`, `iqp_sel_55_repro`, `V1`, `V2`, `V3`, `vae`, `wgan_cnn`, `wgan_lstm`, `wgan_mlp` (9 files).

The plan lists `default_75`, `iqp_sel_55`, `V1`, `V2`, `V3`, `vae`, `wgan_cnn`, `wgan_lstm`, `wgan_mlp` — i.e. it includes `default_75` (no such timeseries figure on disk) and `iqp_sel_55` (the actual file is `iqp_sel_55_repro`); it omits `ar` and `iqp_sel_55_repro`.

The Task 5 determinism sanity verify (line 1095) iterates the same wrong list, so the determinism check will silently skip the 2 figures that actually exist under different names AND look for 2 files that don't exist (the `if pathlib.Path(...).exists()` guard turns these into silent zeros — false PASS).

**Impact:** CR-1 (non-deterministic seed) is the headline CRITICAL fix for Task 5. The plan's two-pass byte-identity test is supposed to prove the SHA-256 seeding actually delivers determinism. With the wrong filenames, the test will exit 0 against an empty first/second snapshot — a vacuous PASS that wouldn't catch a regression of the very bug the task is meant to fix.

**Fix:** Replace the model list in 3 places (frontmatter, Task 5 `<files>`, Task 5 verify Python list) with `['ar','iqp_sel_55_repro','V1','V2','V3','vae','wgan_cnn','wgan_lstm','wgan_mlp']`. Also update the must_haves artifact path glob (line 111) to use `{ar,iqp_sel_55_repro,V1,V2,V3,wgan_mlp,wgan_cnn,wgan_lstm,vae}`.

### MEDIUM

#### M-CHK-1 — Verification check 3 caption-string mismatch (would spuriously fail T7 verify)

**Plan lines:** 1532 (verification.3) declares `cross_model_emd.json` must contain `metric: "OD-scale final-eval EMD, mean over seeds 42-46"`. Task 3's emit specification (lines 63, 96, 700–701) says the companion JSON caption is `OD scale, final-eval mean ± std over 5 seeds 42-46; frozen headline reference line on the same scale`.

These are two different strings; check 3's grep target won't appear in the emitted companion JSON. The Task 3 verify block at line 763 uses a more permissive check (`'OD scale' in caption OR 'final-eval mean' in caption`) which will pass, but the standalone verification.3 prose at line 1532 won't.

**Impact:** When a reviewer (or the executor) runs the 12-point verification checklist at T7 close, check 3 will appear to fail. The actual emit is correct; only the prose target needs to align.

**Fix:** Change verification.3 (line 1532) to either match the planned emit exactly (`"OD scale, final-eval mean ± std over 5 seeds 42-46"`) or use a key-name check (`caption contains "OD scale" AND contains "final-eval mean"`).

#### M-CHK-2 — MED-4 reviewer description references `n: null` but actual current state has `n_seeds: 5` (already populated)

**Plan lines:** 66 (must_haves truth), 92 (artifacts), 745–751 (Task 3 Step F), 1422 (remediation table)
**Conflicts with:** `matched2000_dualscale.json` aggregate rows currently carry `n_seeds: 5` (populated), NOT `n: null` as MED-4 (provenance-review.md:159) claims. The aggregator's emit has the count under a different key name.

**Impact:** The plan's prescribed fix is to "populate `n=5`/`n=1` per row" — which would add a SECOND sample-size field on top of the existing `n_seeds`. This is harmless but redundant; the reviewer's actual concern (a paper-facing aggregate doesn't carry its own n) is already addressed by the existing `n_seeds` field.

**Fix:** Either (a) reconcile the MED-4 source claim by inspecting the actual JSON (which would show MED-4 was based on stale state and is already partially fixed) and update the plan to just add an `n` alias for backward compat, OR (b) document explicitly that the fix is "add `n` field alongside `n_seeds`" so the executor doesn't trip on the discrepancy mid-execution and conclude the reviewer was wrong.

#### M-CHK-3 — HI-5 partially mitigated already (`headline_model_kind` top-level field is already present)

**Plan lines:** 69 (must_haves truth), 98 (artifacts), 892–897 (Task 4 Step D), 1323 (remediation table)
**Conflicts with:** `matched2000_dualscale.json` already has top-level `"headline_model_kind": "frozen_checkpoint_headline"`. The reviewer's HI-5 concern is that `model_kinds` excludes the headline; the plan correctly notes "or add `headline_model_kind` (already done at line 600)" in the read_first prose, but the action prescribes adding the field anyway. Only the missing-from-model_kinds piece is the live defect.

**Impact:** Executor may attempt to add a field that exists, hit a no-op or duplicate, and waste a verify cycle. Or worse, may overwrite the existing value if reading the source incorrectly.

**Fix:** Tighten Task 4 Step D to: "Include `HEADLINE_MODEL_KIND` in the `model_kinds` list at top — the existing `headline_model_kind` top-level field (already present in current emit) requires no change." Drop the "Add a `headline_model_kind` documentation field" sub-step.

### LOW

#### L-CHK-1 — Plan's "C-1 / PROV-CRIT-1" double-cite pattern is technically correct but the must_haves narrative blurs them as a single finding

**Plan lines:** 61, 94 — describes C-1 (math-review) and PROV-CRIT-1 (provenance-review) as a single shared resolution. They ARE the same root-cause defect surfaced by two independent reviewers, and the plan correctly resolves both via Task 3. No fix required, but for clarity the remediation index in Task 7 could note "PROV-CRIT-1 is the same finding as C-1, independently triangulated" — currently it just says "(same as C-1)" at line 1414 which is sufficient.

**Fix:** None required — informational only.

---

## Dimensions verified

| # | Check | Status |
|---|---|---|
| 1 | Goal coverage — 27/28 findings traced to concrete tasks | PASS (HI-9 explicit OOS) |
| 2 | Cross-task dependencies — T2 before T7; T3 before T7 | PASS (T3 depends on T2's v2 gate; T7 reruns the gate against T3/T4/T5 emits) |
| 3 | Atomic commit boundaries — file edits per task non-overlapping | PASS (methods_full.md edited in T1+T3+T5+T6 but at distinct sections §4.1+§5.1 / §2.k / CR-4 disclosure / §3.x+§2.i+§2.j) |
| 4 | `core/` byte-freeze preserved | PASS (every task's verify block asserts `[ -z "$(git diff --stat core/)" ]`) |
| 5 | D-14-16 lift scope — Task 2 is the only task editing `scripts/verify_number_provenance.py` | PASS |
| 6 | File:line citations — 100% accurate spot-check | PASS (`run_methods_full.py:466-468`, `verify_number_provenance.py:119`, `run_figure_suite.py:382`, `run_figure_suite.py:1117`, `run_figure_suite.py:1146`, `run_figure_suite.py:620-654`, `run_matched2000.py:344`, `run_matched2000.py:434-492`, `run_matched2000_dualscale.py:522`, `eval.py:25-36`, `verify_freeze_ready.py:82,116`, `run_canonical_headline.py:280,334` — all verified live) |
| 7 | Verification section actionability | PASS-WITH-FIXES (12 checks runnable, but check 3 has caption-string mismatch — M-CHK-1) |
| 8 | Resume-after-clear flow — foreground-executor requirement | PASS (line 1607–1610 explicitly cites the canonical reference) |
| 9 | ROADMAP.md consistency — Plans 12→13, Wave 11 added, D-14-23 extended | PASS (live ROADMAP shows "13 plans", Wave 11 with 14-13, D-14-23 scope note updated; progress row not yet flipped to 11/13 because 14-13 not yet executed) |
| 10 | Goal-backward — if 7 tasks land as written, do the 12 verification checks pass? | PASS-WITH-FIXES (10/12 yes; check 3 fails per M-CHK-1; the timeseries determinism check in T5 silently passes vacuously per H-CHK-2) |

---

## False-positives skipped

The following were considered as potential findings but correctly NOT flagged because they are locked decisions explicitly carried in the planning conversation:

1. **CR-4 disclose+future-gate (no classical sweep re-run).** The plan ships a 1-paragraph honest disclosure in `methods_full.md` and `reviewer_response.md` plus a forward-only gate extension (D-14-13 + `training_time_device` strict-accept). The omission of a full ~3–6 h classical sweep re-run is the user's explicit decision. Not a finding.

2. **`best_checkpoint.pt` direct commit via `.gitignore` exception.** The plan's choice to ship the ~6 MB binary directly (not via git-lfs, not via Zenodo-deferred) is the locked decision. The `.gitignore` edit at the executor side (not stashed) is intentional. Not a finding.

3. **D-14-16 lift for Task 2 only.** The schema bump to `"v2 (Phase 14 plan 14-13 — boundary-strict resolution + render-only exclusion)"` is authorized for this plan; the gate is back in byte-freeze under the new schema after T2 closes. Not a finding.

4. **D-14-22 preserved for M-2 / M-3 / MD-1 / MD-7 / LO-1 / M-4.** Documenting these findings in `methods_full.md §3.x` (instead of patching `core/`) is the locked decision. The plan correctly distinguishes "DOCUMENTED" (Task 6) from "CHANGED" (which would require lifting D-14-22). Not a finding.

5. **HI-9 (is_complete subprocess perf) OUT OF SCOPE.** The reviewer's own annotation says perf-only, no correctness impact. The plan explicitly excludes it. Not a finding.

6. **Quantum-review.md MINOR-1 / MINOR-2 / MEDIUM (barren-plateau).** These items are NOT in the "12 CRITICAL + 16 HIGH" target set; they are MINOR / MEDIUM and were not promised to be addressed in this plan. The IQP-nomenclature footnote, quantum.py:1 docstring fix, and V2 barren-plateau acknowledgement could be future work, but their omission from 14-13 is NOT a scope violation since the plan's scope is explicitly the 28 CRITICAL+HIGH findings. Not a finding (would be a separate follow-up plan).

7. **Pre-existing `.gitignore` modifications stash cycle.** The user's local working tree has unrelated dirty `.gitignore` edits that stay on the stash; only the new `!checkpoints/best_checkpoint.pt` exception is applied at the executor side. This is the documented Phase 14 protocol per T-14-43 / project memory. Not a finding.

8. **No retraining, no new figures beyond re-renders.** The plan explicitly forbids retraining (matches the "pure correction sweep" framing). All figure changes are re-renders of existing artifacts against post-fix aggregates. Not a finding.

---

## Recommended action

The 5 fixable issues (1 HIGH + 1 HIGH + 3 MEDIUM) above are all surface-level — they involve correcting the remediation-index header attribution (H-CHK-1), aligning 9 file paths to disk reality (H-CHK-2), aligning a verification-check string with the planned emit (M-CHK-1), reconciling MED-4 vs current state (M-CHK-2), and tightening HI-5's redundant sub-step (M-CHK-3).

None require redesigning the 7-task structure, none change which findings get addressed, and none threaten D-14-22 or the atomic-commit boundary. The planner can apply all 5 corrections to the plan file in a single revision pass.

**After the 5 fixes are applied, the plan is APPROVED for foreground execution via `/gsd-execute-phase 14`.**
