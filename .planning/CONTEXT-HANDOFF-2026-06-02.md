# Context Handoff — 2026-06-02 (for incoming clean-context agent)

**Audience:** A fresh Claude session picking up this work cold. The user has edits/comments to share for the current state of the AIChE QWGAN-GP paper manuscript.

**Your first 60 seconds:**
1. **Recompile the PDF** before assuming you know what the user is looking at — there's an open PDF-staleness issue (see §1).
2. Skim §2 (project identity) and §6 (hard prohibitions) so you don't accidentally re-introduce known-corrected regressions.
3. The user will share their edits/comments next. Treat each edit as a candidate change against the v1.2.4 baseline; run gates (§7) before committing.

---

## §1 — IMMEDIATE PDF-FRESHNESS WARNING

**Symptom (user-reported 2026-06-02):** "I don't see any of the changes you made in the current rendered pdf."

**Diagnosis:** The repo-root `main (4) copy.pdf` was found at timestamp `Jun 1 13:52` — that is BEFORE the v1.2.4 commits (Fig 6 at ~14:00, Supp §A.9 grid at ~14:30+). It has been recompiled fresh at `Jun 2 11:20` to **60 pages** with all v1.2.4 content visible. If the user is still looking at a stale render:

- They may have it open in a PDF viewer that doesn't auto-reload (Preview.app on macOS is notorious for this — close the file and reopen, or use `open -F`)
- They may be looking at `~/Desktop/aiche_upload_v1.2.1/main (4) copy.pdf` (which is intentionally preserved as historical, 52 pages, 11 figures)
- The current/correct PDF is at `/Users/shawngibford/dev/phd/qGAN/main (4) copy.pdf` (60 pages, 22 visible figure panels)

**If the user says "still don't see it":** run this and report the SHA + timestamp + page count back:

```bash
ls -la "main (4) copy.pdf"
md5 "main (4) copy.pdf"
pdfinfo "main (4) copy.pdf" 2>/dev/null | grep -E "Pages|Title"
```

---

## §2 — Project identity

- **Repo:** `/Users/shawngibford/dev/phd/qGAN/`
- **Project:** qGAN — quantum WGAN-GP for bioprocess optical-density time-series synthesis (PhD research)
- **Current milestone:** v2.0 AIChE Major Revision Response
- **Manuscript:** AIChE Journal `aic-4719598`, Major Revision. Three-week extension expires ≈ 2026-06-17 (15 days out as of 2026-06-02)
- **Authors:** Shawn M. Gibford, Mohammad Reza Boskabadi, Christopher J. Savoie, Seyed Soheil Mansouri
- **Filename quirk:** the main `.tex` is `main (4) copy.tex` — literal space + parenthesis in filename. **Always quote it in shell** (`"main (4) copy.tex"`).

---

## §3 — Current state at HEAD (`e63f18a`, tag `v1.2.4`)

| Item | Value |
|---|---|
| Branch | `main`, in sync with `origin/main` |
| HEAD | `e63f18a docs(state): record v1.2.4 loss-diagnostics patch` |
| Working tree | Clean |
| Tag | **`v1.2.4`** on origin (5 tags total: `v1.2`, `v1.2.1`, `v1.2.2`, `v1.2.3`, `v1.2.4`) |
| Main pages | **60** |
| Main figures | **6** (Fig 1–6) |
| Supp figures | **15** (A1–A15; A15 is an 8-sub-panel grid) |
| Total visible panels | 28 (6 main + 22 supp counting sub-panels) |
| Compile state | 0 LaTeX errors, 0 BibTeX warnings, 0 undefined refs/cites |
| Provenance gate | PASS — 149 main + 198 supp literals trace to `results/*.json` |
| Freeze gates a/b/c | PASS |
| Freeze gate d (`release.md`) | Expected-deferred to plan 14-07 (Zenodo at journal acceptance) |
| AIChE upload bundle | `~/Desktop/aiche_upload_v1.2.4/` + `aiche_upload_v1.2.4.zip` (35 files, 2.9 MB) |
| Stale bundle preserved | `~/Desktop/aiche_upload_v1.2.1/` + `.zip` (52 pages, 11 figures — historical reference only) |

---

## §4 — Tag history (5 tags, scientific content is monotonic)

| Tag | Commit | What changed | When |
|---|---|---|---|
| `v1.2` | `34eb34e` | Post-swarm + 4-parallel-audit cleanup release; had **silent** `\Url`-in-moving-arg compile errors (PDF still rendered to 52 pages but captions had malformed `\path{}` content) | 2026-05-28 |
| `v1.2.1` | `3f4c2ef` | Clean-compile fix: wrapped 10 caption `\path{}` calls with `\protect`; dropped stray `\\` after `\bibliography{bib}` | 2026-05-28 |
| `v1.2.2` | `e89fd04` | Calibration-honesty patch: 4-agent audit found 1 BLOCK + 9 FLAGs; all 10 applied across 6 atomic commits + 1 audit-trail commit | 2026-05-28 |
| `v1.2.3` | `7bf3f2c` | Figure expansion: +8 reviewer-response figures (R1-M1 Pareto, R1-M2 TSTR, R1-M5 sensitivity ×2, R2-6 introspection ×3) | 2026-05-28 |
| **`v1.2.4`** | **`e63f18a`** | **Loss diagnostics: +Main Fig 6 training_convergence_all_models + Supp §A.9 8-panel per-model loss grid + 4 paragraphs of commentary** | 2026-06-01 |

**Key lesson from v1.2 → v1.2.1:** Pre-tag compile gates must `grep '^! '` in pdflatex log, not just count undefined refs/cites. pdflatex exits non-zero on `\Url`-in-moving-arg errors but still produces a PDF with garbled content where `\path{}` should be.

**Key lesson from v1.2.2:** Prohibition-sentinels that match exact phrases ("posterior collapse") can miss the same prohibited concept under different lexical shape ("std collapses toward zero"). The 4-agent text↔evidence audit catches mechanism-shape drift the phrase-matching gate cannot.

---

## §5 — The science (so you don't accidentally over- or under-claim)

The paper reports a **bifurcated empirical finding** under matched-parameter, matched-epoch (2000), 5-seed protocol against parameter-matched classical baselines (3 adversarial WGAN-GP at 73–78p + VAE 562p + AR(2) 3p):

1. **Exceed** (positive) — log-return temporal alignment: quantum LR-DTW 0.94–1.12 vs classical adversarial 1.58–6.86 (per-seed dominance, 60/60 cells, no overlap)
2. **Exceed** (positive, mean-level only) — lag-1 ACF closeness to real (-0.0641): quantum cluster mean −0.0997 to −0.0895 closer to real than any classical-baseline mean; **per-seed overlap exists** (e.g. wgan_lstm seed-46 lag-1 = −0.0761)
3. **Match** (null) — OD single-step marginal (OD-EMD): all 20 quantum-vs-classical pairs show Welch *p* > 0.36, |Cohen's *d*| ≤ 0.65 at *n*=5 power ≈ 15%, TOST equivalence NOT satisfied. Reported as "no statistically detectable difference under low power," NOT as "demonstrated equivalence."
4. **Fall short** (negative) — LR single-step marginal (LR-EMD): on per-model means, every classical adversarial baseline outperforms every quantum variant; AR(2) leads (with isolated per-seed counter-example at wgan_cnn seed 42 where wgan_cnn loses)

Framed as a positive-but-scope-limited proof-of-concept, NOT a general "quantum advantage" claim.

---

## §6 — Hard prohibitions (LOAD-BEARING; do NOT undo)

From `.planning/PAPER-SUBMISSION-HANDOFF.md §5` and `.planning/review-findings/REVIEW-FINDINGS.md`:

1. **VAE is a DEGENERATE GENERATION REGIME**, NEVER "posterior collapse" or "variance collapse" or "std collapses toward zero" or "near-constant sequence". Log-return std = 0.0186 ≈ 86% of real 0.0217 — the std is NOT collapsed. The anomaly is lag-1 ACF = −0.648 vs real −0.064 (anti-correlated step-to-step structure produces a high-frequency oscillation warped-aligned to real at low DTW cost). **v1.2.2 fixed a regression at 4 sites where the prohibited mechanism had returned under different wording — do not let it slip back.**

2. **Quantum dominates LR-EMD + LR-DTW + OD-DTW at the matched budget (post-14-21 correction).** Prior framing in v1.2.4 ("On LR-EMD, every classical adversarial baseline outperforms every quantum variant") was an artifact of the ×0.1 WGAN inverse-pipeline attenuation bug diagnosed and fixed in Plan 14-21 (smoking gun: `archive/qgan_pennylane_SEL.py:661-663` x0.1 scaling preserved verbatim into Pipeline B). Post-fix at the matched 2000-epoch budget: Q LR-EMD mean 0.00439 vs WGAN 0.06580 (~15× quantum advantage); Q OD-EMD mean 0.0288 vs WGAN 0.331 (~11.5×, Welch p=0.019, refuting the prior parametric-equivalence framing of §5 item 3); Q OD-DTW Welch p improves 0.011 → 0.002 post-fix; Q LR-DTW dominance survives directionally (range 6.09–9.48 vs WGAN 18.23–69.02). **NEW prohibition (carry forward):** do NOT over-claim per-SEED LR-EMD dominance — the ~15× advantage is on per-model means (n=5), not on every individual seed. Honest n=5 power language must be preserved everywhere. Do NOT discard the 14-18 "underpowered" caveat blanket — restate it scoped to where applicable; per-metric statistical-power language stays in §4/§5/supp. (post-14-21 correction — see `.planning/phases/14-paper-revision-release-freeze/.continue-here-t05.md` for full directional-shift report and `14-21-SUMMARY.md` for execution record.)

3. **Real-data lag-1 ACF reference is −0.0641** (matched-pipeline, with dither), NOT −0.029 (legacy unmatched).

4. **Pipeline B = log-returns + standardize + linear rescale to [−1, 1]** — NO Lambert W. Pipeline C (the dropped one) is `log-return → standardize → inverse Lambert W → rescale to [−1, 1]` (Lambert W is inserted BETWEEN standardize and rescale, not appended after — v1.2.2 F-4 fixed this). Lambert W only appears in the explicit "Pipeline C dropped per D-10-05" rationale (main §3.2, supp §A.7). `lambert_w_transform` / `inverse_lambert_w_transform` functions in `core/data.py` are retained for ablation reproducibility only.

Plus the A2-sentinel regex prohibitions: "deployable framework", "industrial bioprocess monitoring" (outside title), "high fidelity", "strong performance", "computational advantages", "Hybrid-GAN demonstrated/implemented/evaluated", "closed-loop feedback control" (for AI workflow), "n=1" for shot/noise context (those use n=3). See `PAPER-SUBMISSION-HANDOFF.md §5` for full sentinel list.

---

## §7 — Verification gates (run after EVERY edit before committing)

```bash
cd /Users/shawngibford/dev/phd/qGAN

# Provenance gate (numeric literals → JSON)
./qgan_env/bin/python scripts/verify_number_provenance.py --target "main (4) copy.tex"
./qgan_env/bin/python scripts/verify_number_provenance.py --target "supp_material.tex"
./qgan_env/bin/python scripts/verify_number_provenance.py --differential-test

# Freeze gates a/b/c (gate d expected-deferred to 14-07)
./qgan_env/bin/python scripts/verify_freeze_ready.py
# (gate d failure is OK; flag any OTHER failure)

# Clean compile — CRITICAL: grep '^! ' for actual errors, don't trust exit code alone
rm -f "main (4) copy.aux" "main (4) copy.log" "main (4) copy.out" "main (4) copy.toc" "main (4) copy.bbl" "main (4) copy.blg"
pdflatex -interaction=nonstopmode "main (4) copy.tex" > /tmp/c1.log 2>&1
BSTINPUTS=".:$HOME/Documents/main_qgan:" bibtex "main (4) copy" > /tmp/cb.log 2>&1
pdflatex -interaction=nonstopmode "main (4) copy.tex" > /dev/null 2>&1
pdflatex -interaction=nonstopmode "main (4) copy.tex" > /tmp/c3.log 2>&1
echo "Errors: $(grep -cE '^! ' /tmp/c3.log)"      # MUST be 0
grep -ciE "undefined" /tmp/c3.log                  # MUST be 0
grep -cE "duplicate" /tmp/c3.log                   # MUST be 0
grep -oE "Output written.*\([0-9]+ page" /tmp/c3.log
```

**Acceptance criteria for any edit-and-commit cycle:**
- 0 LaTeX errors
- 0 undefined refs/cites
- 0 hyperref duplicates
- Provenance gate PASS on both .tex (literal count may grow but never drop without justification)
- Differential self-test PASS
- Freeze gates a/b/c PASS (gate d's release.md is the expected deferral)

---

## §8 — Commit / tag conventions

**Commit style:** atomic per logical change, `<type>(<scope>): <imperative summary>`. Examples:

```
fix(paper-rewrite): correct VAE LR-DTW mechanism — lag-1 ACF, not variance collapse [B-1]
feat(paper-rewrite): add Fig 6 training_convergence_all_models — matched-budget convergence
docs(state): record v1.2.4 loss-diagnostics patch
```

**Tag policy:** new patch-level tag (`v1.2.X+1`) for each session's scientific or visual change set; OLD tags stay on origin as historical references (never rewrite published tags). Always include `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` footer.

**For commits with substantive prose changes:** the .tex must pass the provenance gate. If a new number is introduced, it MUST already exist in `results/*.json` — derived numbers (sums, differences, etc.) will be rejected. Either trace them or rephrase to avoid the literal.

---

## §9 — Where authoritative artifacts live

| Artifact | Path |
|---|---|
| Main manuscript | `main (4) copy.tex` |
| Supplement (`\input`-ed) | `supp_material.tex` |
| Bibliography | `bib.bib` (59 entries) |
| Bib style | `~/Documents/main_qgan/ama.bst` (external, not git-tracked) |
| Compiled PDF | `main (4) copy.pdf` (recompile if timestamp older than HEAD commit) |
| Submission handoff | `.planning/PAPER-SUBMISSION-HANDOFF.md` |
| Rebuttal handoff | `.planning/REBUTTAL-HANDOFF.md` |
| Per-reviewer rebuttal | `docs/reviewer_response.md` |
| 4-agent audit findings (v1.2.2) | `.planning/review-findings/REVIEW-FINDINGS.md` + per-agent files |
| Headline data JSON | `results/matched2000_dualscale.json` (705 KB) |
| Per-pair Welch tests | `results/welch_pairwise.json` |
| Cross-model EMD | `figures/cross_model_emd.json` |
| TSTR utility data | `results/tstr_matched2000.json` |
| Loss-curve sidecars | `figures/loss_<model>.json` |
| Per-model run metrics | `results/matched2000/runs/<model>/<seed>/metrics.json` |
| AIChE upload bundle | `~/Desktop/aiche_upload_v1.2.4/` + `.zip` |
| Project state | `.planning/STATE.md` |
| Project requirements | `.planning/REQUIREMENTS.md` |
| Project decisions | `.planning/DECISIONS.md` |
| Roadmap | `.planning/ROADMAP.md` |
| Memory (auto, this session) | `/Users/shawngibford/.claude/projects/-Users-shawngibford-dev-phd-qGAN/memory/` |

---

## §10 — Last 5 sessions chronology (what got done)

1. **2026-05-22 to 2026-05-27** — Paper-rewrite swarm (4 writers + 7 auditors sequential). Produced v1.2 candidate.
2. **2026-05-28 morning** — Post-swarm 4-parallel-audit-agent pre-tag cleanup sweep. Caught 4 BLOCK + 10 FLAG findings the swarm missed (especially Lambert W misdescription in Pipeline B prose; stale single-model figures with retrofitted captions). Tagged `v1.2`, pushed to origin.
3. **2026-05-28 afternoon (THIS chat-history continuation)** —
   - **`v1.2.1`**: Fresh-context session caught that v1.2's compile gate counted only undefined refs/cites and missed 11 `\Url`-in-moving-arg fatal errors. Fixed with `\protect\path{}` at 10 sites + dropped stray `\\` after `\bibliography`. Tagged + pushed.
   - **`v1.2.2`**: 4-agent calibration audit (text↔evidence, cross-section, figures↔captions, prose↔code). Found 1 BLOCK (B-1 VAE std-collapse mis-mechanization at 4 sites — the prohibited mechanism survived under different lexical shape than the killed phrase) + 9 FLAGs. All 10 applied in 6 atomic commits + audit-trail commit. Tagged + pushed.
4. **2026-05-28 evening** — **`v1.2.3`**: User flagged "figure/plot light." Added 8 reviewer-rebuttal figures (main +2: cross_model_emd, param_efficiency_pareto; supp +6: training_progression, entanglement_trajectory, param_trajectory, tstr_crossmodel, shot_noise, noise_robustness). Three new supp subsections. Tagged + pushed.
5. **2026-06-01** — **`v1.2.4`**: User requested loss plots + commentary. Added Main Fig 6 training_convergence + Supp §A.9 8-panel grid + 4 paragraphs of commentary surfacing WGAN-CNN drift vs quantum stability + VAE regularization collapse + AR(2) closed-form note. Tagged + pushed. Rebuilt AIChE upload bundle as `~/Desktop/aiche_upload_v1.2.4/`.

Each session's full commit + tag detail is in `git log v1.2..HEAD --oneline`.

---

## §11 — Open items

**External / human-only (not GSD-routable):**
- AIChE portal upload (use `~/Desktop/aiche_upload_v1.2.4.zip` — instructions in `UPLOAD-CHECKLIST.md` inside the bundle)
- GitHub release notes at `https://github.com/shawngibford/qGAN/releases/new?tag=v1.2.4` (draft was provided to user in a previous turn; may already be pasted)
- Plan 14-07 Zenodo DOI mint — **explicitly deferred to journal acceptance**. Rebuttal currently cites `ZENODO-DOI-PLACEHOLDER`; mint real DOI at acceptance and replace placeholder.

**Verification debt (orthogonal to submission, can fix anytime):**
- 6 `human_needed` items in `.planning/phases/13-architecture-introspection/13-VERIFICATION.md` — visual inspection of training_progression / entanglement / param_trajectory figures, REQUIREMENTS.md traceability update, CR-01 dead-code disposition

**For the INCOMING SESSION:**
- The user has stated they have "edits and comments about the current version of the project." These have NOT yet been shared. Be ready to:
  - Receive their feedback (likely against the v1.2.4 PDF)
  - Distinguish "scientific edit" (needs to preserve hard prohibitions §6 + provenance gate §7) from "framing edit" (less constrained) from "typo fix" (trivial)
  - For each edit: apply, run gates (§7), commit atomically with descriptive message, ask whether to bundle into a v1.2.5 tag or wait for more changes

---

## §12 — How to handle user feedback (incoming-session-specific)

**Triage flow:**

1. **Read the user's full edit list before applying anything.** Some edits may interact (e.g., changing the §4.2 contribution ordering may invalidate §4.5 cross-references).
2. **Sort edits into batches:**
   - **Trivial:** typos, single-word polish — batch into one commit
   - **Scientific:** any change to numbers, claims, comparator-set scope, hedges, or mechanism descriptions — needs provenance gate run AND check against §6 hard prohibitions; one atomic commit per logical change
   - **Structural:** section reorder, new subsection, figure insert/remove — needs full compile + page-count check + freeze-ready gate
3. **Quote the user's verbatim feedback in commit messages** so the diff trail explains the *why* not just the *what*.
4. **After all edits land:** decide with the user whether to cut `v1.2.5` (typical for any scientific change) or hold and continue iterating.
5. **If the user asks for changes that contradict §6 hard prohibitions:** push back politely. The prohibitions exist because they encode past corrections; reintroducing them is a regression. Surface the specific historical commit that fixed each one (e.g., "v1.2.2 commit `86ff7ab` corrected the VAE variance-collapse framing at 4 sites; this edit would re-introduce that error").

**If the user asks to "review the current PDF":**
- First verify they're looking at the correct PDF (see §1). 
- If they want a structured re-read, suggest re-running the 4-agent audit pattern from `.planning/review-findings/REVIEW-FINDINGS.md` (~12 min wall time, caught 1 BLOCK + 9 FLAGs in v1.2.2).

---

## §13 — What NOT to do

- ❌ Don't trust pdflatex exit code alone — `grep '^! '` for actual errors (v1.2.1 gate-gap lesson)
- ❌ Don't accept "the PDF looks fine" without re-running provenance gate (v1.2.2 lesson — Lambert W slipped past because numbers were correct in isolation)
- ❌ Don't rewrite published tags (`v1.2` through `v1.2.4` are on origin). New patch tag only.
- ❌ Don't delete `lambert_w_transform` / `inverse_lambert_w_transform` from `core/data.py` — retained for ablation reproducibility
- ❌ Don't delete the historical `~/Desktop/aiche_upload_v1.2.1/` bundle (user explicitly chose to keep it)
- ❌ Don't introduce new numeric literals without a JSON cell to trace them — either find the right JSON path or rephrase qualitatively (e.g., "remaining 1950 evaluation steps" → "remainder of training" because 1950 was a derived literal)
- ❌ Don't claim "quantum advantage" generically — the bifurcated finding is the only positive claim, and it's scope-bounded
- ❌ Don't re-introduce any of the 4 hard prohibitions in §6 even if "more elegant prose" suggests it
- ❌ Don't push tags to origin without checking that all gates PASS first

---

## §14 — Quick orientation cheatsheet

| Question | Answer |
|---|---|
| What's the current tag? | `v1.2.4` |
| What's the current page count? | 60 |
| What's the current figure count? | Main 6, Supp 15 (one is 8-panel grid) = 22 visible panels |
| What's the deadline? | ≈ 2026-06-17 AIChE resubmission (15 days from 2026-06-02) |
| Is the rebuttal letter ready? | Yes — `docs/reviewer_response.md` (520 lines, markdown) |
| Is the AIChE bundle ready? | Yes — `~/Desktop/aiche_upload_v1.2.4.zip` |
| What's left to do externally? | AIChE portal upload, GitHub release notes (draft provided), Zenodo DOI at acceptance |
| What's the WGAN-CNN seed-42 outlier? | OD-EMD = 0.1587 (other 4 seeds 0.020–0.034); disclosed at face value in §4.1 |
| Why isn't VAE in the LR-DTW dominance comparison? | Degenerate generation regime — see §6 prohibition #1 |
| What's the matched-budget protocol? | n=5 seeds {42,43,44,45,46}, 2000 epochs, shared 250881-param critic, Adam (β₁=0.0, β₂=0.9), η_G=6.9173e-5, η_C=1.8046e-5, n_critic=9, λ_GP=2.16, BS=12 |

---

**End of handoff. Ready to receive user edits and apply against `v1.2.4` baseline.**
