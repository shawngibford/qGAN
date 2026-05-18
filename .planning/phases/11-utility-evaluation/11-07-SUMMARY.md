---
phase: 11-utility-evaluation
plan: 07
subsystem: testing
tags: [gap-closure, wr-05, wr-06, run-timegan-scores, determinism, rng, shape-contract, eval-03]

# Dependency graph
requires:
  - phase: 11-utility-evaluation
    provides: "11-02 run_timegan_scores.py (EVAL-03 predictive/discriminative driver)"
provides:
  - "revision/run_timegan_scores.py — discriminative_score driven by a single explicit np.random.default_rng(seed) for splits + minibatches, with an enforced logits/labels shape contract"
affects: [11-08, 14-paper-revision]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single explicit Generator threaded through all stochastic draws (no legacy global np.random)"
    - "Order-explicit, documented load-bearing RNG consumption"
    - "Shape-contract assertion before reduction to prevent silent broadcast"

key-files:
  created: []
  modified:
    - revision/run_timegan_scores.py

key-decisions:
  - "logits.squeeze(-1) used unconditionally (torch squeeze only collapses size-1 dims, so it is a safe no-op when output is already (B,)) then assert logits.shape == yte.shape"

patterns-established:
  - "_split takes the Generator as an explicit parameter; split call order documented as load-bearing"

requirements-completed: [EVAL-03]

# Metrics
duration: ~7min
completed: 2026-05-18
---

# Phase 11 Plan 07: WR-05 Single-Generator discriminative_score

**Replaced the mixed legacy-global / Generator RNG pattern in `discriminative_score` with a single explicit `np.random.default_rng(seed)` for both 80/20 splits and minibatch draws, and added a logits-vs-labels shape contract — making the discriminative path deterministic and broadcast-safe (the 11-08 test proves it).**

## Performance

- **Duration:** ~7 min (inline orchestrator execution — subagents were Bash-denied this session)
- **Started:** 2026-05-18 (base 1f1c186)
- **Completed:** 2026-05-18
- **Tasks:** 1
- **Files modified:** 1 (`revision/run_timegan_scores.py`)

## Accomplishments

- **WR-05 closed:** Removed `np.random.seed(seed)`. One `g = np.random.default_rng(seed)` is constructed after `torch.manual_seed(seed)` and reused for both `_split(rw, g)` / `_split(sw, g)` (now `g.permutation(n)`) and the minibatch `g.integers(0, n, ...)` draw — no second Generator. The two splits are no longer correlated through global-RNG order and are immune to upstream global-RNG mutation / call reordering. Split order documented as load-bearing.
- **Shape contract (WR-06 prep):** `logits = logits.squeeze(-1)` then `assert logits.shape == yte.shape, (logits.shape, yte.shape)` before the accuracy computation, so a `(B,1)` vs `(B,)` broadcast can never silently produce a meaningless `(B,B)` accuracy.
- **Untouched:** `DiscriminativeGRU`, loss/optimizer, `iters`/`bs` defaults, the `abs(0.5 - test_accuracy)` return formula, and `revision/core/` — all unchanged. The recorded JSON `seed` metadata value is unchanged (only RNG plumbing changed).
- **Invariants held:** `git diff --stat -- revision/core/` empty; `predictive_discriminative.json` `data_hash` still `91e447d4624e25b3`; `pytest revision/tests/test_timegan_scores.py -q` → 4 passed; `pytest revision/tests/ -q` → 22 passed.

## Task Commits

1. **Task 1: single Generator for splits + minibatches; logits/labels shape assertion** - `ba084f6` (fix)

## Files Created/Modified

- `revision/run_timegan_scores.py` - `discriminative_score`: single explicit `np.random.default_rng(seed)` for splits + minibatches; `_split` takes the Generator; documented load-bearing split order; `logits.squeeze(-1)` + shape assertion before accuracy.

## Decisions Made

- **Unconditional `logits.squeeze(-1)`:** torch `squeeze(-1)` only collapses a size-1 trailing dim, so it is a safe no-op when the GRU already returns `(B,)`. Simpler than a conditional `if logits.dim() > 1 and logits.shape[-1] == 1` and behaviourally identical for the contract.

## Deviations from Plan

- None. RNG plumbing and shape contract implemented exactly as the plan's action specified.

## Self-Check: PASSED

- `! grep -nE 'np\.random\.seed\('` → no match ✓
- `grep 'g = np.random.default_rng(seed)'` → present (line 278) ✓
- `_split` uses `g.permutation`; minibatch uses `g.integers` ✓
- `assert logits.shape == yte.shape` present (line 329) ✓
- `git diff --stat -- revision/core/` → empty ✓
- `predictive_discriminative.json` data_hash `91e447d4624e25b3` ✓
- `pytest revision/tests/test_timegan_scores.py -q` → 4 passed ✓
- `pytest revision/tests/ -q` → 22 passed ✓
