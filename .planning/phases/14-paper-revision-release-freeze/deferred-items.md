# Phase 14 — Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Test (env-only, out-of-scope) | `test_utility.py::test_sample_shape_invariant[wgan_mlp-B-42]` fails with `FileNotFoundError` for `results/baselines/runs/wgan_mlp/B/42/samples.npy` — a gitignored frozen Phase-10 runtime artifact NOT copied into the parallel-execution worktree. Not caused by 14-01 changes (scope boundary: pre-existing, unrelated frozen-artifact territory). The `@_skip_no_frozen` guard does not fully cover the partially-present baselines tree in worktree mode. | Open | 14-01 (parallel-executor worktree env) |
