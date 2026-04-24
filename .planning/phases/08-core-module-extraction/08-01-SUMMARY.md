---
phase: 08
plan: 01
requirements: [INFRA-01]
status: complete
completed: 2026-04-23
---

# Phase 8 Plan 01: Core Package Scaffold — Summary

Empty-but-importable `revision/core/` package with signature stubs and v1.1 HPO-tuned constants, ready for Wave 2 parallel fill-in.

## Package Structure Created

```
revision/
├── __init__.py
├── core/
│   ├── __init__.py          # exposes constants + submodules
│   ├── data.py              # data-pipeline stubs
│   ├── eval.py              # metric stubs
│   ├── training.py          # WGAN-GP training stub
│   └── models/
│       ├── __init__.py
│       ├── quantum.py       # QuantumGenerator stub
│       └── critic.py        # Critic stub
├── docs/.gitkeep
└── results/.gitkeep         # force-added past gitignore
```

## Signatures Stubbed (all bodies raise NotImplementedError)

- `data.py`: normalize, denormalize, compute_log_delta, lambert_w_transform, inverse_lambert_w_transform, rolling_window, rescale, full_denorm_pipeline, load_and_preprocess (9 fns)
- `eval.py`: compute_emd, compute_moments, compute_acf, compute_dtw, compute_jsd, compute_psd, full_metric_suite (7 fns)
- `models/quantum.py`: class `QuantumGenerator(nn.Module)` with __init__, count_params, encoding_layer, generator_circuit, forward
- `models/critic.py`: class `Critic(nn.Module)` with __init__, forward
- `training.py`: compute_gradient_penalty, train_wgan_gp (HPO defaults in signature)

## HPO Constants (verified match v1.1 Phase 4)

N_CRITIC=9, LAMBDA=2.16, LR_CRITIC=1.8046e-05, LR_GENERATOR=6.9173e-05, NOISE_HIGH=4π.

## Commits

- `a1d9476` chore(08-01): scaffold revision/core package with HPO constants
- `4cfdb50` feat(08-01): add signature stubs for all five core modules

## Verification Passed

- `python -c "from revision.core import data, eval, training; from revision.core.models import quantum, critic"` → 0
- HPO constants and `NOISE_HIGH = 4π` assertions → 0

## Next

08-02 (data + eval fill-in) and 08-03 (models) can now run in parallel in Wave 2.
