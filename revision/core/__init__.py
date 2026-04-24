"""revision.core — shared logic extracted from qgan_pennylane.ipynb.

All functions and classes in this subpackage must preserve v1.1 behavior.
No business logic in notebooks: notebooks import from here, orchestrate,
plot, and write JSON to revision/results/.
"""

import math

# HPO-tuned hyperparameter defaults (v1.1 Phase 4 — DO NOT change)
N_CRITIC = 9
LAMBDA = 2.16
LR_CRITIC = 1.8046e-05
LR_GENERATOR = 6.9173e-05

# Architecture constants (v1.0/v1.1 — DO NOT change)
NUM_QUBITS = 5
NUM_LAYERS = 4
WINDOW_LENGTH = 2 * NUM_QUBITS  # 10
NUM_EPOCHS = 2000
BATCH_SIZE = 12
GEN_SCALE = 1.0
EVAL_EVERY = 10
DROPOUT_RATE = 0.2

# Data / preprocessing constants
DITHER = 0.005
DITHER_SEED = 42
PAR_LIGHT_MAX = 12.5

# Noise range (v1.1 Phase 4 — [0, 4π] NOT [0, 2π])
NOISE_LOW = 0.0
NOISE_HIGH = 4 * math.pi

from revision.core import data, eval, training  # noqa: F401,E402
from revision.core import models  # noqa: F401,E402

__all__ = [
    "data", "eval", "training", "models",
    "N_CRITIC", "LAMBDA", "LR_CRITIC", "LR_GENERATOR",
    "NUM_QUBITS", "NUM_LAYERS", "WINDOW_LENGTH",
    "NUM_EPOCHS", "BATCH_SIZE", "GEN_SCALE", "EVAL_EVERY",
    "DROPOUT_RATE", "DITHER", "DITHER_SEED", "PAR_LIGHT_MAX",
    "NOISE_LOW", "NOISE_HIGH",
]
