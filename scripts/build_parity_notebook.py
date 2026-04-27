"""Build revision/01_parity_check.ipynb programmatically via nbformat.

Phase 8 Plan 5 (INFRA-02): proves the revision/core/* extraction reproduces
the qgan_pennylane.ipynb baseline within tolerance (EMD <= 1e-4, moments <= 1e-6).

This script writes the notebook only — it does NOT execute it. Execution is
done by `jupyter nbconvert --to notebook --execute` after this script runs.
"""
from pathlib import Path
import nbformat


REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / "revision" / "01_parity_check.ipynb"


CELL_TITLE_MD = """\
# Phase 8 Parity Check (INFRA-02)

Verifies `revision/core/` modules reproduce `qgan_pennylane.ipynb` behavior
within tolerance:

- **EMD**: |EMD_pre - EMD_post| <= 1e-4
- **Moments** (mean, std, kurtosis): |delta| <= 1e-6

Does **not** retrain. Loads an existing checkpoint
(`best_checkpoint_par_conditioned.pt`, 75 PQC params matching the v1.1
4-layer architecture), runs a deterministic forward pass through both the
inline notebook code path and the extracted-module code path, and compares
metrics.

Output: `revision/results/parity_check.json` with `pass: true`.
"""


CELL_IMPORTS_CODE = """\
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pennylane as qml
from scipy.special import lambertw
from scipy.optimize import minimize_scalar
from scipy.stats import kurtosis as sp_kurtosis, wasserstein_distance

warnings.filterwarnings("ignore")

# nbconvert sets CWD to the notebook's directory. For this notebook to find
# data.csv, best_checkpoint*.pt, and the revision.core package on sys.path,
# walk upward to the repo root (the directory containing data.csv + revision/).
def _find_repo_root():
    here = Path.cwd().resolve()
    for d in [here, *here.parents]:
        if (d / "data.csv").exists() and (d / "revision" / "core").is_dir():
            return d
    raise FileNotFoundError(
        "Could not locate repo root from " + str(here) +
        " (looked for data.csv + revision/core)"
    )


REPO_ROOT = _find_repo_root()
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
print(f"Repo root: {REPO_ROOT}")
print(f"CWD now: {os.getcwd()}")

# Deterministic seed shared across both paths.
SEED = 42

# Pick checkpoint: prefer best_checkpoint_par_conditioned.pt because its 75
# PQC params match the v1.1 4-layer architecture used by the extracted
# QuantumGenerator. Fall back to best_checkpoint.pt only if the preferred one
# is unavailable.
CHECKPOINT = "best_checkpoint_par_conditioned.pt"
if not Path(CHECKPOINT).exists():
    CHECKPOINT = "best_checkpoint.pt"
assert Path(CHECKPOINT).exists(), f"Missing checkpoint {CHECKPOINT}"
print(f"Using checkpoint: {CHECKPOINT}")

# Architecture constants — must match cell 28 of qgan_pennylane.ipynb.
NUM_QUBITS = 5
WINDOW_LENGTH = 2 * NUM_QUBITS

# Choose num_layers based on checkpoint param count (5 + 3*L*5 + 10 = 75 -> L=4).
ckpt_for_shape = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
n_pqc = ckpt_for_shape["params_pqc"].numel()
NUM_LAYERS = (n_pqc - NUM_QUBITS - 2 * NUM_QUBITS) // (3 * NUM_QUBITS)
assert NUM_QUBITS + NUM_LAYERS * (3 * NUM_QUBITS) + 2 * NUM_QUBITS == n_pqc, (
    f"Checkpoint param shape ({n_pqc}) does not fit "
    f"NUM_QUBITS={NUM_QUBITS}, NUM_LAYERS={NUM_LAYERS}"
)
print(f"Detected NUM_LAYERS={NUM_LAYERS} from checkpoint ({n_pqc} PQC params)")

# Number of fake windows generated for metric computation. 500 is enough to
# get stable distributional statistics while keeping wall time small.
NUM_FAKE_WINDOWS = 500
"""


CELL_PATHA_HEADER_MD = """\
## Path A — Inline (pre-extraction baseline)

Verbatim re-implementation of the relevant `qgan_pennylane.ipynb` code
WITHOUT importing from `revision.core`. Mirrors:

- Cell 5: CSV load + OD/PAR_LIGHT extraction
- Cell 7: `normalize`, `compute_log_delta`
- Cell 9: log-delta with `DITHER=0.005`, `DITHER_SEED=42`
- Cell 17: `inverse_lambert_w_transform`
- Cell 18: optimal Lambert delta
- Cell 26: PQC generator circuit body
- Cell 65: EMD + moments

Loads `best_checkpoint_par_conditioned.pt`, generates `NUM_FAKE_WINDOWS`
windows with a deterministic seed, computes EMD + moments inline.
"""


CELL_PATHA_CODE = """\
# ---- Inline preprocessing (verbatim from cells 5/7/9/17/18) ----------------
def _inline_normalize(data):
    mu = torch.mean(data)
    sigma = torch.std(data)
    return (data - mu) / sigma, mu, sigma


def _inline_compute_log_delta(od_values, dither=0.0, rng=None):
    od_np = od_values.numpy() if isinstance(od_values, torch.Tensor) else od_values.copy()
    if dither > 0:
        if rng is None:
            rng = np.random.default_rng()
        od_np = od_np + rng.uniform(-dither, dither, size=len(od_np))
    log_od = np.log(od_np)
    return torch.tensor(log_od[1:] - log_od[:-1], dtype=torch.float32)


def _inline_inverse_lambert_w_transform(data, delta):
    data = data.double()
    sign = torch.sign(data)
    data_squared = data ** 2
    lambert_input = (delta * data_squared).cpu().numpy()
    lambert_result = lambertw(lambert_input).real
    lambert_tensor = torch.tensor(lambert_result, dtype=torch.float64, device=data.device)
    transformed_data = sign * torch.sqrt(lambert_tensor / delta)
    return transformed_data


def _inline_kurtosis_for_delta(d, data):
    sign = np.sign(data)
    lr = lambertw(d * data ** 2).real
    lr = np.maximum(lr, 0)
    transformed = sign * np.sqrt(lr / d)
    return abs(sp_kurtosis(transformed, fisher=True))


# Cell 5: load CSV
_full_data = pd.read_csv("./data.csv")
_raw = _full_data[["OD"]].copy()
_raw.columns = ["value"]
_raw["value"] = pd.to_numeric(_raw["value"], errors="coerce")
_raw["value"] = _raw["value"].fillna(_raw["value"].rolling(window=10, min_periods=10).mean())
_raw = _raw.dropna()
_OD_A = torch.tensor(_raw["value"].values, dtype=torch.float32)

# Cell 9: log-delta with dither
_DITHER = 0.005
_DITHER_SEED = 42
_log_delta_A = _inline_compute_log_delta(_OD_A, dither=_DITHER, rng=np.random.default_rng(_DITHER_SEED))

# Cell 15: normalize
_norm_log_delta_A, _mu_A, _sigma_A = _inline_normalize(_log_delta_A)

# Cell 18: optimal Lambert delta
_normed_np_A = _norm_log_delta_A.numpy()
_result_A = minimize_scalar(_inline_kurtosis_for_delta, bounds=(0.01, 2.0),
                             args=(_normed_np_A,), method="bounded")
_delta_A = float(_result_A.x)
print(f"[Path A] Lambert delta: {_delta_A:.6f}")
print(f"[Path A] mu={_mu_A.item():.6f} sigma={_sigma_A.item():.6f}")

# ---- Inline PQC generator (verbatim from cell 26 generator_circuit) --------
_dev_A = qml.device("default.qubit", wires=NUM_QUBITS, shots=None)


def _inline_generator_circuit(noise_params, params_pqc):
    idx = 0
    # Step 1: Hadamard
    for q in range(NUM_QUBITS):
        qml.Hadamard(wires=q)
    # Step 2: IQP encoding
    for q in range(NUM_QUBITS):
        if idx < len(params_pqc):
            qml.RZ(phi=params_pqc[idx], wires=q)
            idx += 1
    # Step 3: noise encoding (IQP-style RZ)
    for i in range(min(noise_params.shape[0] if noise_params.dim() >= 1 else len(noise_params), NUM_QUBITS)):
        qml.RZ(noise_params[i], wires=i)
    # Step 4: strongly entangled layers
    for layer in range(NUM_LAYERS):
        for q in range(NUM_QUBITS):
            if idx + 2 < len(params_pqc):
                qml.Rot(phi=params_pqc[idx],
                        theta=params_pqc[idx + 1],
                        omega=params_pqc[idx + 2],
                        wires=q)
                idx += 3
        if NUM_QUBITS > 1:
            range_param = (layer % (NUM_QUBITS - 1)) + 1
            for q in range(NUM_QUBITS):
                target = (q + range_param) % NUM_QUBITS
                qml.CNOT(wires=[q, target])
    # Step 5: final RX/RY
    for q in range(NUM_QUBITS):
        if idx + 1 < len(params_pqc):
            qml.RX(phi=params_pqc[idx], wires=q)
            idx += 1
            qml.RY(phi=params_pqc[idx], wires=q)
            idx += 1
    measurements = []
    for i in range(NUM_QUBITS):
        measurements.append(qml.expval(qml.PauliX(i)))
        measurements.append(qml.expval(qml.PauliZ(i)))
    return (*measurements,)


_qnode_A = qml.QNode(_inline_generator_circuit, _dev_A,
                     interface="torch", diff_method="backprop")

# Load checkpoint and bind params
_ckpt_A = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
_params_A = _ckpt_A["params_pqc"].clone().detach()
print(f"[Path A] Loaded params_pqc shape={tuple(_params_A.shape)}")

# Generate NUM_FAKE_WINDOWS windows with batched QNode call (cell 26 style).
torch.manual_seed(SEED)
np.random.seed(SEED)
_noise_A = torch.tensor(
    np.random.uniform(0, 4 * np.pi, size=(NUM_QUBITS, NUM_FAKE_WINDOWS)),
    dtype=torch.float32,
)
with torch.no_grad():
    _results_A = _qnode_A(_noise_A, _params_A)
    _stacked_A = torch.stack(list(_results_A))           # (window_length, batch)
    _gen_windows_A = _stacked_A.T                        # (batch, window_length)

# Flatten generator output for direct comparison with log_delta distribution.
_fake_samples_A = _gen_windows_A.detach().cpu().numpy().ravel()
_log_delta_np_A = _log_delta_A.detach().cpu().numpy()

# Cell 65: EMD + moments (np.std default ddof=0, scipy kurtosis Fisher).
_emd_A = float(wasserstein_distance(_log_delta_np_A, _fake_samples_A))
_mean_A = float(np.mean(_fake_samples_A))
_std_A = float(np.std(_fake_samples_A))
_kurt_A = float(sp_kurtosis(_fake_samples_A))

pre_metrics = {
    "emd": _emd_A,
    "mean": _mean_A,
    "std": _std_A,
    "kurtosis": _kurt_A,
}
print(f"[Path A] pre_metrics = {pre_metrics}")
"""


CELL_PATHB_HEADER_MD = """\
## Path B — Extracted modules (post-extraction)

Identical computation via imports from `revision.core`. Same seed, same
checkpoint, same noise sample, same number of windows. The objective is
that `pre_metrics` (Path A) and `post_metrics` (Path B) match within the
locked tolerance.
"""


CELL_PATHB_CODE = """\
from revision.core.data import load_and_preprocess
from revision.core.eval import compute_emd, compute_moments
from revision.core.models.quantum import QuantumGenerator
from revision.core.models.critic import Critic  # noqa: F401  (verifies import works)
from revision.core import (
    NUM_QUBITS as MOD_NUM_QUBITS,
    WINDOW_LENGTH as MOD_WINDOW_LENGTH,
    DROPOUT_RATE,
)

assert MOD_NUM_QUBITS == NUM_QUBITS, "module NUM_QUBITS mismatch"
assert MOD_WINDOW_LENGTH == WINDOW_LENGTH, "module WINDOW_LENGTH mismatch"

# Re-run preprocessing through the module (must produce same log_delta).
d = load_and_preprocess("./data.csv")
log_delta_B = d["log_delta"]
log_delta_np_B = log_delta_B.detach().cpu().numpy()

# Build extracted generator with the same num_layers as Path A.
gen = QuantumGenerator(
    num_qubits=NUM_QUBITS,
    num_layers=NUM_LAYERS,
    window_length=WINDOW_LENGTH,
)
ckpt_B = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
gen.params_pqc.data.copy_(ckpt_B["params_pqc"])
print(f"[Path B] Loaded params_pqc shape={tuple(gen.params_pqc.shape)}, "
      f"num_layers={NUM_LAYERS}")

# Re-seed and re-build noise IDENTICALLY to Path A so the two QNodes get the
# same input.
torch.manual_seed(SEED)
np.random.seed(SEED)
noise_B = torch.tensor(
    np.random.uniform(0, 4 * np.pi, size=(NUM_QUBITS, NUM_FAKE_WINDOWS)),
    dtype=torch.float32,
)

with torch.no_grad():
    gen_windows_B = gen(noise_B)            # (NUM_FAKE_WINDOWS, WINDOW_LENGTH)

fake_samples_B = gen_windows_B.detach().cpu().numpy().ravel()

# Compute via extracted eval module.
emd_B = compute_emd(log_delta_np_B, fake_samples_B)
m_B = compute_moments(fake_samples_B)
post_metrics = {
    "emd": emd_B,
    "mean": m_B["mean"],
    "std": m_B["std"],
    "kurtosis": m_B["kurtosis"],
}
print(f"[Path B] post_metrics = {post_metrics}")
"""


CELL_COMPARE_HEADER_MD = """\
## Comparison + Artifact

Compute deltas, decide pass/fail against the locked tolerance, and write
`revision/results/parity_check.json`. Tolerances are LOCKED by 08-CONTEXT.md:

- `emd`: 1e-4
- `mean` / `std` / `kurtosis`: 1e-6
"""


CELL_COMPARE_CODE = """\
tolerance = {"emd": 1e-4, "mean": 1e-6, "std": 1e-6, "kurtosis": 1e-6}
delta = {k: abs(post_metrics[k] - pre_metrics[k]) for k in tolerance}
passed = all(delta[k] <= tolerance[k] for k in tolerance)


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


_sha = _git_sha()

artifact = {
    "pre": pre_metrics,
    "post": post_metrics,
    "delta": delta,
    "pass": bool(passed),
    "tolerance": tolerance,
    "seed": SEED,
    "git_sha_pre": _sha,
    "git_sha_post": _sha,
    "checkpoint": CHECKPOINT,
    "num_fake_windows": NUM_FAKE_WINDOWS,
    "num_qubits": NUM_QUBITS,
    "num_layers": NUM_LAYERS,
    "notes": (
        "Phase 8 INFRA-02 parity check: inline notebook path vs "
        "revision/core extracted modules. Both paths load the same "
        "checkpoint and run identical seeded forward passes; the "
        "delta measures any numerical drift introduced by the "
        "refactor."
    ),
}

out = Path("revision/results/parity_check.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(artifact, indent=2))
print(json.dumps(artifact, indent=2))

assert passed, (
    f"Parity FAILED: deltas exceed tolerance. delta={delta} tolerance={tolerance}"
)
print("Parity PASSED")
"""


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(CELL_TITLE_MD),
        nbformat.v4.new_code_cell(CELL_IMPORTS_CODE),
        nbformat.v4.new_markdown_cell(CELL_PATHA_HEADER_MD),
        nbformat.v4.new_code_cell(CELL_PATHA_CODE),
        nbformat.v4.new_markdown_cell(CELL_PATHB_HEADER_MD),
        nbformat.v4.new_code_cell(CELL_PATHB_CODE),
        nbformat.v4.new_markdown_cell(CELL_COMPARE_HEADER_MD),
        nbformat.v4.new_code_cell(CELL_COMPARE_CODE),
    ]
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python", "mimetype": "text/x-python"},
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NB_PATH.open("w") as f:
        nbformat.write(nb, f)
    print(f"Wrote {NB_PATH}")


if __name__ == "__main__":
    main()
