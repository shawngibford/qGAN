"""Phase 14 plan 14-11 Task 3 — methods_full.json aggregator.

Pure aggregator emitting ``revision/results/methods_full.json`` — the
paper-ready Methods bucket JSON that ``revision/docs/methods_full.md`` (Task
4) renders from. The five buckets are:

    1_dataset           — verbatim from model_info.json.dataset
    2_models            — per-model family + n_params + architecture + objective
    3_training          — Adam betas, LR, n_critic, lambda_gp, batch, epochs
    4_hardware_software — device + DISTINCT dtype_params/dtype_samples + backend
    5_reproducibility   — seeds + determinism contract + verbatim rerun template

Pure aggregator constraints (D-14-16, mirrors 14-03 / 14-09 / 14-10):
  - Zero deep-learning-framework imports (no torch, no pennylane).
  - Zero ``revision.core`` package imports.
  - ``revision/core/training.py`` is opened ONLY via ``path.read_text()`` for
    grep-based file:line citation extraction — never executed as a module.
  - ``revision/run_matched2000.py`` is opened ONLY via ``path.read_text()`` for
    a verbatim docstring slice into ``buckets.5_reproducibility.rerun_command_template``.
  - Cross-artifact ``data_hash`` gate: explicit raise AssertionError (python -O
    safe — run_multiseed_rollup.py:86-92 idiom) if any consumed JSON carrying
    a ``data_hash`` field disagrees with the expected ``91e447d4624e25b3``.

The ``dtype_params`` (torch.float32 — the trainable nn.Parameter dtype) vs
``dtype_samples`` (torch.float64 on the CPU/CUDA path, falling back to
float32 on MPS — sample-generation pipeline) split is recorded as TWO
DISTINCT fields in ``buckets.4_hardware_software``; they MUST NOT be merged.
This resolves the documented contradiction § 6(b) of methods_full.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Loud-fail guard: this aggregator is pure-aggregator-only. The presence of
# any unexpected deep-learning import would be a contract violation;
# importing yaml is the only non-stdlib top-level allowance (mirrors
# run_model_info.py's idiom, used for config.yaml multi-line scalars; THIS
# script does not actually read YAML, but the import is harmless and keeps
# the peer-driver shape).

EXPECTED_DATA_HASH = "91e447d4624e25b3"


# ─────────────────────────────────────────────────────────────────────────────
# Repo-root resolution (verbatim run_model_info.py:69-81 idiom).
# ─────────────────────────────────────────────────────────────────────────────
def _find_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "revision" / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError(
        "repo root not found (anchor file revision-slash-core-slash-"
        "preprocessing.py missing)"
    )


REPO = _find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RESULTS = REPO / "revision/results"
DOCS = REPO / "revision/docs"


# ─────────────────────────────────────────────────────────────────────────────
# Loud-fail loader (run_figure_suite.py:144-174 idiom).
# ─────────────────────────────────────────────────────────────────────────────
def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"[run_methods_full] required aggregator artifact missing: "
            f"{path} ({what}). This driver is pure-aggregator (no model-fit, "
            f"no sampling, no checkpoint-reload, no recompute) — regenerate "
            f"the upstream JSON before re-running."
        )
    return path


def _load_json(path: Path, what: str) -> dict:
    _require(path, what)
    return json.loads(path.read_text())


# ─────────────────────────────────────────────────────────────────────────────
# Text-only file:line citation extractor — opens revision/core/training.py
# via path.read_text() and finds the first line matching each target pattern.
# Never executes training.py as a Python module.
# ─────────────────────────────────────────────────────────────────────────────
def _first_lineno(src: str, pattern: str) -> int | None:
    """First line whose STRIPPED prefix matches ``pattern`` exactly (the
    line, after lstrip, starts with ``pattern``). Substring matching would
    confuse ``random.seed(seed)`` with ``np.random.seed(seed)``.
    """
    for i, line in enumerate(src.splitlines(), start=1):
        if line.lstrip().startswith(pattern):
            return i
    return None


def _citations(training_src: str) -> dict[str, str]:
    """Return {label: 'revision/core/training.py:N'} for each grep target.

    The exact target patterns mirror the plan's <interfaces> block — each is
    a leading-token pattern matched against the STRIPPED line prefix (so
    ``random.seed`` does not collide with the ``np.random.seed`` line).
    """
    targets = {
        "manual_seed": "torch.manual_seed(seed)",
        "numpy_seed": "np.random.seed(seed)",
        "random_seed": "random.seed(seed)",
        "cuda_seed": "torch.cuda.manual_seed_all(seed)",
        "compute_dtype_split": "compute_dtype = torch.float32 if device.type",
        "critic_loss": "critic_loss = fake_score_mean - real_score_mean",
        "generator_loss": "generator_loss = -torch.mean(fake_scores)",
        "gradient_penalty": "gp = ((gradients.norm(2, dim=1) - 1)",
        # Plan 14-13 Task 4 (CR-3) — replace hardcoded training.py:347 +
        # training.py:259-268 literals with programmatic _first_lineno
        # citations. `generator_to_compute_dtype` resolves the cast site
        # (`generated_samples = generator(noise_batch)` — the immediately-
        # preceding line of the `.to(compute_dtype) * 0.1` chain).
        # `mps_dtype_block` resolves the MPS-vs-CPU dtype-split line
        # (`compute_dtype = torch.float32 if device.type == "mps" else torch.float64`).
        "generator_to_compute_dtype": "generated_samples = generated_samples.to(compute_dtype)",
        "mps_dtype_block": (
            "compute_dtype = torch.float32 if device.type == \"mps\""
        ),
    }
    out: dict[str, str] = {}
    for label, pat in targets.items():
        ln = _first_lineno(training_src, pat)
        if ln is None:
            raise AssertionError(
                f"[run_methods_full] grep target missing in "
                f"revision/core/training.py: label={label} pattern={pat!r} — "
                f"training.py may have been refactored; regenerate "
                f"methods_full.json on every emitter run so citations cannot "
                f"go stale (see plan 14-11 T-14-29)."
            )
        out[label] = f"revision/core/training.py:{ln}"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Docstring slicer for run_matched2000.py:1-69 — captures the leading module
# docstring VERBATIM, between the first triple-quote and the second.
# ─────────────────────────────────────────────────────────────────────────────
def _slice_module_docstring(src: str) -> str:
    QUOTE = '"' * 3
    first = src.find(QUOTE)
    if first < 0:
        raise AssertionError(
            "[run_methods_full] run_matched2000.py has no leading triple-"
            "quoted module docstring — refusing to emit a paraphrased "
            "rerun_command_template (T-14-32)."
        )
    second = src.find(QUOTE, first + len(QUOTE))
    if second < 0:
        raise AssertionError(
            "[run_methods_full] run_matched2000.py module docstring is not "
            "closed — refusing to emit a partial rerun_command_template "
            "(T-14-32)."
        )
    # Inclusive of the opening triple-quote, exclusive of the closing one,
    # but we want the docstring TEXT only (between the two triple-quotes).
    return src[first + len(QUOTE) : second]


# ─────────────────────────────────────────────────────────────────────────────
# Cross-artifact data_hash gate (run_model_info.py:638-649 idiom). Collect
# data_hash from every consumed JSON that CARRIES one and assert mutual
# equality (== EXPECTED_DATA_HASH). The classical_architectures.json and
# framework_versions.json carry NO data_hash (introspection artifacts), so
# they are excluded from the gate.
# ─────────────────────────────────────────────────────────────────────────────
def _data_hash_gate(consumed: dict[str, dict]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, blob in consumed.items():
        h = blob.get("data_hash") if isinstance(blob, dict) else None
        if isinstance(h, str):
            hashes[name] = h
    if hashes and set(hashes.values()) != {EXPECTED_DATA_HASH}:
        raise AssertionError(
            f"[run_methods_full] data_hash mismatch across consumed "
            f"artifacts: {hashes} (expected {EXPECTED_DATA_HASH!r} — see "
            f"plan 14-11 T-14-28)."
        )
    return hashes


def _arch_quantum(lock: dict, model_rec: dict | None) -> dict:
    """Build the per-model architecture block for a quantum entry from a
    config-lock JSON + (optionally) its model_info.json row."""
    decomp = lock.get("decomposition", {})
    gate = decomp.get("gate_layout", {})
    arch: dict = {
        "source": f"revision/results/{lock.get('_lock_relpath', '?')}",
        "num_qubits": decomp.get("num_qubits"),
        "num_layers": decomp.get("num_layers"),
        "topology": gate.get("entangler", lock.get("topology")),
        "encoding": "IQP (Hadamard + RZ per qubit)",
        "variational_block": "SEL (Rot(phi,theta,omega) per qubit per layer)",
        "final_rotation": gate.get("final_rotation"),
        "n_params": decomp.get("param_count", lock.get("param_count")),
        "circuit_id": lock.get("locked_circuit_id"),
    }
    if "checkpoint_epoch" in lock:
        arch["checkpoint_epoch"] = lock["checkpoint_epoch"]
    return arch


def main() -> None:
    # ── Load every consumed JSON (loud-fail on absence) ────────────────
    model_info = _load_json(
        RESULTS / "model_info.json", "per-model registry"
    )
    canonical = _load_json(
        RESULTS / "canonical_config_lock.json", "iqp_sel_55 lock"
    )
    canonical["_lock_relpath"] = "canonical_config_lock.json"
    default_75 = _load_json(
        RESULTS / "default_75_config_lock.json", "default_75 lock"
    )
    default_75["_lock_relpath"] = "default_75_config_lock.json"
    v1_lock = _load_json(RESULTS / "v1_config_lock.json", "V1 lock")
    v1_lock["_lock_relpath"] = "v1_config_lock.json"
    v2_lock = _load_json(RESULTS / "v2_config_lock.json", "V2 lock")
    v2_lock["_lock_relpath"] = "v2_config_lock.json"
    v3_lock = _load_json(RESULTS / "v3_config_lock.json", "V3 lock")
    v3_lock["_lock_relpath"] = "v3_config_lock.json"
    classical = _load_json(
        RESULTS / "classical_architectures.json",
        "classical layer-tree extract (plan 14-11 Task 1)",
    )
    framework = _load_json(
        RESULTS / "framework_versions.json",
        "framework pin (plan 14-11 Task 2)",
    )

    # ── Cross-artifact data_hash gate ─────────────────────────────────
    consumed_with_hash = {
        "model_info.json": model_info,
        "canonical_config_lock.json": canonical,
        "default_75_config_lock.json": default_75,
        "v1_config_lock.json": v1_lock,
        "v2_config_lock.json": v2_lock,
        "v3_config_lock.json": v3_lock,
    }
    hashes = _data_hash_gate(consumed_with_hash)

    # ── Text-only citation extraction over revision/core/training.py ──
    training_path = REPO / "revision/core/training.py"
    training_src = _require(
        training_path, "text-only citation source"
    ).read_text()
    cits = _citations(training_src)

    # ── Verbatim docstring slice from run_matched2000.py:1-69 ─────────
    matched_path = REPO / "revision/run_matched2000.py"
    matched_src = _require(
        matched_path, "verbatim rerun-template source"
    ).read_text()
    rerun_template = _slice_module_docstring(matched_src)

    # ── Build per-model index from model_info.json ─────────────────────
    mi_by_model = {m["model"]: m for m in model_info.get("models", [])}

    # ── LaTeX equation strings (rendered VERBATIM in the doc) ──────────
    # Sourced from the canonical training.py expressions + standard texts:
    #   WGAN-GP critic:    L_C = E_g[C(g)] - E_r[C(r)] + lambda_gp * E[(||grad||_2 - 1)^2]
    #   WGAN-GP generator: L_G = -E_g[C(g)]
    #   VAE ELBO:          L_ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
    #   AR(p):             x_t = sum_k phi_k x_{t-k} + eps_t, eps ~ N(0,sigma^2)
    WGAN_CRITIC_LATEX = (
        r"L_C = \mathbb{E}_{\tilde x \sim P_g}[C(\tilde x)] "
        r"- \mathbb{E}_{x \sim P_r}[C(x)] "
        r"+ \lambda_{gp}\, \mathbb{E}_{\hat x \sim P_{\hat x}}"
        r"\big[(\|\nabla_{\hat x} C(\hat x)\|_2 - 1)^2\big]"
    )
    WGAN_GEN_LATEX = (
        r"L_G = - \mathbb{E}_{\tilde x \sim P_g}[C(\tilde x)]"
    )
    VAE_ELBO_LATEX = (
        r"\mathcal{L}_{ELBO}(x) = "
        r"\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] "
        r"- D_{KL}(q_\phi(z|x) \| p(z))"
    )
    AR_LATEX = (
        r"x_t = \sum_{k=1}^{p} \phi_k\, x_{t-k} + \varepsilon_t,\quad "
        r"\varepsilon_t \sim \mathcal{N}(0, \sigma^2);\quad "
        r"\hat\phi = \arg\min_\phi \|X\phi - y\|_2^2"
    )

    # Shared WGAN-GP objective citation block (used by 8 of 10 models).
    wgan_obj = {
        "name": "WGAN-GP (Wasserstein with two-sided gradient penalty)",
        "source_path": (
            f"{cits['critic_loss']} (critic loss) + "
            f"{cits['generator_loss']} (generator loss) + "
            f"{cits['gradient_penalty']} (gradient penalty)"
        ),
        "equation_latex_critic": WGAN_CRITIC_LATEX,
        "equation_latex_generator": WGAN_GEN_LATEX,
    }
    vae_obj = {
        "name": "ELBO (variational autoencoder)",
        "source_path": (
            "revision/run_baselines.py (ELBO training loop — NOT in "
            "revision/core/training.py which is WGAN-GP only; D-10-13)"
        ),
        "equation_latex": VAE_ELBO_LATEX,
    }
    ar_obj = {
        "name": "Conditional Gaussian likelihood (closed-form least squares)",
        "source_path": (
            "revision/core/models/nonadversarial.py:139-158 "
            "(lstsq fit, no training loop)"
        ),
        "equation_latex": AR_LATEX,
    }

    # ── buckets.2_models ────────────────────────────────────────────────
    classical_models = classical.get("models", {})

    def _classical_arch_block(kind: str) -> dict:
        rec = classical_models.get(kind, {})
        block: dict = {
            "source": "revision/results/classical_architectures.json",
            "class": rec.get("class"),
            "carve_strategy": rec.get("carve_strategy"),
        }
        if "generator" in rec:
            block["layers"] = rec["generator"]
        if "encoder" in rec:
            block["encoder"] = rec["encoder"]
            block["decoder"] = rec.get("decoder", [])
            block["latent_dim"] = rec.get("latent_dim")
            block["hidden_dim"] = rec.get("hidden_dim")
            block["window"] = rec.get("window")
        if rec.get("type") == "AR(p)":
            block["type"] = "AR(p)"
            block["order_p"] = rec.get("order_p")
            block["fit_method"] = rec.get("fit_method")
        return block

    bucket_2 = {
        "iqp_sel_55_headline": {
            "family": "adversarial-quantum",
            "n_params": canonical.get("param_count"),
            "architecture": _arch_quantum(
                canonical, mi_by_model.get("iqp_sel_55_headline")
            ),
            "training_objective": dict(wgan_obj),
            "tier": "T1 (frozen checkpoint)",
            "checkpoint_epoch": canonical.get("checkpoint_epoch"),
        },
        "iqp_sel_55_repro": {
            "family": "adversarial-quantum",
            "n_params": canonical.get("param_count"),
            "architecture": _arch_quantum(
                canonical, mi_by_model.get("iqp_sel_55_repro")
            ),
            "training_objective": dict(wgan_obj),
            "tier": "T2 (matched 2000ep reproduction)",
        },
        "V1": {
            "family": "adversarial-quantum",
            "n_params": v1_lock.get("param_count"),
            "architecture": _arch_quantum(
                v1_lock, mi_by_model.get("V1")
            ),
            "training_objective": dict(wgan_obj),
            "tier": "T3 (matched-budget ansatz)",
        },
        "V2": {
            "family": "adversarial-quantum",
            "n_params": v2_lock.get("param_count"),
            "architecture": _arch_quantum(
                v2_lock, mi_by_model.get("V2")
            ),
            "training_objective": dict(wgan_obj),
            "tier": "T3 (matched-budget ansatz)",
        },
        "V3": {
            "family": "adversarial-quantum",
            "n_params": v3_lock.get("param_count"),
            "architecture": _arch_quantum(
                v3_lock, mi_by_model.get("V3")
            ),
            "training_objective": dict(wgan_obj),
            "tier": "T3 (matched-budget ansatz)",
        },
        "wgan_mlp": {
            "family": "adversarial-classical",
            "n_params": classical_models.get("wgan_mlp", {}).get("total_params"),
            "architecture": _classical_arch_block("wgan_mlp"),
            "training_objective": dict(wgan_obj),
            "tier": "T2 (matched-parameter baseline)",
        },
        "wgan_cnn": {
            "family": "adversarial-classical",
            "n_params": classical_models.get("wgan_cnn", {}).get("total_params"),
            "architecture": _classical_arch_block("wgan_cnn"),
            "training_objective": dict(wgan_obj),
            "tier": "T2 (matched-parameter baseline)",
        },
        "wgan_lstm": {
            "family": "adversarial-classical",
            "n_params": classical_models.get("wgan_lstm", {}).get("total_params"),
            "architecture": _classical_arch_block("wgan_lstm"),
            "training_objective": dict(wgan_obj),
            "tier": "T2 (matched-parameter baseline)",
        },
        "vae": {
            "family": "non-adversarial",
            "n_params": classical_models.get("vae", {}).get("total_params"),
            "architecture": _classical_arch_block("vae"),
            "training_objective": dict(vae_obj),
            "tier": "T2 (non-adversarial baseline)",
        },
        "ar": {
            "family": "non-adversarial",
            "n_params": classical_models.get("ar", {}).get("total_params"),
            "architecture": _classical_arch_block("ar"),
            "training_objective": dict(ar_obj),
            "tier": "T2 (non-adversarial baseline)",
        },
    }

    # ── buckets.3_training ─────────────────────────────────────────────
    repro = mi_by_model.get("iqp_sel_55_repro", {})
    bucket_3 = {
        "optimizer": "Adam",
        "optimizer_betas": repro.get("optimizer_betas", [0.0, 0.9]),
        "lr_critic": repro.get("lr_critic"),
        "lr_generator": repro.get("lr_generator"),
        "n_critic": repro.get("n_critic"),
        "lambda_gp": repro.get("lambda_gp"),
        "batch_size": repro.get("batch_size"),
        "epochs": repro.get("epochs"),
        "early_stopping": (
            "OFF for matched-budget reproduction (D-14-13); ON during "
            "original training that produced the frozen-checkpoint headline "
            "at epoch 1969"
        ),
        "seeds": repro.get("seeds", [42, 43, 44, 45, 46]),
        "source": (
            "revision/results/model_info.json (iqp_sel_55_repro row "
            "values; same across the 8 WGAN-GP models in the matched-2000ep "
            "sweep)"
        ),
    }

    # ── buckets.4_hardware_software (DISTINCT dtype fields!) ──────────
    bucket_4 = {
        "device": (
            "cpu (PennyLane default.qubit is CPU-only; classical runs also "
            "pinned to CPU for parity)"
        ),
        "pennylane_device": "default.qubit",
        "diff_method": "backprop",
        "dtype_params": (
            "torch.float32 (classical: nn.Parameter constructed with "
            "dtype=torch.float32 — revision/core/models/classical.py:78; "
            "quantum: params_pqc nn.Parameter dtype=torch.float32)"
        ),
        "dtype_samples": (
            "torch.float64 (sample-generation pipeline: "
            f"{cits['compute_dtype_split']} compute_dtype = torch.float64 on "
            f"CPU/CUDA; {cits['generator_to_compute_dtype']} generator output "
            "cast .to(compute_dtype) * 0.1; MPS path falls back to float32 "
            f"because MPS lacks float64 — see {cits['mps_dtype_block']})"
        ),
        "dtype_note": (
            "dtype_params and dtype_samples are DISTINCT — params live in "
            "float32 (the trainable nn.Parameter), samples are cast to "
            "float64 at generation time on CPU/CUDA to match the float64 "
            "critic (revision/core/models/critic.py:67 .double()). The two "
            "fields MUST NOT be conflated."
        ),
        "framework_versions": framework.get("packages", {}),
        "python_version": framework.get("python_version"),
        "platform": framework.get("platform"),
        "framework_versions_captured_at_iso": framework.get(
            "captured_at_iso"
        ),
    }

    # ── buckets.5_reproducibility ──────────────────────────────────────
    bucket_5 = {
        "data_hash": EXPECTED_DATA_HASH,
        "seed_set": [42, 43, 44, 45, 46],
        "determinism_contract": {
            "manual_seed": (
                f"{cits['manual_seed']} (torch.manual_seed(seed))"
            ),
            "numpy_seed": (
                f"{cits['numpy_seed']} (np.random.seed(seed))"
            ),
            "random_seed": (
                f"{cits['random_seed']} (random.seed(seed))"
            ),
            "cuda_seed": (
                f"{cits['cuda_seed']} (torch.cuda.manual_seed_all(seed) "
                "if cuda available)"
            ),
            "note": (
                "Seeds are set ONCE at the top of train_wgan_gp before "
                "optimizer/data construction (training.py:244-249). The "
                "seed is one of {42,43,44,45,46}; the same seed produces "
                "bit-identical training trajectories on the same "
                "device/dtype path."
            ),
        },
        "rerun_command_template": rerun_template,
        "rerun_command_template_source": (
            "revision/run_matched2000.py:1-69 (module docstring; preserved "
            "verbatim — never paraphrased)"
        ),
    }

    # ── Assemble final dict (bucket order is semantic — DO NOT sort) ──
    out = {
        "schema": "methods-full v1 (Phase 14 plan 14-11)",
        "data_hash": EXPECTED_DATA_HASH,
        "consumed_artifacts": {
            "revision/results/model_info.json": hashes.get(
                "model_info.json"
            ),
            "revision/results/canonical_config_lock.json": hashes.get(
                "canonical_config_lock.json",
                "no-data_hash (architecture lock)",
            ),
            "revision/results/default_75_config_lock.json": hashes.get(
                "default_75_config_lock.json",
                "no-data_hash (architecture lock)",
            ),
            "revision/results/v1_config_lock.json": hashes.get(
                "v1_config_lock.json",
                "no-data_hash (architecture lock)",
            ),
            "revision/results/v2_config_lock.json": hashes.get(
                "v2_config_lock.json",
                "no-data_hash (architecture lock)",
            ),
            "revision/results/v3_config_lock.json": hashes.get(
                "v3_config_lock.json",
                "no-data_hash (architecture lock)",
            ),
            "revision/results/classical_architectures.json": (
                "introspection (no data_hash by design)"
            ),
            "revision/results/framework_versions.json": (
                "introspection (no data_hash by design)"
            ),
            "revision/core/training.py": (
                "text-only file:line citations (no execution)"
            ),
            "revision/run_matched2000.py": (
                "text-only docstring slice (no execution)"
            ),
        },
        "buckets": {
            "1_dataset": model_info.get("dataset", {}),
            "2_models": bucket_2,
            "3_training": bucket_3,
            "4_hardware_software": bucket_4,
            "5_reproducibility": bucket_5,
        },
    }

    out_path = RESULTS / "methods_full.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")

    print(
        f"methods_full.json written: 5 buckets, "
        f"{len(bucket_2)} models, data_hash={EXPECTED_DATA_HASH}"
    )
    print(f"  output: {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
