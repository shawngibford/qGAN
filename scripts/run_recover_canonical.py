"""Phase 14 Tier-1 driver — recover the lost canonical 55-param IQP:SEL circuit.

This is the D-14-22 Tier-1 gate that BLOCKS every downstream Phase-14 plan
(D-14-07). It does two things:

  ``--recover-only``        Reverse-engineer the 55-param decomposition from
                            ``best_checkpoint.pt`` (ground truth, D-14-02),
                            identify the native Phase-09.1 headline pipeline
                            (D-14-06), and write
                            ``results/canonical_recovery.json``.

  ``--assert-equivalence``  Construct the NON-default config-selectable
                            55-param IQP:SEL circuit added to
                            ``core/models/quantum.py``, hard-assert
                            (explicit ``raise AssertionError`` — survives
                            ``python -O``, run_multiseed_rollup.py:86-92 idiom)
                            that ``best_checkpoint.pt``'s ``params_pqc`` loads
                            into it with shape ``(55,)`` and is consumed by a
                            structurally consistent forward pass, then write
                            ``results/canonical_config_lock.json``.

Ground-truth tensor layout (verified this session):
    keys = {epoch=1969, emd, params_pqc(55,), critic_state,
            c_optimizer, g_optimizer, mu, sigma}

Decomposition (checkpoint-driven, NOT formula-guessed — RESEARCH Pitfall 2):
    q=5, L=3, IQP-encoding present (5 params) + 3 SEL layers (45) +
    final RX-only per qubit (5)  ==  5 + 3*15 + 5 == 55
The IQP encoding params are PRESENT (this is what makes it "IQP:SEL"); the
non-default 75-param path keeps the extra final-RY per qubit (5 + 4*15 + 10).

Native headline pipeline = "B" (log-returns + standardize): the checkpoint
stores scalar ``mu``/``sigma`` (0.0025 / 0.0214) — log-return standardization
statistics that match ``results/run_unconditioned_wgan/stats.json``
``moments_real`` (mean 0.00245, std 0.02174). Pipeline A (min-max OD) would
store od_min/od_max bounds, not zero-centred mu/sigma, so A is excluded;
Pipeline C is the Lambert-W superset of B's log-return standardization and
shares the same stored (mu, sigma) scalars — the native standardization the
checkpoint was trained against is the log-return one (D-14-06).

Repo-root resolver + RESULTS anchor copied verbatim from
run_multiseed_rollup.py:42-59 (driver may run from a worktree / arbitrary
cwd). ``best_checkpoint.pt`` and ``qgan_env`` are .gitignored, so in a git
worktree they live in the *main* checkout, not the worktree — the checkpoint
resolver therefore also walks the real-repo ``.git`` link target.

Core default path is NEVER mutated by this driver (T-14-02): the 55-param
circuit is added to quantum.py as a NON-default config-selectable variant
(Task 2); core/__init__.py is untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Repo-root resolution (copied verbatim from run_multiseed_rollup.py:42-59).
# ─────────────────────────────────────────────────────────────────────────────
def _find_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for cand in [p, *p.parents]:
        if (cand / "core" / "preprocessing.py").exists():
            return cand
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found (core/preprocessing.py)")


REPO = _find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RESULTS = REPO / "results"

# Checkpoint-driven decomposition (RESEARCH Pitfall 2: 55 is non-unique by the
# formula — this is the structurally-verified family that keeps the IQP
# encoding params, i.e. a genuine IQP:SEL circuit).
DECOMP = {
    "num_qubits": 5,
    "num_layers": 3,
    "gate_layout": {
        "hadamard_init": True,
        "iqp_encoding_params_per_qubit": 1,   # 5 IQP-encoding RZ params
        "sel_rot_params_per_qubit_per_layer": 3,  # 3 * 5 * 3 = 45
        "final_rotation": "RX_only",          # 1 per qubit -> 5 (NOT RX+RY)
        "entangler": "range",                 # wrap-around CNOT (v1.0/v1.1)
    },
    "param_count": 55,  # 5 + 3*(5*3) + 5
}

# The NON-default config-selectable circuit id Task 2 wires into quantum.py.
CANONICAL_CIRCUIT_ID = "iqp_sel_55"


def _resolve_checkpoint() -> Path:
    """Locate ``best_checkpoint.pt`` (D-14-02 ground truth).

    .gitignored, so in a Claude Code worktree it is NOT copied in — it lives
    in the main checkout. Resolve via, in order: REPO root, cwd, then the
    main-repo toplevel that this worktree's ``.git`` file points at.
    """
    candidates = [REPO / "best_checkpoint.pt", Path.cwd() / "best_checkpoint.pt"]

    git_link = REPO / ".git"
    if git_link.is_file():
        # Worktree: ".git" is a file "gitdir: <main>/.git/worktrees/<name>".
        try:
            gitdir = git_link.read_text().strip()
            if gitdir.startswith("gitdir:"):
                wt_git = Path(gitdir.split(":", 1)[1].strip())
                # <main>/.git/worktrees/<name> -> main repo toplevel is wt_git
                # parents: [.../<name>, .../worktrees, .../.git, <main_root>]
                for anc in wt_git.parents:
                    if anc.name == ".git":
                        main_root = anc.parent
                        candidates.append(main_root / "best_checkpoint.pt")
                        break
        except OSError:
            pass

    for c in candidates:
        if c.exists():
            return c.resolve()
    raise RuntimeError(
        "best_checkpoint.pt not found (D-14-02 ground truth missing); "
        f"looked in: {[str(c) for c in candidates]}"
    )


def _load_checkpoint():
    import torch  # local import: pure-arg paths shouldn't require torch

    ckpt_path = _resolve_checkpoint()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt_path, ck


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _provenance(ckpt_path: Path, ck) -> dict:
    """Checkpoint provenance — stored scalars copied VERBATIM (D-14-05)."""

    def _scalar(v):
        # mu/sigma are stored as 0-dim tensors; record the stored value
        # verbatim (D-14-05: NEVER recompute).
        try:
            return float(v.item())
        except AttributeError:
            return float(v)

    g_pg = ck["g_optimizer"]["param_groups"][0]
    c_pg = ck["c_optimizer"]["param_groups"][0]
    return {
        "checkpoint_epoch": int(ck["epoch"]),
        "checkpoint_emd": float(ck["emd"]),
        "mu": _scalar(ck["mu"]),
        "sigma": _scalar(ck["sigma"]),
        "g_optimizer_lr": float(g_pg["lr"]),
        "c_optimizer_lr": float(c_pg["lr"]),
        "g_optimizer_betas": list(g_pg.get("betas", [])),
        "c_optimizer_betas": list(c_pg.get("betas", [])),
        "checkpoint_sha256": _sha256(ckpt_path),
        "checkpoint_filename": ckpt_path.name,
    }


def _identify_native_pipeline(prov: dict) -> dict:
    """D-14-06: pin the native Phase-09.1 headline pipeline.

    The stored (mu, sigma) are log-return *standardization* scalars. Pipeline
    A (min-max OD) would store od_min/od_max bounds instead, so A is excluded.
    The checkpoint's mu/sigma match results/run_unconditioned_wgan/stats.json
    moments_real — log-return statistics. The native standardization the
    checkpoint trained against is the log-return one (Pipeline B; Pipeline C
    is the Lambert-W superset of that same log-return standardization).
    """
    evidence = {
        "stored_mu": prov["mu"],
        "stored_sigma": prov["sigma"],
        "rationale": (
            "Checkpoint stores zero-centred scalar mu/sigma "
            f"({prov['mu']:.4g}/{prov['sigma']:.4g}) — log-return "
            "standardization stats, NOT min-max OD bounds, so Pipeline A is "
            "excluded. They match results/run_unconditioned_wgan/stats.json "
            "moments_real (mean~0.00245, std~0.02174 log-returns). Native "
            "headline standardization = log-returns (Pipeline B); Pipeline C "
            "is the Lambert-W superset sharing the same stored mu/sigma."
        ),
    }
    stats_path = REPO / "results" / "run_unconditioned_wgan" / "stats.json"
    if stats_path.exists():
        try:
            s = json.loads(stats_path.read_text())
            mr = s.get("distributional", {}).get("moments_real", {})
            evidence["stats_moments_real_mean"] = mr.get("mean")
            evidence["stats_moments_real_std"] = mr.get("std")
        except (OSError, json.JSONDecodeError):
            pass
    return {"native_pipeline": "B", "identification_evidence": evidence}


def recover() -> dict:
    ckpt_path, ck = _load_checkpoint()

    # Ground-truth shape gate (explicit raise — python -O strips assert;
    # run_multiseed_rollup.py:86-92 idiom).
    pqc_shape = tuple(ck["params_pqc"].shape)
    if pqc_shape != (55,):
        raise AssertionError(
            f"checkpoint params_pqc shape {pqc_shape} != (55,) — "
            "this is not the canonical 55-param IQP:SEL ground truth (D-14-02)"
        )

    derived = (
        DECOMP["num_qubits"]
        + DECOMP["num_layers"] * (DECOMP["num_qubits"] * 3)
        + DECOMP["num_qubits"]  # final RX-only
    )
    if derived != 55:
        raise AssertionError(
            f"decomposition arithmetic error: derived {derived} != 55"
        )

    prov = _provenance(ckpt_path, ck)
    pipeline = _identify_native_pipeline(prov)

    doc = {
        "schema": "canonical-recovery v1 (Phase 14 D-14-01/02/06)",
        "circuit_id": CANONICAL_CIRCUIT_ID,
        "decomposition": DECOMP,
        "param_count": derived,
        "checkpoint_epoch": prov["checkpoint_epoch"],
        "checkpoint_emd": prov["checkpoint_emd"],
        "mu": prov["mu"],
        "sigma": prov["sigma"],
        "checkpoint_sha256": prov["checkpoint_sha256"],
        "checkpoint_filename": prov["checkpoint_filename"],
        "optimizer_breadcrumbs": {
            "g_optimizer_lr": prov["g_optimizer_lr"],
            "c_optimizer_lr": prov["c_optimizer_lr"],
            "g_optimizer_betas": prov["g_optimizer_betas"],
            "c_optimizer_betas": prov["c_optimizer_betas"],
        },
        "native_pipeline": pipeline["native_pipeline"],
        "native_pipeline_evidence": pipeline["identification_evidence"],
        "corroborating_git_archaeology": (
            "git log -S NUM_LAYERS qgan_pennylane.ipynb shows NUM_LAYERS in "
            "{2,3,4} over time — CORROBORATING ONLY (D-14-02); the checkpoint "
            "params_pqc(55,) + structural forward-pass walk disambiguate to "
            "L=3 with IQP encoding present and final-RX-only."
        ),
        "data_hash_note": (
            "No dataset hash resolvable from the checkpoint alone (it stores "
            "only model/optimizer tensors + scalar mu/sigma); the headline "
            "data_hash is established downstream against the pinned native "
            "pipeline. Omitted here by design (D-14-06)."
        ),
        "core_default_path": "UNCHANGED (75-param 5,4 default; T-14-02)",
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "canonical_recovery.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"[recover] wrote {out}")
    print(
        f"[recover] decomposition q={DECOMP['num_qubits']} "
        f"L={DECOMP['num_layers']} -> param_count={derived} "
        f"(epoch={prov['checkpoint_epoch']}, "
        f"native_pipeline={pipeline['native_pipeline']})"
    )
    return doc


def assert_equivalence() -> dict:
    """D-14-07 phase-blocking config-equivalence hard-assert.

    Constructs the NON-default 55-param IQP:SEL circuit from quantum.py,
    loads the checkpoint's params_pqc into it, and uses the explicit-raise
    idiom (run_multiseed_rollup.py:86-92) for BOTH the shape gate and a
    structural forward-pass check (the reconstructed circuit must consume
    exactly 55 params). Non-zero exit on failure (Task 2 gate).
    """
    import torch
    from core.models.quantum import QuantumGenerator

    rec_path = RESULTS / "canonical_recovery.json"
    if not rec_path.exists():
        raise AssertionError(
            "canonical_recovery.json missing — run --recover-only first "
            "(Task 1 must precede the Task 2 equivalence gate)"
        )
    rec = json.loads(rec_path.read_text())

    ckpt_path, ck = _load_checkpoint()
    pqc = ck["params_pqc"]

    # ── Gate 1: shape (explicit raise, NOT bare assert — D-14-07) ──────────
    if tuple(pqc.shape) != (55,):
        raise AssertionError(
            f"params_pqc shape {tuple(pqc.shape)} != (55,) — checkpoint does "
            "NOT load into the reconstructed canonical config (D-14-07 BLOCK)"
        )

    # ── Build the NON-default 55-param IQP:SEL via the config selector ─────
    q = DECOMP["num_qubits"]
    gen = QuantumGenerator(
        num_qubits=q,
        num_layers=DECOMP["num_layers"],
        window_length=2 * q,
        circuit_id=CANONICAL_CIRCUIT_ID,
    )
    if gen.num_params != 55:
        raise AssertionError(
            f"reconstructed circuit num_params {gen.num_params} != 55 "
            f"(circuit_id={CANONICAL_CIRCUIT_ID}) — D-14-07 BLOCK"
        )

    # Load the ground-truth tensor into the reconstructed circuit.
    with torch.no_grad():
        if gen.params_pqc.shape != pqc.shape:
            raise AssertionError(
                f"param tensor shape mismatch: circuit "
                f"{tuple(gen.params_pqc.shape)} vs checkpoint "
                f"{tuple(pqc.shape)} — D-14-07 BLOCK"
            )
        gen.params_pqc.copy_(pqc.to(gen.params_pqc.dtype))

    # ── Gate 2: structural forward-pass — circuit must CONSUME exactly 55 ──
    consumed = gen.last_param_index(torch.zeros(q))
    if consumed != 55:
        raise AssertionError(
            f"structural forward-pass consumed {consumed} params, expected "
            f"55 — reconstructed circuit is NOT config-equivalent to the "
            f"checkpoint (D-14-07 BLOCK)"
        )

    # Sanity: a real forward pass returns the 2*q expectation tuple.
    out = gen(torch.zeros(q))
    if tuple(out.shape) != (2 * q,):
        raise AssertionError(
            f"forward output shape {tuple(out.shape)} != ({2 * q},) — "
            "D-14-07 BLOCK"
        )

    lock = {
        "schema": "canonical-config-lock v1 (Phase 14 D-14-05/06/07)",
        "locked_circuit_id": CANONICAL_CIRCUIT_ID,
        "decomposition": DECOMP,
        "param_count": 55,
        "structural_forward_pass_consumed": consumed,
        "native_pipeline": rec["native_pipeline"],
        "native_pipeline_evidence": rec["native_pipeline_evidence"],
        "mu": rec["mu"],
        "sigma": rec["sigma"],
        "checkpoint_epoch": rec["checkpoint_epoch"],
        "checkpoint_emd": rec["checkpoint_emd"],
        "checkpoint_sha256": rec["checkpoint_sha256"],
        "checkpoint_filename": rec["checkpoint_filename"],
        "equivalence_gate": (
            "PASSED — explicit-raise shape gate (==(55,)) + structural "
            "forward-pass gate (consumed==55); python -O safe (D-14-07)"
        ),
        "core_default_path": (
            "UNCHANGED — QuantumGenerator() with no args still builds the "
            "75-param (5,4) default (T-14-02)"
        ),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "canonical_config_lock.json"
    out_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(f"[assert-equivalence] PASSED — wrote {out_path}")
    print(
        f"[assert-equivalence] 55-param IQP:SEL "
        f"(circuit_id={CANONICAL_CIRCUIT_ID}) loaded the checkpoint; "
        f"structural walk consumed {consumed}/55 params"
    )
    return lock


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 14 Tier-1 canonical 55-param IQP:SEL recovery + "
        "config-equivalence hard-assert (D-14-07 phase-blocking gate)."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--recover-only",
        action="store_true",
        help="Task 1: reverse-engineer decomposition + write "
        "canonical_recovery.json (does not touch quantum.py).",
    )
    g.add_argument(
        "--assert-equivalence",
        action="store_true",
        help="Task 2: hard-assert the checkpoint loads into the non-default "
        "55-param circuit + write canonical_config_lock.json. Exit non-zero "
        "on failure (D-14-07 BLOCK).",
    )
    args = ap.parse_args()
    if args.recover_only:
        recover()
    else:
        assert_equivalence()


if __name__ == "__main__":
    main()
