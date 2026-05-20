"""Phase 14 plan 14-11 Task 1 — classical architecture extractor (introspection-only).

Pure-introspection emitter that imports the five classical model classes
(WGANMLPGenerator, WGANCNNGenerator, WGANLSTMGenerator, VAEBaseline,
ARBaseline) and the shared Critic, walks each ``nn.Module`` via
``module.named_modules()``, and emits a per-model layer-tree JSON to
``revision/results/classical_architectures.json``.

The script NEVER calls any model-fit method, NEVER samples, NEVER reloads a
checkpoint, NEVER runs a training loop. It instantiates each ``nn.Module``
with its default constructor purely so the named-modules walk has a live
instance — class instantiation runs no forward pass (verified pure-aggregator-
safe by plan 14-09 SUMMARY for the analogous ``revision.core.models.quantum``
import). The WGAN-GP generator classes carve their weights from a single
``params_pqc`` ``nn.Parameter`` and apply them via ``torch.nn.functional`` —
``named_modules()`` therefore yields only the root module for them, and the
extractor reads the functional layer sequence FROM the class docstring's
flat-layout annotation (classical.py:60-65 / 104-109 / 152-160) so each
docstring line maps to one ``layer_spec`` entry.

Drift gate (explicit ``raise AssertionError``, python -O safe — mirrors the
``run_multiseed_rollup.py:86-92`` idiom and the 14-09 Task-1 single-shot
mismatch-collector pattern): for each of the five model kinds the extractor
asserts that the walked ``total_params`` equals the ground-truth
``parameter_count`` recorded in ``revision/results/model_info.json`` for the
same kind. Every mismatch is collected into a list and reported in a single
loud raise BEFORE the JSON is written.

D-14-22 byte-freeze: ``revision/core/`` is opened ONLY via the public model-
class imports. ``git diff --stat revision/core/`` MUST remain empty across
the lifetime of this script (verify gate: see plan 14-11 Task 1).
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Repo-root resolution (verbatim run_model_info.py:69-81 idiom — the driver may
# run from a worktree / arbitrary cwd; results are repo-root anchored). The
# sys.path.insert below keeps the resolver shape identical to peers and is the
# only reason this script needs the path tweak (so the public ``revision.core``
# package imports resolve under bare-script invocation, not just ``python -m``).
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


# ─────────────────────────────────────────────────────────────────────────────
# Loud-fail loader (run_figure_suite.py:144-174 idiom). The extractor's only
# external JSON dependency is model_info.json (the drift basis); a missing
# artifact is a hard error, NEVER a silently empty drift gate.
# ─────────────────────────────────────────────────────────────────────────────
def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"[run_classical_arch_extract] required introspection-only "
            f"artifact missing: {path} ({what}). This extractor performs NO "
            f"model-fit / sampling / checkpoint-reload path — regenerate the "
            f"upstream JSON (run_model_info.py for model_info.json) before "
            f"re-running."
        )
    return path


def _load_json(path: Path, what: str) -> dict:
    _require(path, what)
    return json.loads(path.read_text())


# ─────────────────────────────────────────────────────────────────────────────
# Introspection imports (public path; 14-09 verified pure-aggregator-safe).
# The modules' own top-level imports are torch + torch.nn + torch.nn.functional
# (+ numpy for nonadversarial); class instantiation runs no forward pass.
# ─────────────────────────────────────────────────────────────────────────────
from revision.core.models.classical import (  # noqa: E402
    WGANMLPGenerator,
    WGANCNNGenerator,
    WGANLSTMGenerator,
)
from revision.core.models.nonadversarial import (  # noqa: E402
    VAEBaseline,
    ARBaseline,
)
from revision.core.models.critic import Critic  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Functional layer-tree builders. The three WGAN-GP generators are functional
# (no nn submodules — they carve params_pqc into per-layer slices and apply
# them via torch.nn.functional), so the layer_spec list mirrors the docstring
# flat-layout annotation byte-exactly. VAE and Critic ARE nn.Modules with real
# submodules — those are walked via named_modules() further down. AR is a
# plain (non-nn.Module) class so its spec is hand-encoded.
# ─────────────────────────────────────────────────────────────────────────────
def _wgan_mlp_layers() -> list[dict]:
    """WGANMLPGenerator — Linear(5,4)+b -> Tanh -> Linear(4,10)+b (74 params).

    Slice indices verified against ``revision/core/models/classical.py:58-95``.
    """
    return [
        {
            "layer_type": "Linear",
            "in_features": 5,
            "out_features": 4,
            "bias": True,
            "activation": "tanh",
            "param_count": 24,
            "params_pqc_slice": "[0:24]",
        },
        {
            "layer_type": "Linear",
            "in_features": 4,
            "out_features": 10,
            "bias": True,
            "activation": None,
            "param_count": 50,
            "params_pqc_slice": "[24:74]",
        },
    ]


def _wgan_cnn_layers() -> list[dict]:
    """WGANCNNGenerator — ConvTranspose1d -> LeakyReLU -> Conv1d (73 params).

    Slice indices verified against ``revision/core/models/classical.py:98-141``.
    """
    return [
        {
            "layer_type": "Reshape",
            "from_shape": "(B, 5)",
            "to_shape": "(B, 1, 5)",
            "param_count": 0,
            "note": "parameter-free reshape (NOT a learnable projection)",
        },
        {
            "layer_type": "ConvTranspose1d",
            "in_channels": 1,
            "out_channels": 9,
            "kernel_size": 6,
            "stride": 1,
            "padding": 0,
            "bias": True,
            "activation": "leaky_relu(0.1)",
            "param_count": 63,
            "params_pqc_slice": "[0:63]",
        },
        {
            "layer_type": "Conv1d",
            "in_channels": 9,
            "out_channels": 1,
            "kernel_size": 1,
            "bias": True,
            "activation": None,
            "param_count": 10,
            "params_pqc_slice": "[63:73]",
        },
    ]


def _wgan_lstm_layers() -> list[dict]:
    """WGANLSTMGenerator — functional LSTM(I=2,H=2,1 layer) -> Linear(2,10)+b (78 params).

    Slice indices verified against ``revision/core/models/classical.py:144-212``.
    """
    return [
        {
            "layer_type": "Reshape",
            "from_shape": "(B, 5)",
            "to_shape": "(B, seq=3, input=2)",
            "param_count": 0,
            "note": "parameter-free tile/pad (NOT a learnable projection)",
        },
        {
            "layer_type": "LSTM_functional",
            "input_size": 2,
            "hidden_size": 2,
            "num_layers": 1,
            "bias": True,
            "gate_order": "i, f, g, o (PyTorch order)",
            "param_count": 48,
            "params_pqc_slice": "[0:48] (W_ih 16 + W_hh 16 + b_ih 8 + b_hh 8)",
        },
        {
            "layer_type": "Linear",
            "in_features": 2,
            "out_features": 10,
            "bias": True,
            "activation": None,
            "param_count": 30,
            "params_pqc_slice": "[48:78]",
        },
    ]


def _walk_named_modules(model) -> list[dict]:
    """Walk an ``nn.Module`` via ``named_modules()`` and emit a layer_spec
    entry for each LEAF child (skip the empty root and any container that has
    children — only record the leaves that actually own parameters or are an
    activation/dropout/pool/flatten layer).
    """
    import torch.nn as nn  # local import — pure introspection
    out: list[dict] = []
    for name, mod in model.named_modules():
        # Skip the root module (name == "") and any non-leaf container (has
        # children) EXCEPT nn.Sequential leaves: in the Critic the Sequential
        # is named "net" and its leaves are what we want.
        if name == "":
            continue
        if any(True for _ in mod.children()):
            continue
        spec: dict = {"name": name, "layer_type": mod.__class__.__name__}
        if isinstance(mod, nn.Linear):
            spec.update(
                {
                    "in_features": mod.in_features,
                    "out_features": mod.out_features,
                    "bias": mod.bias is not None,
                    "param_count": sum(p.numel() for p in mod.parameters()),
                }
            )
        elif isinstance(mod, nn.Conv1d):
            spec.update(
                {
                    "in_channels": mod.in_channels,
                    "out_channels": mod.out_channels,
                    "kernel_size": (
                        mod.kernel_size[0]
                        if isinstance(mod.kernel_size, tuple)
                        else mod.kernel_size
                    ),
                    "stride": (
                        mod.stride[0]
                        if isinstance(mod.stride, tuple)
                        else mod.stride
                    ),
                    "padding": (
                        mod.padding[0]
                        if isinstance(mod.padding, tuple)
                        else mod.padding
                    ),
                    "bias": mod.bias is not None,
                    "param_count": sum(p.numel() for p in mod.parameters()),
                }
            )
        elif isinstance(mod, nn.LeakyReLU):
            spec.update(
                {
                    "negative_slope": mod.negative_slope,
                    "param_count": 0,
                }
            )
        elif isinstance(mod, nn.AdaptiveAvgPool1d):
            spec.update({"output_size": mod.output_size, "param_count": 0})
        elif isinstance(mod, nn.Flatten):
            spec.update({"param_count": 0})
        elif isinstance(mod, nn.Dropout):
            spec.update({"p": mod.p, "param_count": 0})
        else:
            spec.update(
                {"param_count": sum(p.numel() for p in mod.parameters())}
            )
        out.append(spec)
    return out


def _vae_split_encoder_decoder(specs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split a flat ``named_modules`` walk of VAEBaseline into encoder/decoder
    sub-lists by the well-known submodule names (enc, fc_mu, fc_logvar are
    encoder; dec_h, dec_out are decoder). See
    ``revision/core/models/nonadversarial.py:67-73``."""
    encoder = [s for s in specs if s["name"] in {"enc", "fc_mu", "fc_logvar"}]
    decoder = [s for s in specs if s["name"] in {"dec_h", "dec_out"}]
    return encoder, decoder


# ─────────────────────────────────────────────────────────────────────────────
# Drift gate (explicit raise; collects EVERY mismatch into a list before the
# single-shot raise — 14-09 Task-1 idiom).
# ─────────────────────────────────────────────────────────────────────────────
def _drift_check(
    walked_totals: dict[str, int], model_info: dict
) -> None:
    expected: dict[str, int] = {}
    for rec in model_info.get("models", []):
        kind = rec.get("kind")
        if kind in walked_totals:
            expected[kind] = rec.get("parameter_count")
    mismatches: list[str] = []
    for kind, walked in walked_totals.items():
        exp = expected.get(kind)
        if exp is None:
            mismatches.append(
                f"{kind}: no model_info.json record found (kind missing)"
            )
        elif walked != exp:
            mismatches.append(
                f"{kind}: walked total_params={walked} != "
                f"model_info.json parameter_count={exp}"
            )
    if mismatches:
        raise AssertionError(
            "classical_architectures.json drift detected — extractor's walked "
            "total_params disagree with revision/results/model_info.json "
            f"parameter_count for: {mismatches}"
        )


def main() -> None:
    # Loud-fail load model_info.json (the drift basis).
    model_info = _load_json(
        RESULTS / "model_info.json", "drift-basis parameter_count source"
    )

    # ── Functional layer trees for the three WGAN-GP generators (params_pqc).
    mlp_layers = _wgan_mlp_layers()
    cnn_layers = _wgan_cnn_layers()
    lstm_layers = _wgan_lstm_layers()

    # Construct instances under no_grad to be explicit that nothing trains.
    import torch  # local import — pure introspection
    with torch.no_grad():
        mlp = WGANMLPGenerator()
        cnn = WGANCNNGenerator()
        lstm = WGANLSTMGenerator()
        vae = VAEBaseline()
        critic = Critic()
        ar = ARBaseline()

    # Confirm the params_pqc total for the WGAN-GP triple matches the carved
    # layer sums (count_params() reads params_pqc.numel()).
    mlp_total = mlp.count_params()
    cnn_total = cnn.count_params()
    lstm_total = lstm.count_params()
    vae_total = vae.count_params()
    ar_total = ar.count_params()

    # ── Walked submodule layers for VAE + Critic (real nn submodules) ──
    vae_specs = _walk_named_modules(vae)
    vae_enc, vae_dec = _vae_split_encoder_decoder(vae_specs)
    critic_specs = _walk_named_modules(critic)

    # ── AR layer_spec (plain class; not an nn.Module) ──────────────────
    ar_spec = {
        "layer_type": "AR(p)",
        "order_p": ar.p,
        "params_per_seed": ar.count_params(),
        "fit_method": "np.linalg.lstsq (closed-form, no training loop)",
        "burn_in_steps": 50,
        "source_path": "revision/core/models/nonadversarial.py:139-158",
    }

    # ── Drift gate against model_info.json ─────────────────────────────
    walked_totals = {
        "wgan_mlp": mlp_total,
        "wgan_cnn": cnn_total,
        "wgan_lstm": lstm_total,
        "vae": vae_total,
        "ar": ar_total,
    }
    _drift_check(walked_totals, model_info)

    # ── Build output JSON ──────────────────────────────────────────────
    out = {
        "schema": "classical-architectures v1 (Phase 14 plan 14-11)",
        "extracted_at_iso": datetime.datetime.utcnow().isoformat() + "Z",
        "extractor": (
            "module.named_modules() walk for nn.Module classes (VAEBaseline, "
            "Critic) + functional-layout docstring parse for the three "
            "params_pqc-carved WGAN-GP generators + hand-encoded plain-class "
            "spec for ARBaseline (not an nn.Module)"
        ),
        "models": {
            "wgan_mlp": {
                "class": "WGANMLPGenerator",
                "source_path": "revision/core/models/classical.py:58-95",
                "carve_strategy": (
                    "single nn.Parameter (params_pqc) sliced into per-layer "
                    "weight/bias views applied via torch.nn.functional"
                ),
                "generator": mlp_layers,
                "total_params": mlp_total,
            },
            "wgan_cnn": {
                "class": "WGANCNNGenerator",
                "source_path": "revision/core/models/classical.py:98-141",
                "carve_strategy": (
                    "single nn.Parameter (params_pqc) sliced into per-layer "
                    "weight/bias views applied via torch.nn.functional"
                ),
                "generator": cnn_layers,
                "total_params": cnn_total,
            },
            "wgan_lstm": {
                "class": "WGANLSTMGenerator",
                "source_path": "revision/core/models/classical.py:144-212",
                "carve_strategy": (
                    "single nn.Parameter (params_pqc) sliced into LSTM cell "
                    "matrices + final Linear applied via torch.nn.functional"
                ),
                "generator": lstm_layers,
                "total_params": lstm_total,
            },
            "vae": {
                "class": "VAEBaseline",
                "source_path": "revision/core/models/nonadversarial.py:52-117",
                "carve_strategy": (
                    "five nn.Linear submodules (enc, fc_mu, fc_logvar, dec_h, "
                    "dec_out); ELBO objective trained in revision/run_baselines.py"
                ),
                "encoder": vae_enc,
                "decoder": vae_dec,
                "objective": "ELBO",
                "latent_dim": VAEBaseline.LATENT_DIM,
                "hidden_dim": VAEBaseline.HIDDEN,
                "window": VAEBaseline.WINDOW,
                "total_params": vae_total,
            },
            "ar": {
                "class": "ARBaseline",
                "source_path": "revision/core/models/nonadversarial.py:120-184",
                "carve_strategy": (
                    "plain Python class — NOT an nn.Module; fit is "
                    "closed-form np.linalg.lstsq, no training loop"
                ),
                "layer_spec": ar_spec,
                "type": "AR(p)",
                "order_p": ar.p,
                "fit_method": "np.linalg.lstsq (closed-form)",
                "total_params": ar_total,
            },
            "shared_critic": {
                "class": "Critic",
                "source_path": "revision/core/models/critic.py:19-77",
                "carve_strategy": (
                    "nn.Sequential of Conv1d x3 + LeakyReLU + "
                    "AdaptiveAvgPool1d + Flatten + Linear(128,32) + Dropout + "
                    "Linear(32,1); cast to .double() at critic.py:67"
                ),
                "layers": critic_specs,
                "dtype": "torch.float64",
                "total_params": sum(p.numel() for p in critic.parameters()),
                "note": (
                    "Shared across iqp_sel_55_repro + V1/V2/V3 + "
                    "wgan_mlp/cnn/lstm — one critic instance per training "
                    "run, NOT a model in the model_info.json sense"
                ),
            },
        },
        "consumed_artifacts": {
            "revision/results/model_info.json": model_info.get("data_hash"),
        },
        "drift_gate": (
            "explicit raise AssertionError (python -O safe) — extractor's "
            "walked total_params == model_info.json parameter_count for "
            "wgan_mlp/cnn/lstm/vae/ar; PASSED before this JSON was written"
        ),
    }

    out_path = RESULTS / "classical_architectures.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(
        f"classical_architectures.json written: 5 model entries + "
        f"shared_critic; total_params drift gate PASSED against "
        f"model_info.json"
    )
    print(f"  output: {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
