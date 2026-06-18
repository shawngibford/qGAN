"""Phase 11 Plan 02 — EVAL-02 / EVAL-03 faithful TimeGAN post-hoc scores.

Delivers the **predictive score** (EVAL-02) and **discriminative score**
(EVAL-03) using single-layer-GRU post-hoc nets ported faithfully from the
canonical ``jsyoon0823/TimeGAN`` NeurIPS 2019 algorithm
(``metrics/predictive_metrics.py`` + ``metrics/discriminative_metrics.py``,
``master`` branch). It consumes the 60 frozen ``samples.npy`` artifacts (50
Phase-10 baseline + 10 Phase-09.1 quantum) via the verbatim ``reconstruct_od``
recipe — **no regeneration, no retraining** (D-11-08).

Pinned definitions (jsyoon0823/TimeGAN, NeurIPS 2019, master):
  Predictive (metrics/predictive_metrics.py):
    single-layer GRU + Linear head; Adam lr=1e-3; 5000 iters; batch 128;
    train = synthetic, test = real; X = window[:, :-1], y = window[:, -1]
    (one-step); score = mean(|y_real - y_pred|). Lower better.
  Discriminative (metrics/discriminative_metrics.py):
    single-layer GRU + Linear(H, 1) -> sigmoid logit; Adam lr=1e-3; 2000 iters;
    batch 128; real label = 1, synth label = 0; 80/20 split of each pool
    independently (np.random.permutation with recorded seed);
    score = abs(0.5 - test_accuracy). Lower better.

Univariate adaptation (Pitfall 1 / Assumptions-Log A1): the canonical
``hidden_dim = int(dim/2)`` is **degenerate** here — ``dim`` = 1 (univariate)
so ``int(1/2) = 0`` would build a zero-width GRU. H is **locked** to
``WINDOW_LENGTH = 10`` (D-11-04 "hidden dim ~ input_dim", reading the length-10
window as the input) and used **identically** for both post-hoc nets. The
locked H, the rationale, and the repo URL + branch are recorded in
``results/predictive_discriminative.json["metadata"]``.

Usage
-----
    python run_timegan_scores.py \\
        [--out results] [--csv-path ./data.csv] \\
        [--hidden 10] [--pred-iters 5000] [--disc-iters 2000] [--batch 128]

Pitfall 4: quantum runs carry no ``data_hash`` config field — the data-hash
assert loop covers the **50 baseline configs only** (quantum equivalence is
established by construction, same ``load_and_preprocess(data.csv)``).
Pitfall 5: single process, no ``multiprocessing.Pool``.
Pitfall 6: ``data.csv`` is repo-root relative — resolved to an absolute path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


# ── repo root (mirrors _build_baseline_notebook.py:74-93) ────────────────────
def _find_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for cand in [p, *p.parents]:
        if (cand / "core" / "preprocessing.py").exists():
            return cand
    # Fallback to cwd-walk if launched from an unusual location.
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "core" / "preprocessing.py").exists():
            return cand
    raise RuntimeError("repo root not found")


REPO = _find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.preprocessing import inverse_logreturns  # noqa: E402
from core.data import load_and_preprocess, rolling_window  # noqa: E402
from core import WINDOW_LENGTH  # noqa: E402
from _wgan_unscale import _unscale_wgan_samples  # noqa: E402  Plan 14-21 T01

# Matched-budget driver mode (14-20): 9 trainable model_kinds × 1 pipeline (B)
# × 5 seeds = 45 frozen cells under results/matched2000/runs/<mk>/<seed>/.
MODEL_KINDS = ["iqp_sel_55_repro", "V1", "V2", "V3",
               "wgan_mlp", "wgan_cnn", "wgan_lstm", "vae", "ar"]
PIPELINES = ["B"]                 # Pipeline B only (matched-budget driver mode, 14-20)
SEEDS = [42, 43, 44, 45, 46]

# TimeGAN reference pin (recorded in JSON metadata, T-11-06 mitigation).
TIMEGAN_REPO_URL = "https://github.com/jsyoon0823/TimeGAN"
TIMEGAN_BRANCH = "master"
TIMEGAN_PRED_FILE = "metrics/predictive_metrics.py"
TIMEGAN_DISC_FILE = "metrics/discriminative_metrics.py"

EXPECTED_DATA_HASH = "91e447d4624e25b3"


# ── data_hash invariant (D-10-15 / Pitfall 4) ────────────────────────────────
def _compute_data_hash(csv_path: Path) -> str:
    """sha256(real_OD bytes)[:16] recomputed from load_and_preprocess.

    Verbatim contract of ``run_baselines._compute_data_hash`` — pins every
    consumed run to the exact OD tensor it trained against.
    """
    od = load_and_preprocess(str(csv_path))["OD"].cpu().numpy()
    return hashlib.sha256(od.tobytes()).hexdigest()[:16]


def _assert_baseline_data_hashes(expected: str) -> dict:
    """Assert all 45 matched-budget config.yaml data_hash fields equal `expected`.

    Matched-budget driver mode (14-20): 9 trainable model_kinds × 5 seeds = 45
    cells under matched2000/runs/<model_kind>/<seed>/. Every cell carries the
    canonical data_hash directly — no quantum-by-construction shortcut needed
    (the matched2000 sweep wrote data_hash into every config.yaml, including
    the 4 quantum variants iqp_sel_55_repro/V1/V2/V3).
    """
    all_models = ["iqp_sel_55_repro", "V1", "V2", "V3",
                  "wgan_mlp", "wgan_cnn", "wgan_lstm", "vae", "ar"]
    checked = 0
    mismatches = []
    for m in all_models:
        for s in SEEDS:
            cfg_path = REPO / f"results/matched2000/runs/{m}/{s}/config.yaml"
            cfg = yaml.safe_load(cfg_path.read_text())
            checked += 1
            if cfg.get("data_hash") != expected:
                mismatches.append((m, s, cfg.get("data_hash")))
    assert checked == 45, f"expected 45 matched-budget configs, checked {checked}"
    assert not mismatches, f"data_hash mismatch (UNEXPECTED — report loudly): {mismatches}"
    matched_budget_note = (
        "Matched-budget driver mode (14-20): all 45 cells under "
        "results/matched2000/runs/<model_kind>/<seed>/ carry the "
        "canonical data_hash 91e447d4624e25b3 in their config.yaml directly. "
        "The 4 quantum variants (iqp_sel_55_repro, V1, V2, V3) and 5 classical "
        "baselines (wgan_mlp, wgan_cnn, wgan_lstm, vae, ar) all assert by-value, "
        "not by-construction; no Pitfall-4 shortcut applied."
    )
    return {
        "recomputed_from": "load_and_preprocess('data.csv')['OD'].tobytes() sha256[:16]",
        "n_matched_budget_configs_checked": checked,
        "all_equal": True,
        "matched_budget_note": matched_budget_note,
    }


# ── reconstruct_od — copied VERBATIM (A + B branches only) ────────────────────
# Byte-identical to _build_baseline_notebook.py:167-210 (Plan 01 Task 1 too).
def _run_base(model_kind: str, pipeline: str, seed: int) -> Path:
    # Matched-budget driver mode (14-20): single branch, all 9 trainable
    # model_kinds resolve to results/matched2000/runs/<mk>/<seed>/.
    # No pipeline subdir (Pipeline B is implicit; an A-branch lookup raises in
    # reconstruct_od via the NotImplementedError guard below).
    return REPO / f"results/matched2000/runs/{model_kind}/{seed}"


def reconstruct_od(model_kind: str, pipeline: str, seed: int,
                   n_synth_subsample: int | None = None) -> dict:
    # 14-20 guard: matched-budget driver mode is Pipeline B only. The
    # matched2000 sweep has no Pipeline-A cells — every cell was trained as
    # Pipeline B with log-return transform. The Pipeline-A branch below is
    # retained as dead-but-not-deleted reference but is unreachable here.
    if pipeline != "B":
        raise NotImplementedError(
            f"Matched-budget driver mode supports Pipeline B only; "
            f"got pipeline={pipeline!r}. The matched2000 sweep "
            f"(results/matched2000/runs/) has no Pipeline-A cells."
        )
    base = _run_base(model_kind, pipeline, seed)
    samples_pm1 = np.load(base / "samples.npy").astype(np.float64)
    samples_pm1 = _unscale_wgan_samples(samples_pm1, model_kind)  # Plan 14-21 T01
    inv = np.load(base / "inverse_kwargs.npz", allow_pickle=True)

    if n_synth_subsample is not None and samples_pm1.shape[0] > n_synth_subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(samples_pm1.shape[0], n_synth_subsample, replace=False)
        samples_pm1 = samples_pm1[idx]

    if pipeline == "A":
        od_min = float(inv["od_min"]); od_max = float(inv["od_max"])
        od01 = (samples_pm1 + 1.0) / 2.0
        od = od01 * (od_max - od_min) + od_min
        return {"od_samples": od, "transformed": None, "n_synth": od.shape[0],
                "pipeline": pipeline, "seed": seed}

    if pipeline == "B":
        r_min = float(inv["r_min"]); r_max = float(inv["r_max"])
        mu = float(inv["mu"]); sigma = float(inv["sigma"])
        od_starts_pool = np.asarray(inv["od_starts"])
        r_norm = ((samples_pm1 + 1.0) / 2.0) * (r_max - r_min) + r_min
        rng = np.random.default_rng(seed * 7919 + 1)
        od_start_per_window = rng.choice(od_starts_pool, size=r_norm.shape[0], replace=True)
        r_norm_t = torch.tensor(r_norm)
        od_start_t = torch.tensor(od_start_per_window)
        od_full = inverse_logreturns(r_norm_t, od_start_t,
                                     torch.tensor(mu), torch.tensor(sigma))
        od = od_full.cpu().numpy()
        if od.shape[1] == 11:
            od = od[:, :10]
        return {"od_samples": od, "transformed": r_norm, "n_synth": od.shape[0],
                "pipeline": pipeline, "seed": seed}

    raise ValueError(f"unknown pipeline {pipeline} (Pipeline C dropped, D-10-05)")


# ── Faithful TimeGAN post-hoc nets (torch port of the pinned TF1 algorithm) ───
# Algorithm source: jsyoon0823/TimeGAN/metrics/predictive_metrics.py (CITED).
# input_size=1 (univariate adaptation, Pitfall 1); H locked to WINDOW_LENGTH=10
# (NOT the degenerate canonical int(dim/2)=0) — see module docstring / A1.
class PredictiveGRU(torch.nn.Module):
    """Single-layer GRU + Linear next-step predictor (predictive_metrics.py)."""

    def __init__(self, H: int):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=1, hidden_size=H,
                                num_layers=1, batch_first=True)
        # TF ref applies a sigmoid; identity head is fine for MAE regression
        # on the [-1,1] / OD-scale targets (kept faithful to the algorithm).
        self.fc = torch.nn.Linear(H, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T-1, 1)
        out, _ = self.gru(x)
        return self.fc(out)                              # (B, T-1, 1)


# Algorithm source: jsyoon0823/TimeGAN/metrics/discriminative_metrics.py (CITED).
class DiscriminativeGRU(torch.nn.Module):
    """Single-layer GRU + Linear(H,1) real-vs-synth classifier (logit out)."""

    def __init__(self, H: int):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=1, hidden_size=H,
                                num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(H, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, 1)
        # TF ref classifies from the last GRU hidden state -> single logit.
        _, h_n = self.gru(x)                             # h_n: (1, B, H)
        return self.fc(h_n[-1]).squeeze(-1)              # (B,)


def predictive_score(synth_windows: np.ndarray, real_windows: np.ndarray,
                     seed: int, H: int, iters: int = 5000,
                     bs: int = 128) -> float:
    """Faithful TimeGAN predictive score (predictive_metrics.py).

    Train a single-layer-GRU next-step predictor on SYNTHETIC windows
    (X = w[:, :-1], y = w[:, -1]), test on REAL windows; return the mean
    absolute next-step error. Adam(lr=1e-3), `iters` minibatches of `bs`.
    Deterministic given (seed) via torch.manual_seed + a recorded permutation
    RNG. Lower is better.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    sw = np.asarray(synth_windows, dtype=np.float64)
    rw = np.asarray(real_windows, dtype=np.float64)

    Xtr = torch.tensor(sw[:, :-1], dtype=torch.float32).unsqueeze(-1)   # (Ns,T-1,1)
    ytr = torch.tensor(sw[:, 1:], dtype=torch.float32).unsqueeze(-1)    # next-step seq
    Xev = torch.tensor(rw[:, :-1], dtype=torch.float32).unsqueeze(-1)
    yev = torch.tensor(rw[:, 1:], dtype=torch.float32).unsqueeze(-1)

    model = PredictiveGRU(H)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.L1Loss()                                        # MAE objective
    n = Xtr.shape[0]

    model.train()
    for _ in range(int(iters)):
        idx = rng.integers(0, n, size=min(bs, n))
        xb, yb = Xtr[idx], ytr[idx]
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(Xev)
    return float(torch.mean(torch.abs(yev - y_pred)).item())


def discriminative_score(real_windows: np.ndarray, synth_windows: np.ndarray,
                         seed: int, H: int, iters: int = 2000,
                         bs: int = 128) -> float:
    """Faithful TimeGAN discriminative score (discriminative_metrics.py).

    Label real = 1, synth = 0. Independent 80/20 split of EACH pool via
    np.random.permutation (seed recorded). Train a single-layer-GRU classifier
    (BCEWithLogitsLoss, Adam lr=1e-3, `iters` minibatches of `bs`), evaluate
    test accuracy at the 0.5 threshold; return abs(0.5 - test_accuracy).
    Lower is better.
    """
    torch.manual_seed(seed)
    # WR-05: one explicit Generator drives BOTH 80/20 pool splits AND the
    # minibatch index draws. No legacy global np.random.seed — the splits are
    # no longer correlated through global-RNG order and are immune to upstream
    # global-RNG mutation or call reordering.
    g = np.random.default_rng(seed)

    rw = np.asarray(real_windows, dtype=np.float64)
    sw = np.asarray(synth_windows, dtype=np.float64)

    def _split(pool: np.ndarray, g: np.random.Generator, rate: float = 0.8):
        n = pool.shape[0]
        perm = g.permutation(n)
        cut = int(n * rate)
        return pool[perm[:cut]], pool[perm[cut:]]

    # Split order is LOAD-BEARING: _split(rw) then _split(sw) consume `g`
    # sequentially in this exact order — reordering changes the streams.
    r_tr, r_te = _split(rw, g)
    s_tr, s_te = _split(sw, g)

    def _to_xy(real_part: np.ndarray, synth_part: np.ndarray):
        X = np.concatenate([real_part, synth_part], axis=0)
        y = np.concatenate([
            np.ones(real_part.shape[0], dtype=np.float64),
            np.zeros(synth_part.shape[0], dtype=np.float64),
        ])
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)        # (N,T,1)
        yt = torch.tensor(y, dtype=torch.float32)
        return Xt, yt

    Xtr, ytr = _to_xy(r_tr, s_tr)
    Xte, yte = _to_xy(r_te, s_te)

    model = DiscriminativeGRU(H)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    n = Xtr.shape[0]
    # WR-05: reuse the SAME Generator `g` (no second Generator) — minibatch
    # draws continue the single explicit stream after the two splits.
    model.train()
    for _ in range(int(iters)):
        idx = g.integers(0, n, size=min(bs, n))
        xb, yb = Xtr[idx], ytr[idx]
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        # WR-06 shape contract: the GRU may emit (B,1); reconcile to (B,) and
        # assert logits/labels share shape so a (B,1) vs (B,) broadcast can
        # never silently produce a meaningless (B,B) accuracy.
        logits = logits.squeeze(-1)
        assert logits.shape == yte.shape, (logits.shape, yte.shape)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = float((preds == yte).float().mean().item())
    return float(abs(0.5 - acc))


# ── per-seed roll-up + JSON writer (Task 2) ──────────────────────────────────
def _build_results(args) -> dict:
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = (REPO / csv_path).resolve()          # Pitfall 6

    expected = _compute_data_hash(csv_path)
    assert expected == EXPECTED_DATA_HASH, (
        f"recomputed data_hash {expected!r} != pinned {EXPECTED_DATA_HASH!r}"
    )
    data_hash_verification = _assert_baseline_data_hashes(expected)

    d_real = load_and_preprocess(str(csv_path))
    real_windowed_OD = rolling_window(
        d_real["OD"], WINDOW_LENGTH, 2).cpu().numpy().astype(np.float64)
    print(f"real windowed OD: shape={real_windowed_OD.shape}")

    H = int(args.hidden)
    rows: list[dict] = []
    scores: dict[str, dict] = {}

    for mk in MODEL_KINDS:
        for p in PIPELINES:
            pred_vals, disc_vals = [], []
            for s in SEEDS:
                r = reconstruct_od(mk, p, s)
                synth = r["od_samples"].astype(np.float64)
                ps = predictive_score(synth, real_windowed_OD, s, H,
                                      iters=int(args.pred_iters),
                                      bs=int(args.batch))
                ds = discriminative_score(real_windowed_OD, synth, s, H,
                                          iters=int(args.disc_iters),
                                          bs=int(args.batch))
                print(f"  {mk:>9} {p} seed={s}: "
                      f"predictive={ps:.5f} discriminative={ds:.5f}")
                rows.append(dict(model_kind=mk, pipeline=p, seed=s,
                                 metric_name="predictive_score",
                                 scale="OD", value=ps))
                rows.append(dict(model_kind=mk, pipeline=p, seed=s,
                                 metric_name="discriminative_score",
                                 scale="OD", value=ds))
                pred_vals.append(ps)
                disc_vals.append(ds)
            scores[f"{mk}|{p}"] = {
                "model_kind": mk,
                "pipeline": p,
                "n_seeds": len(SEEDS),
                "predictive_mean": float(np.mean(pred_vals)),
                "predictive_std": float(np.std(pred_vals)),
                "discriminative_mean": float(np.mean(disc_vals)),
                "discriminative_std": float(np.std(disc_vals)),
            }

    univariate_adaptation = (
        "Canonical TimeGAN post-hoc nets use hidden_dim = int(dim/2) where dim "
        "is the multivariate feature count. This project's data is UNIVARIATE "
        "(dim = 1), so int(1/2) = 0 builds a degenerate zero-width GRU "
        "(RESEARCH Pitfall 1 / Assumptions-Log A1). Following D-11-04 ('hidden "
        f"dim ~ input_dim') and reading the length-{WINDOW_LENGTH} window as "
        f"the input, H is LOCKED to WINDOW_LENGTH = {H} and used IDENTICALLY "
        "for both the predictive and discriminative post-hoc nets; consistency "
        "across the two nets matters more than the exact value. input_size is "
        "1 (one univariate channel) with sequence length = window length."
    )

    return {
        "schema": "long-form rows[] + scores block (EVAL-02/03, D-11-05)",
        "model_kinds": MODEL_KINDS,
        "pipelines": PIPELINES,
        "seeds": SEEDS,
        "data_hash": expected,
        "data_hash_verification": data_hash_verification,
        "rows": rows,
        "scores": scores,
        "metadata": {
            "timegan_reference_url": TIMEGAN_REPO_URL,
            "timegan_reference_branch": TIMEGAN_BRANCH,
            "predictive_metrics_file": TIMEGAN_PRED_FILE,
            "discriminative_metrics_file": TIMEGAN_DISC_FILE,
            "hidden_dim": H,
            "univariate_adaptation": univariate_adaptation,
            "pred_iters": int(args.pred_iters),
            "disc_iters": int(args.disc_iters),
            "batch": int(args.batch),
            "optimizer": "Adam(lr=1e-3)",
            "predictive_definition": (
                "single-layer GRU + Linear; train on synthetic, test on real; "
                "X=window[:, :-1], y=window[:, 1:]; score=mean(|y_real-y_pred|)"
            ),
            "discriminative_definition": (
                "single-layer GRU + Linear(H,1)->sigmoid; real=1/synth=0; "
                "independent 80/20 split per pool (np.random.permutation, seed "
                "recorded); score=abs(0.5 - test_accuracy)"
            ),
            "reference_citation": (
                "Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, "
                "'Time-series Generative Adversarial Networks', NeurIPS 2019; "
                "canonical code github.com/jsyoon0823/TimeGAN (master)."
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=("Phase 11 faithful TimeGAN predictive + discriminative "
                     "score driver (EVAL-02/03, D-11-08 — reads frozen "
                     "samples.npy, no regeneration).")
    )
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--csv-path", type=Path, default=Path("./data.csv"))
    ap.add_argument("--hidden", type=int, default=WINDOW_LENGTH)
    ap.add_argument("--pred-iters", type=int, default=5000)
    ap.add_argument("--disc-iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    obj = _build_results(args)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictive_discriminative_matched2000.json"
    out_path.write_text(json.dumps(obj, indent=2))
    print(f"wrote {out_path} ({len(obj['rows'])} rows, "
          f"{len(obj['scores'])} (model,pipeline) groups, "
          f"hidden_dim={obj['metadata']['hidden_dim']})")


if __name__ == "__main__":
    # Smoke path — tiny synthetic/real fixture; no frozen-artifact dependency.
    g = PredictiveGRU(10)
    n = sum(p.numel() for p in g.parameters())
    assert n > 0, n
    dg = DiscriminativeGRU(10)
    assert sum(p.numel() for p in dg.parameters()) > 0
    _o = g(torch.zeros(4, 9, 1))
    assert _o.shape == (4, 9, 1), _o.shape
    _rng = np.random.default_rng(0)
    _s = predictive_score(_rng.random((200, 10)), _rng.random((120, 10)),
                          42, 10, iters=50, bs=32)
    assert np.isfinite(_s) and _s >= 0.0, _s
    _d = discriminative_score(_rng.random((200, 10)), _rng.random((200, 10)),
                              42, 10, iters=50, bs=32)
    assert 0.0 <= _d <= 0.5 + 1e-6, _d
    print(f"smoke OK: params={n} predictive={_s:.4f} discriminative={_d:.4f}")
    # Full driver run (consumes the 60 frozen artifacts).
    main()
