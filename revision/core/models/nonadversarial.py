"""Non-adversarial baselines — VAE + AR (BASE-02, D-10-03/09/13).

Model DEFINITIONS ONLY (D-10-13). The ELBO training loop, the AR
least-squares fit *orchestration*, checkpoint serialization, and sample
saving all live in ``revision/run_baselines.py`` (Wave 2) — NOT here.

Two baselines:

``VAEBaseline`` (an ``nn.Module``)
    The "smallest deep VAE that trains stably" on length-10 windows
    (RESEARCH §"VAE Sizing", lines 217-229). NOT parameter-matched to the
    quantum generator (D-10-03) — its ~562-param size is reported
    transparently in the comparison table's ``models[]`` array.

        encoder : Linear(10,16) -> ReLU -> [Linear(16,4) mu, Linear(16,4) logvar]
        latent  : Lz = 4 ;  z = mu + eps*exp(0.5*logvar), eps~N(0,I)
        decoder : Linear(4,16) -> ReLU -> Linear(16,10)   (no final activation)

    Param count:
        enc Linear(10,16)=176 ; mu Linear(16,4)=68 ; logvar Linear(16,4)=68
        dec Linear(4,16)=80   ; Linear(16,10)=170
        total = 562

    ``count_params()`` mirrors ``quantum.py:80-85`` and returns
    ``sum(p.numel() for p in self.parameters())``.

``ARBaseline`` (a PLAIN class, NOT an ``nn.Module``)
    Autoregressive linear model of order ``p`` (default ``p=2``) on the
    windowed series (RESEARCH §"AR Baseline", lines 241-267).
    Parameter-minimal by definition: ``p`` AR coefficients + 1 noise
    variance = ``p+1`` params (== 3 for ``p=2``), reported as a plain int.

Sample-space asymmetry (RESEARCH Pitfall 3, lines 237-239 / 473-478):
    VAE/AR do NOT go through ``train_wgan_gp``, so they do NOT inherit the
    ``*0.1`` post-scaling at ``training.py:283``. The ``*0.1`` is a
    quantum-output-magnitude artifact (PauliX/PauliZ expectations live in
    [-1,1]), NOT part of the preprocessing inverse. Both ``VAEBaseline.sample``
    and ``ARBaseline.sample`` therefore emit samples directly in the ``[-1,1]``
    window space the shared ``reconstruct_od`` inverse consumes — and explicitly
    do NOT apply ``*0.1``. A Wave-2 smoke gate (run_baselines.py) reconstructs
    one VAE sample and one WGAN sample through the identical pipeline inverse
    and asserts both land in the real OD range.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEBaseline(nn.Module):
    """Smallest deep VAE for length-10 windows (~562 params, D-10-03).

    Interface (ELBO loop lives in run_baselines.py per D-10-13):
        encode(x[B,10])        -> (mu[B,4], logvar[B,4])
        reparameterize(mu,lv)  -> z[B,4]  (mu + eps*exp(0.5*logvar))
        decode(z[B,4])         -> x_hat[B,10]
        forward(x[B,10])       -> (x_hat, mu, logvar)
        sample(n, gen)         -> (n,10)  in [-1,1] window space (NO *0.1)
    """

    LATENT_DIM = 4
    WINDOW = 10
    HIDDEN = 16

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(self.WINDOW, self.HIDDEN)
        self.fc_mu = nn.Linear(self.HIDDEN, self.LATENT_DIM)
        self.fc_logvar = nn.Linear(self.HIDDEN, self.LATENT_DIM)
        self.dec_h = nn.Linear(self.LATENT_DIM, self.HIDDEN)
        self.dec_out = nn.Linear(self.HIDDEN, self.WINDOW)

    def count_params(self) -> int:
        """Return total trainable parameter count (mirrors quantum.py:80-85)."""
        return sum(p.numel() for p in self.parameters())

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x[B,10] -> (mu[B,4], logvar[B,4])."""
        h = F.relu(self.enc(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """z = mu + eps * exp(0.5*logvar), eps ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z[B,4] -> x_hat[B,10] (no final activation; data in [-1,1])."""
        h = F.relu(self.dec_h(z))
        return self.dec_out(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x[B,10] -> (x_hat[B,10], mu[B,4], logvar[B,4])."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, gen: torch.Generator | None = None) -> torch.Tensor:
        """Draw z ~ N(0, I) of shape (n,4), decode -> (n,10).

        Output is in the [-1,1]-ish window space the shared ``reconstruct_od``
        inverse consumes. The ``*0.1`` scaling from ``training.py:283`` is a
        quantum-output artifact and is deliberately NOT applied here so VAE
        samples remain comparable to WGAN/quantum samples after the identical
        pipeline inverse (RESEARCH Pitfall 3).
        """
        device = self.dec_h.weight.device
        z = torch.randn(n, self.LATENT_DIM, generator=gen, device=device)
        return self.decode(z)


class ARBaseline:
    """AR(p) baseline (default p=2) — plain class, NOT an nn.Module.

    Parameter count is ``p + 1`` (p AR coefficients + 1 noise variance),
    reported as a plain int (== 3 for p=2). The least-squares ``fit`` is a
    closed-form solve (no training loop); ``sample`` recursively simulates
    length-10 windows. Checkpoint (.npz {phi,sigma2,p}) serialization is done
    by run_baselines.py in Wave 2, not here (D-10-13).
    """

    def __init__(self, p: int = 2) -> None:
        self.p = int(p)
        self.phi: np.ndarray | None = None
        self.sigma2: float | None = None

    def count_params(self) -> int:
        """Return p + 1 (AR coefficients + noise variance), a plain int."""
        return self.p + 1

    def fit(self, x_1d: np.ndarray) -> "ARBaseline":
        """Closed-form least-squares fit on a flattened 1-D series.

        x[t] ~= phi_1 x[t-1] + ... + phi_p x[t-p] + eps
        Solved once via np.linalg.lstsq (no iterative loop — D-10-13).
        """
        x = np.asarray(x_1d, dtype=np.float64).reshape(-1)
        p = self.p
        if x.shape[0] <= p + 1:
            raise ValueError(
                f"series too short ({x.shape[0]}) for AR({p}) fit"
            )
        # Design matrix of p lagged columns; column k is lag (k+1).
        X = np.stack([x[p - 1 - k : -1 - k] for k in range(p)], axis=1)
        y = x[p:]
        phi, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ phi
        self.phi = phi.astype(np.float64)
        self.sigma2 = float(resid.var(ddof=0))
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Recursively simulate n windows of length 10 -> (n,10).

        First p values seeded from zeros with a short burn-in so the
        recursion is stationary before the saved window begins. Output is
        in the same transformed/windowed space as the other models (NO
        ``*0.1`` — RESEARCH Pitfall 3).
        """
        if self.phi is None or self.sigma2 is None:
            raise RuntimeError("ARBaseline.sample called before fit()")
        p = self.p
        phi = self.phi
        sigma = float(np.sqrt(max(self.sigma2, 0.0)))
        burn = 50
        total = burn + 10
        out = np.empty((n, 10), dtype=np.float64)
        for i in range(n):
            buf = np.zeros(total + p, dtype=np.float64)
            for t in range(p, total + p):
                # buf[t-1 .. t-p] dotted with phi (phi[0] is lag-1 coef).
                # Build explicitly: negative-step slicing wraps at index 0.
                lags = buf[t - p : t][::-1]
                buf[t] = float(phi @ lags) + rng.normal(0.0, sigma)
            out[i] = buf[p + burn : p + burn + 10]
        return out
