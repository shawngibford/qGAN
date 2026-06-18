"""Classical WGAN-GP generators — matched-parameter baselines (BASE-01).

These three generators are the classical analogs of ``quantum.QuantumGenerator``.
They satisfy the *exact same* ``train_wgan_gp`` interface contract so the WGAN-GP
training loop (``core/training.py``) runs them unchanged:

    - class attributes ``num_qubits = 5`` and ``window_length = 10``
      (``training.py:228-229`` reads these via ``getattr``)
    - a SINGLE live ``nn.Parameter`` named ``self.params_pqc``
      (``training.py:234`` builds ``torch.optim.Adam([generator.params_pqc])`` —
      it steps exactly one tensor, never ``generator.parameters()``)
    - ``forward(noise)`` accepting ``(5, B)`` noise and returning ``(B, 10)``
      (``training.py:282`` calls ``generator(noise_batch)`` then ``.to(float64) * 0.1``)
    - ``count_params() -> int`` mirroring ``quantum.py:80-85``

Design rationale (RESEARCH §"Pitfall 1", lines 456-464): the quantum generator
holds its *one and only* trainable tensor as a flat ``(75,)`` ``nn.Parameter`` and
implements the circuit *functionally* off slices of it. A naive classical
``nn.Module`` with multiple ``nn.Parameter``s (or a ``@property`` returning a
flattened *copy*) would detach autograd from the tensor the optimizer holds —
the generator would never learn (silent failure). We therefore mirror the
quantum design exactly: each classical generator stores ALL trainable weights as
one contiguous flat ``nn.Parameter`` (``self.params_pqc``) and ``forward`` slices
that vector into per-layer weight/bias views applied via ``torch.nn.functional``.
This makes ``count_params() == params_pqc.numel()`` trivially the matched count.

Parameter arithmetic (RESEARCH §"Standard Stack", lines 156-211), latent dim 5,
output length 10, target 71..79 (quantum = 75, ±5% per D-10-02):

    MLP : Linear(5,4)+b -> Tanh -> Linear(4,10)+b   = (5*4+4)+(4*10+10)        = 74
    CNN : view(B,1,5) -> ConvT1d(1,9,k=6,s=1)+b -> LeakyReLU
                       -> Conv1d(9,1,k=1)+b -> view(B,10)
                                                    = (1*9*6+9)+(9*1*1+1)      = 73
    LSTM: view(B,3,2) -> functional LSTM(I=2,H=2,1 layer,bias) -> last h(B,2)
                       -> Linear(2,10)+b             = 4*(2*2+2*2+2+2)+(2*10+10)= 78

The ``(5, B)`` noise is the latent input; each generator transposes it to
``(B, 5)`` and treats the row as a 5-dim latent vector. CNN/LSTM reshape the
``(B, 5)`` latent into their expected input shape with a *parameter-free*
deterministic tile/pad (a learnable projection would add params and blow the
budget — RESEARCH line 423).

Pitfall 2 (PyTorch two-bias-vector LSTM trap) does not apply here because the
LSTM cell is carved by hand from ``params_pqc`` with exactly one bias per gate
group; the empirical ``sum(p.numel())`` assertion in the acceptance test is the
ground truth regardless.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Mirror quantum.py's small-scale init style (``torch.randn(...) * 0.5``).
_INIT_SCALE = 0.5


class WGANMLPGenerator(nn.Module):
    """MLP generator — Linear(5,4)+b -> Tanh -> Linear(4,10)+b. 74 params.

    ``params_pqc`` flat layout (length 74), in order:
        [0:20]   W1  -> (4, 5)   Linear(5, 4) weight
        [20:24]  b1  -> (4,)     Linear(5, 4) bias
        [24:64]  W2  -> (10, 4)  Linear(4, 10) weight
        [64:74]  b2  -> (10,)    Linear(4, 10) bias

    Tanh between layers (zero params) bounds output toward [-1,1], matching the
    generator-output range the training loop scales by 0.1.
    """

    num_qubits = 5
    window_length = 10

    def __init__(self) -> None:
        super().__init__()
        self._n = 74
        self.params_pqc = nn.Parameter(
            torch.randn(self._n, dtype=torch.float32) * _INIT_SCALE,
            requires_grad=True,
        )

    def count_params(self) -> int:
        """Return total trainable parameter count (== params_pqc.numel())."""
        return self.params_pqc.numel()

    def forward(self, noise_params: torch.Tensor) -> torch.Tensor:
        """(5, B) noise -> (B, 10) window."""
        x = noise_params.t()  # (B, 5)
        p = self.params_pqc
        w1 = p[0:20].view(4, 5)
        b1 = p[20:24]
        w2 = p[24:64].view(10, 4)
        b2 = p[64:74]
        h = torch.tanh(F.linear(x, w1, b1))  # (B, 4)
        return F.linear(h, w2, b2)  # (B, 10)


class WGANCNNGenerator(nn.Module):
    """1D-CNN generator — ConvTranspose1d(1,9,k=6,s=1) -> LeakyReLU -> Conv1d(9,1,1). 73 params.

    Parameter-free reshape: latent (B,5) -> (B,1,5). ConvTranspose1d with
    ``L_in=5, stride=1, pad=0, k=6`` gives ``L_out = (5-1)*1 - 0 + 6 = 10``.

    ``params_pqc`` flat layout (length 73), in order:
        [0:54]   Wt  -> (1, 9, 6)  ConvTranspose1d weight (in, out, k)
        [54:63]  bt  -> (9,)       ConvTranspose1d bias
        [63:72]  Wc  -> (1, 9, 1)  Conv1d weight (out, in, k)
        [72:73]  bc  -> (1,)       Conv1d bias

    LeakyReLU(0.1) between (zero params).
    """

    num_qubits = 5
    window_length = 10

    def __init__(self) -> None:
        super().__init__()
        self._n = 73
        self.params_pqc = nn.Parameter(
            torch.randn(self._n, dtype=torch.float32) * _INIT_SCALE,
            requires_grad=True,
        )

    def count_params(self) -> int:
        """Return total trainable parameter count (== params_pqc.numel())."""
        return self.params_pqc.numel()

    def forward(self, noise_params: torch.Tensor) -> torch.Tensor:
        """(5, B) noise -> (B, 10) window."""
        x = noise_params.t()  # (B, 5)
        B = x.shape[0]
        x = x.reshape(B, 1, 5)  # parameter-free reshape (NOT a learnable proj)
        p = self.params_pqc
        wt = p[0:54].view(1, 9, 6)  # ConvTranspose1d(in=1, out=9, k=6)
        bt = p[54:63]
        wc = p[63:72].view(1, 9, 1)  # Conv1d(out=1, in=9, k=1)
        bc = p[72:73]
        y = F.conv_transpose1d(x, wt, bt, stride=1, padding=0)  # (B, 9, 10)
        y = F.leaky_relu(y, 0.1)
        y = F.conv1d(y, wc, bc)  # (B, 1, 10)
        return y.reshape(B, 10)


class WGANLSTMGenerator(nn.Module):
    """LSTM generator — functional LSTM(I=2,H=2,1 layer,bias) -> Linear(2,10)+b. 78 params.

    Parameter-free reshape: latent (B,5) is tiled/padded to length 6, then
    viewed as a sequence ``(B, seq=3, input=2)``. The LSTM cell is carved by
    hand from ``params_pqc`` (NOT ``nn.LSTM`` holding its own parameters) so all
    weights live in the single ``params_pqc`` tensor.

    LSTM gate stacking order is PyTorch's (i, f, g, o). ``params_pqc`` flat
    layout (length 78), in order:
        [0:16]   W_ih -> (8, 2)   input->hidden  (4 gates * H=2 rows, I=2 cols)
        [16:32]  W_hh -> (8, 2)   hidden->hidden (4 gates * H=2 rows, H=2 cols)
        [32:40]  b_ih -> (8,)     input bias  (4 gates * H=2)
        [40:48]  b_hh -> (8,)     hidden bias (4 gates * H=2)
        [48:68]  Wout -> (10, 2)  Linear(2, 10) weight
        [68:78]  bout -> (10,)    Linear(2, 10) bias
    LSTM params = 16+16+8+8 = 48 ; Linear = 20+10 = 30 ; total 78.
    """

    num_qubits = 5
    window_length = 10

    _H = 2
    _I = 2
    _SEQ = 3

    def __init__(self) -> None:
        super().__init__()
        self._n = 78
        self.params_pqc = nn.Parameter(
            torch.randn(self._n, dtype=torch.float32) * _INIT_SCALE,
            requires_grad=True,
        )

    def count_params(self) -> int:
        """Return total trainable parameter count (== params_pqc.numel())."""
        return self.params_pqc.numel()

    def forward(self, noise_params: torch.Tensor) -> torch.Tensor:
        """(5, B) noise -> (B, 10) window."""
        x = noise_params.t()  # (B, 5)
        B = x.shape[0]
        # Parameter-free tile/pad (B,5) -> (B,6) -> (B, seq=3, input=2).
        pad = x[:, :1]  # repeat first latent dim to reach length 6
        seq_in = torch.cat([x, pad], dim=1).reshape(B, self._SEQ, self._I)

        p = self.params_pqc
        H = self._H
        w_ih = p[0:16].view(4 * H, self._I)  # (8, 2)
        w_hh = p[16:32].view(4 * H, H)  # (8, 2)
        b_ih = p[32:40]  # (8,)
        b_hh = p[40:48]  # (8,)
        wout = p[48:68].view(10, H)  # Linear(2, 10) weight
        bout = p[68:78]  # Linear(2, 10) bias

        h_t = x.new_zeros(B, H)
        c_t = x.new_zeros(B, H)
        for t in range(self._SEQ):
            xt = seq_in[:, t, :]  # (B, 2)
            gates = F.linear(xt, w_ih, b_ih) + F.linear(h_t, w_hh, b_hh)  # (B, 8)
            i, f, g, o = gates.chunk(4, dim=1)  # PyTorch gate order
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)

        return F.linear(h_t, wout, bout)  # (B, 10)
