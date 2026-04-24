"""Classical 1D-CNN critic with configurable dropout (v1.1 Phase 7).

Extracted verbatim from ``qgan_pennylane.ipynb`` cell 26
(``qGAN.define_critic_model``). Dropout rate is a constructor kwarg so downstream
phases can sweep it without editing the class (v1.1 Phase 7 decision).

Expected input shape (from cell 26 training loop, e.g. ``real_batch_tensor``):
    ``(batch_size, 1, window_length)`` — a 1-channel log-returns window.
    dtype ``torch.float64`` (notebook calls ``.double()`` on the model).

Output shape: ``(batch_size, 1)`` — scalar critic score per sample.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Critic(nn.Module):
    """1D-CNN critic with adaptive pooling + configurable dropout.

    Architecture (verbatim from cell 26 ``define_critic_model``):
        Conv1d(1 -> 64,   k=10, s=1, p=5) + LeakyReLU(0.1)
        Conv1d(64 -> 128, k=10, s=1, p=5) + LeakyReLU(0.1)
        Conv1d(128 -> 128, k=10, s=1, p=5) + LeakyReLU(0.1)
        AdaptiveAvgPool1d(output_size=1)
        Flatten
        Linear(128 -> 32) + LeakyReLU(0.1) + Dropout(p=dropout_rate)
        Linear(32 -> 1)

    Notes:
        - Adaptive pool makes the network robust to variable ``window_length``.
        - Only one Dropout layer in the notebook — placed between the hidden
          dense and the output dense. Do not add more.
        - Model is cast to ``double()`` to match the notebook's float64 tensors.
    """

    def __init__(self, window_length: int = 10, dropout_rate: float = 0.2) -> None:
        super().__init__()

        self.window_length = window_length
        self.dropout_rate = dropout_rate

        self.net = nn.Sequential(
            # 1 channel: log-returns only.
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=10, stride=1, padding=5),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, stride=1, padding=5),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=10, stride=1, padding=5),
            nn.LeakyReLU(negative_slope=0.1),

            # Adaptive pooling -> fixed-size representation regardless of window length.
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),

            nn.Linear(in_features=128, out_features=32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_rate),

            nn.Linear(in_features=32, out_features=1),
        )

        # Match cell 26: model cast to float64.
        self.net = self.net.double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar critic score per sample.

        Expected input: ``(batch, 1, window_length)`` (matches notebook
        ``real_batch_tensor`` and ``fake_batch_tensor`` shapes).
        Output: ``(batch, 1)``.
        """
        return self.net(x)
