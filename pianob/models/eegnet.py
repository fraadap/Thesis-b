"""EEGNet (Lawhern et al. 2018), implementazione a mano. Input (B, C, T)."""
from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int, fs: int = 200,
                 F1: int = 8, D: int = 2, F2: int | None = None, dropout: float = 0.5,
                 kern_s: float = 0.5, max_norm: bool = False):
        super().__init__()
        # Vincoli max-norm del paper originale (Lawhern et al. 2018): norma <= 1 per ciascun filtro spaziale
        # (depthwise) e <= 0.25 per ciascuna unità del classificatore
        self.max_norm = max_norm
        F2 = F2 or F1 * D
        k1 = int(kern_s * fs) | 1  # dispari, per padding simmetrico
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        k2 = int(fs / 8) | 1
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, k2), padding=(0, k2 // 2), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            n_feat = self.embed(torch.zeros(1, n_channels, n_samples)).shape[1]
        self.fc = nn.Linear(n_feat, n_classes)

    @torch.no_grad()
    def apply_constraints(self) -> None:
        if not self.max_norm:
            return
        dw = self.block1[2].weight          # (F1*D, 1, C, 1)
        dw.copy_(torch.renorm(dw, p=2, dim=0, maxnorm=1.0))
        self.fc.weight.copy_(torch.renorm(self.fc.weight, p=2, dim=0, maxnorm=0.25))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block2(self.block1(x.unsqueeze(1)))
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(x))
