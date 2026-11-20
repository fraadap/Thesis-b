"""TSception (Ding et al., IEEE TAC 2022), reimplementata seguendo il codice ufficiale
(deepBrains/TSception): kernel temporali multi-scala (0.5, 0.25, 0.125 s) + kernel spaziali
globale ed emisferico + fusione. Input (B, C, T) con i canali ordinati
[emisfero sinistro..., emisfero destro...] in coppie simmetriche corrispondenti.
`num_T`/`num_S` = numero di kernel temporali/spaziali (spazio Optuna {9,15,21} / {6,9,15}).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data import faced_meta as FM


def hemispheric_order(channels: list[str], pairs=FM.SYMMETRIC_PAIRS) -> list[int]:
    """Indici dei canali: prima tutti i sinistri poi i destri corrispondenti (escluse le linee mediane).
    Le coppie con un canale mancante (es. rimosso dall'ablazione) vengono scartate."""
    chans = [FM.canonical_channel(c) for c in channels]
    ok = [(l, r) for l, r in pairs if l in chans and r in chans]
    return [chans.index(l) for l, _ in ok] + [chans.index(r) for _, r in ok]


def _block(cin, cout, kernel, stride, pool):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride=stride), nn.LeakyReLU(),
                         nn.AvgPool2d((1, pool), stride=(1, pool)))


class TSception(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int, fs: int = 200,
                 num_T: int = 15, num_S: int = 15, hidden: int = 32, dropout: float = 0.5):
        super().__init__()
        assert n_channels % 2 == 0, "TSception richiede coppie emisferiche (numero pari di canali)"
        pool = 8
        self.T1 = _block(1, num_T, (1, int(0.5 * fs)), 1, pool)
        self.T2 = _block(1, num_T, (1, int(0.25 * fs)), 1, pool)
        self.T3 = _block(1, num_T, (1, int(0.125 * fs)), 1, pool)
        half = n_channels // 2
        self.S1 = _block(num_T, num_S, (n_channels, 1), 1, int(pool * 0.25))
        self.S2 = _block(num_T, num_S, (half, 1), (half, 1), int(pool * 0.25))
        self.fusion = _block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)
        self.fc = nn.Sequential(nn.Linear(num_S, hidden), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden, n_classes))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        out = torch.cat([self.T1(x), self.T2(x), self.T3(x)], dim=-1)
        out = self.BN_t(out)
        out = torch.cat([self.S1(out), self.S2(out)], dim=2)
        out = self.BN_s(out)
        out = self.BN_fusion(self.fusion(out))
        return out.mean(dim=-1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(x))
