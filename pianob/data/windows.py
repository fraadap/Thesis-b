"""Finestratura dei trial. Le finestre sono indici (trial, inizio): non si materializzano copie.

Ogni finestra sta dentro un singolo trial (soggetto x clip), quindi non può stare a cavallo di
due trial né di due split. La sovrapposizione si usa solo in training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def window_starts(n_samples: int, win: int, stride: int) -> np.ndarray:
    assert 0 < win <= n_samples, (win, n_samples)
    return np.arange(0, n_samples - win + 1, stride, dtype=np.int64)


@dataclass
class WindowSet:
    trial_subject: np.ndarray   # (n_trials,)
    trial_clip: np.ndarray
    trial_label: np.ndarray
    win_trial: np.ndarray       # (n_win,) indice del trial
    win_start: np.ndarray       # (n_win,) campione iniziale
    win_len: int

    @property
    def n(self) -> int:
        return len(self.win_trial)

    @property
    def subject(self) -> np.ndarray:
        return self.trial_subject[self.win_trial]

    @property
    def clip(self) -> np.ndarray:
        return self.trial_clip[self.win_trial]

    @property
    def label(self) -> np.ndarray:
        return self.trial_label[self.win_trial]

    def subset(self, mask) -> "WindowSet":
        return WindowSet(self.trial_subject, self.trial_clip, self.trial_label,
                         self.win_trial[mask], self.win_start[mask], self.win_len)


def make_windows(trial_subject, trial_clip, trial_label, n_samples: int, win_len: int,
                 overlap: float = 0.0) -> WindowSet:
    """Finestre ordinate per trial e poi nel tempo (ordine naturale: clip 0..27, inizio crescente)."""
    stride = max(1, int(round(win_len * (1.0 - overlap))))
    starts = window_starts(n_samples, win_len, stride)
    n_tr = len(trial_subject)
    win_trial = np.repeat(np.arange(n_tr), len(starts))
    win_start = np.tile(starts, n_tr)
    return WindowSet(np.asarray(trial_subject), np.asarray(trial_clip), np.asarray(trial_label),
                     win_trial, win_start, win_len)


def materialize(data: np.ndarray, ws: WindowSet, idx=None) -> np.ndarray:
    """data: (n_trials, C, T) -> (n, C, win_len). Solo per CPU/feature e test."""
    idx = np.arange(ws.n) if idx is None else np.asarray(idx)
    return np.stack([data[ws.win_trial[i], :, ws.win_start[i]:ws.win_start[i] + ws.win_len] for i in idx])


def gather_windows_torch(data, trial_idx, starts, win_len: int):
    """data: tensor (n_trials, C, T) anche su GPU; restituisce (B, C, win_len) senza loop Python."""
    import torch
    ar = torch.arange(win_len, device=data.device)
    t_idx = (starts.to(data.device)[:, None] + ar[None, :])                     # (B, L)
    x = data[trial_idx.to(data.device)]                                         # (B, C, T)
    return torch.gather(x, 2, t_idx[:, None, :].expand(-1, x.shape[1], -1))
