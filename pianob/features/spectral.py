"""Feature spettrali: DE, DASM, RASM (su segmenti da 1 s aggregati per finestra) e log-power.

Ordine delle finestre identico a `data.windows.make_windows(..., overlap=0)`.
Calcolate su dati in microvolt (scale=1), così DE > 0 in pratica e RASM è ben definito.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from ..data import faced_meta as FM
from ..prep.filters import bandpass

BANDS5 = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 45)]


def _split_windows(x: np.ndarray, win: int) -> np.ndarray:
    """(n_tr, C, T) -> (n_tr, n_win, C, win) senza sovrapposizione."""
    n_tr, C, T = x.shape
    k = T // win
    return x[:, :, :k * win].reshape(n_tr, C, k, win).transpose(0, 2, 1, 3)


def differential_entropy(trials: np.ndarray, fs: int, win_s: float, seg_s: float = 1.0,
                         bands=BANDS5) -> np.ndarray:
    """DE = 0.5 log(2 pi e var) per banda, su segmenti da `seg_s` mediati nella finestra.
    Ritorna (n_tr, n_win, C, n_bands)."""
    win, seg = int(win_s * fs), int(seg_s * fs)
    assert win % seg == 0
    out = []
    for b in bands:
        xb = bandpass(trials, b, fs)
        w = _split_windows(xb, win)                                  # (n_tr, n_win, C, win)
        s = w.reshape(*w.shape[:3], win // seg, seg)
        var = s.var(axis=-1) + 1e-12
        de = 0.5 * np.log(2 * np.pi * np.e * var)
        out.append(de.mean(axis=-1))
    return np.stack(out, axis=-1).astype(np.float32)


def asymmetry(de: np.ndarray, channels: list[str], pairs=FM.EEG_SYMMETRIC_PAIRS):
    """DASM = DE_sx - DE_dx; RASM = DE_sx / DE_dx. de: (..., C, B)."""
    chans = [FM.canonical_channel(c) for c in channels]
    pairs = [(l, r) for l, r in pairs if l in chans and r in chans]
    li = [chans.index(l) for l, _ in pairs]
    ri = [chans.index(r) for _, r in pairs]
    L, R = de[..., li, :], de[..., ri, :]
    dasm = L - R
    safe = np.where(np.abs(R) < 1e-6, 1e-6, R)
    rasm = L / safe
    return dasm, rasm


def log_power(trials: np.ndarray, fs: int, win_s: float, bands=BANDS5) -> np.ndarray:
    """log della potenza di banda (Welch, segmenti da 1 s). Ritorna (n_tr, n_win, C, n_bands)."""
    w = _split_windows(trials, int(win_s * fs))
    f, p = welch(w, fs=fs, nperseg=min(fs, w.shape[-1]), axis=-1)
    out = [np.log(p[..., (f >= lo) & (f < hi)].mean(-1) + 1e-12) for lo, hi in bands]
    return np.stack(out, axis=-1).astype(np.float32)


def feature_matrix(kind: str, trials: np.ndarray, channels: list[str], fs: int, win_s: float) -> np.ndarray:
    """Ritorna (n_tr * n_win, F) nell'ordine trial-major delle finestre."""
    if kind == "logpower5":
        F = log_power(trials, fs, win_s)
        F = F.reshape(*F.shape[:2], -1)
    elif kind == "de_asym":
        de = differential_entropy(trials, fs, win_s)
        dasm, rasm = asymmetry(de, channels)
        F = np.concatenate([de.reshape(*de.shape[:2], -1), dasm.reshape(*dasm.shape[:2], -1),
                            rasm.reshape(*rasm.shape[:2], -1)], axis=-1)
    else:
        raise ValueError(kind)
    return F.reshape(-1, F.shape[-1])
