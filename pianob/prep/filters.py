"""Filtri zero-phase e selezione dei canali per le ablazioni EMG."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..data import faced_meta as FM


def bandpass(x: np.ndarray, band, fs: float, order: int = 4) -> np.ndarray:
    """Band-pass Butterworth zero-phase (sosfiltfilt) lungo l'ultimo asse, sull'intero trial
    (prima della finestratura, per evitare effetti di bordo per finestra)."""
    lo, hi = band
    if lo <= 0:      # solo passa-basso (es. [0, 30])
        sos = butter(order, hi, btype="lowpass", fs=fs, output="sos")
    else:
        sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def channel_drop_set(spec: str | None) -> list[str]:
    """Nomi dei canali da rimuovere.

    - None / "none": nessuno
    - "edge8": canali di bordo {Fp1, Fp2, F7, F8, FC5, FC6, T7, T8}
    - "edge8_mastoids": edge8 + A1, A2
    - "rand8_d{k}": 4 coppie simmetriche NON di bordo estratte con seed k (controllo appaiato, 8 canali)
    """
    if spec in (None, "", "none"):
        return []
    if spec == "edge8":
        return list(FM.EMG_EDGE_CHANNELS)
    if spec == "edge8_mastoids":
        return list(FM.EMG_EDGE_CHANNELS) + ["A1", "A2"]
    if spec == "ocular4":     # bordo frontale: Fp1, Fp2, F7, F8 (più esposti ai residui oculari)
        return ["FP1", "FP2", "F7", "F8"]
    if spec == "temporal4":   # bordo temporale: FC5, FC6, T7, T8 (più esposti all'EMG dei temporali)
        return ["FC5", "FC6", "T7", "T8"]
    if spec.startswith("rand4_d"):   # controllo: 2 coppie simmetriche non di bordo
        k = int(spec[len("rand4_d"):])
        rng = np.random.default_rng(2_000 + k)
        pool = FM.EMG_CONTROL_PAIR_POOL
        pick = rng.choice(len(pool), size=2, replace=False)
        return [ch for i in sorted(pick) for ch in pool[i]]
    if spec.startswith("rand8_d"):
        k = int(spec[len("rand8_d"):])
        rng = np.random.default_rng(1_000 + k)
        pool = FM.EMG_CONTROL_PAIR_POOL
        pick = rng.choice(len(pool), size=4, replace=False)
        return [ch for i in sorted(pick) for ch in pool[i]]
    raise ValueError(f"set di canali sconosciuto: {spec}")


def kept_channels(all_channels: list[str], drop_spec: str | None) -> list[str]:
    drop = {FM.canonical_channel(c) for c in channel_drop_set(drop_spec)}
    allc = [FM.canonical_channel(c) for c in all_channels]
    missing = drop - set(allc)
    assert not missing, f"canali da rimuovere assenti: {missing}"
    return [c for c in allc if c not in drop]
