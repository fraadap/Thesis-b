"""Fatti statici su FACED, con la fonte di ciascuno.

Verificati (nov 2026):
- mappa clip -> classe: `preprocessing/preprocessing_faced.py` del repo ufficiale CBraMod
  (`labels = [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8]`) e `_EMOTION_DICT`
  di TorchEEG (`torcheeg/datasets/emotion_recognition/faced.py`): 0 anger, 1 disgust,
  2 fear, 3 sadness, 4 neutral, 5 amusement, 6 inspiration, 7 joy, 8 tenderness.
- valenza: clip 1-12 negative, 13-16 neutre, 17-28 positive (`_VALENCE_DICT` di TorchEEG).
- ordine dei primi 30 canali (EEG): lista `FACEDDataset.channels` di TorchEEG; i canali
  31-32 sono i mastoidi A1/A2 (docstring TorchEEG: "30 channels ... and two channels of
  left/right mastoid signals"). L'ordine A1/A2 vs A2/A1 NON è documentato: [DA VERIFICARE].
  I .pkl processed sono array nudi senza nomi di canale, quindi l'ordine non si legge dal file.
- scala d'ampiezza CBraMod: `datasets/faced_dataset.py` restituisce `data/100`.
- ricampionamento CBraMod: `scipy.signal.resample(array, 6000, axis=2)` (FFT, sull'intero
  segmento da 30 s), poi reshape (28, 32, 30, 200) e 3 finestre da 10 s non sovrapposte.
- split CBraMod: file ordinati, `files[:80]` train, `[80:100]` val, `[100:]` test.
"""
from __future__ import annotations

N_SUBJECTS = 123
N_CLIPS = 28
FS_PROCESSED = 250
FS_TARGET = 200
N_SAMPLES_PROCESSED = 7500  # 30 s @ 250 Hz
N_SAMPLES_TARGET = 6000     # 30 s @ 200 Hz

CHANNELS = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1",
    "FC2", "FC5", "FC6", "CZ", "C3", "C4", "T7", "T8",
    "CP1", "CP2", "CP5", "CP6", "PZ", "P3", "P4", "P7",
    "P8", "PO3", "PO4", "OZ", "O1", "O2",
    "A1", "A2",  # mastoidi; ordine [DA VERIFICARE]
]
EEG_CHANNELS = CHANNELS[:30]

# Alias 10-20 vecchia nomenclatura -> nuova (T3/T4/T5/T6).
CHANNEL_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8", "FP1": "FP1", "FP2": "FP2"}

CATEGORIES = ["anger", "disgust", "fear", "sadness", "neutral",
              "amusement", "inspiration", "joy", "tenderness"]
CLIP_LABELS = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
NEUTRAL_CLASS = 4
# valenza per clip: -1 negativa, 0 neutra, +1 positiva
CLIP_VALENCE = [-1] * 12 + [0] * 4 + [1] * 12


def clip_index_in_category(labels=CLIP_LABELS) -> list[int]:
    out, seen = [], {}
    for lab in labels:
        out.append(seen.get(lab, 0))
        seen[lab] = seen.get(lab, 0) + 1
    return out


# Coppie simmetriche sinistra/destra (per DASM/RASM e per i kernel emisferici di TSception).
SYMMETRIC_PAIRS = [
    ("FP1", "FP2"), ("F7", "F8"), ("F3", "F4"), ("FC5", "FC6"), ("FC1", "FC2"),
    ("T7", "T8"), ("C3", "C4"), ("CP5", "CP6"), ("CP1", "CP2"), ("P7", "P8"),
    ("P3", "P4"), ("PO3", "PO4"), ("O1", "O2"), ("A1", "A2"),
]
EEG_SYMMETRIC_PAIRS = [p for p in SYMMETRIC_PAIRS if p[0] != "A1"]

# Canali di bordo per l'ablazione EMG (Goncharova 2003; Whitham 2007).
EMG_EDGE_CHANNELS = ["FP1", "FP2", "F7", "F8", "FC5", "FC6", "T7", "T8"]
# Pool per il controllo appaiato: coppie simmetriche NON di bordo. Escludiamo A1/A2 (mastoidi,
# vicini ai temporali) e P7/P8 (temporo-parietali laterali): si estraggono 4 coppie = 8 canali,
# così il controllo ha la stessa struttura (4 coppie) del set di bordo.
EMG_CONTROL_PAIR_POOL = [("F3", "F4"), ("FC1", "FC2"), ("C3", "C4"), ("CP1", "CP2"),
                         ("CP5", "CP6"), ("P3", "P4"), ("PO3", "PO4"), ("O1", "O2")]


def canonical_channel(name: str) -> str:
    n = name.strip().upper()
    return CHANNEL_ALIASES.get(n, n)
