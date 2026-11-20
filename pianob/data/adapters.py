"""Interfaccia generica dei dataset e implementazione FACED.

Ogni adapter espone trial (soggetto x stimolo) come array (canali, campioni) a fs fissa.
Le label sono definite per stimolo (clip) e per task.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import numpy as np

from . import faced_meta as FM
from .. import kaggle_env


class DatasetAdapter(ABC):
    name: str = "base"
    fs: int = 200

    @abstractmethod
    def load(self) -> "DatasetAdapter": ...

    @abstractmethod
    def subjects(self) -> list[int]: ...

    @abstractmethod
    def stimuli(self) -> list[int]: ...

    @abstractmethod
    def labels(self, task: str) -> np.ndarray:
        """Label per stimolo; -1 = stimolo escluso dal task."""

    @abstractmethod
    def channel_names(self) -> list[str]: ...

    @abstractmethod
    def trials(self, subjects, clips=None, channels=None, band=None, scale=None,
               dtype=np.float32) -> np.ndarray:
        """Array (n_subj, n_clips, n_chan, n_samples)."""

    def trial_iter(self, subjects=None, clips=None, **kw) -> Iterator[tuple[int, int, np.ndarray]]:
        subjects = self.subjects() if subjects is None else subjects
        clips = self.stimuli() if clips is None else clips
        for s in subjects:
            arr = self.trials([s], clips, **kw)[0]
            for j, c in enumerate(clips):
                yield s, c, arr[j]

    def task_clips(self, task: str) -> list[int]:
        lab = self.labels(task)
        return [c for c in self.stimuli() if lab[c] >= 0]

    def n_classes(self, task: str) -> int:
        lab = self.labels(task)
        return int(lab[lab >= 0].max() + 1)

    # Hook per la Proposta 2 (ISC/MIL): tutte le finestre di un trial come bag.
    def trial_bag(self, subject: int, clip: int, win_samples: int, stride: int | None = None, **kw):
        x = self.trials([subject], [clip], **kw)[0, 0]
        stride = stride or win_samples
        starts = np.arange(0, x.shape[-1] - win_samples + 1, stride)
        return np.stack([x[:, s:s + win_samples] for s in starts])


def _task_labels(task: str, clip_labels, clip_valence) -> np.ndarray:
    clip_labels = np.asarray(clip_labels)
    if task == "emo9":
        return clip_labels.copy()
    if task == "val2":  # positiva (1) contro negativa (0), neutre escluse (-1), come FACED-2
        v = np.asarray(clip_valence)
        return np.where(v > 0, 1, np.where(v < 0, 0, -1))
    raise ValueError(f"task sconosciuto: {task}")


class FacedAdapter(DatasetAdapter):
    """FACED processed (ultimi 30 s di ogni clip, 0.05-47 Hz, ICA oculare, CAR), ricampionato a 200 Hz.

    `scale` di default 0.01 = convenzione CBraMod (`data/100`, verificata su `faced_dataset.py`).
    Le feature spettrali usano scale=1 (microvolt).
    """
    name = "faced"
    fs = FM.FS_TARGET
    DATA_FILE = "faced_200hz_f16.npy"

    def __init__(self, root: str | Path | None = None, synthetic: bool = False, scale: float = 0.01,
                 source: str = "packed", synth_subjects: int = FM.N_SUBJECTS, cache_dir: Path | None = None):
        self.root = Path(root) if root else None
        self.synthetic = synthetic
        self.default_scale = scale
        self.source = source
        self.synth_subjects = synth_subjects
        self.meta: dict = {}
        self._data = None
        self._band_cache: dict = {}
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------ load
    def load(self) -> "FacedAdapter":
        if self.source == "faced_raw":
            raise NotImplementedError("Sorgente 'faced_raw' (raw + Picard + ICLabel): prevista nella sessione successiva.")
        if self.synthetic:
            self.meta = synthetic_meta(self.synth_subjects)
            return self
        meta_path = self._locate_meta()
        self.meta = json.loads(meta_path.read_text())
        data_path = meta_path.parent / self.DATA_FILE
        assert data_path.exists(), f"Manca {data_path}: il pacchetto dati è incompleto."
        self._data = np.load(data_path, mmap_mode="r")
        exp = (self.meta["n_subjects_slots"], FM.N_CLIPS, len(self.meta["channels"]), FM.N_SAMPLES_TARGET)
        assert tuple(self._data.shape) == exp, f"shape {self._data.shape} != attesa {exp}"
        assert self._data.dtype == np.float16, self._data.dtype
        assert self.meta["fs"] == self.fs, self.meta["fs"]
        assert self.meta["clip_labels"] == FM.CLIP_LABELS, "mappa clip->classe diversa da quella verificata"
        return self

    def _locate_meta(self) -> Path:
        if self.root is not None:
            cands = [self.root / "meta.json", self.root / "faced" / "meta.json"]
            cands += list(self.root.rglob("meta.json"))
        else:
            cands = kaggle_env.find_all("meta.json")
        for c in cands:
            if c.exists():
                try:
                    m = json.loads(c.read_text())
                except Exception:
                    continue
                if m.get("dataset") == "FACED" and (c.parent / self.DATA_FILE).exists():
                    return c
        raise FileNotFoundError(
            "Pacchetto FACED non trovato: serve una cartella con meta.json (dataset=FACED) e "
            f"{self.DATA_FILE}. Su Kaggle: aggiungi il Dataset 'pianob-faced-data' come input.")

    # ------------------------------------------------------------- interfaccia
    def subjects(self) -> list[int]:
        return list(self.meta["subjects_present"])

    def stimuli(self) -> list[int]:
        return list(range(FM.N_CLIPS))

    def labels(self, task: str) -> np.ndarray:
        return _task_labels(task, self.meta["clip_labels"], self.meta["clip_valence"])

    def channel_names(self) -> list[str]:
        return list(self.meta["channels"])

    def channel_indices(self, names) -> list[int]:
        chans = [FM.canonical_channel(c) for c in self.channel_names()]
        idx = []
        for n in names:
            n = FM.canonical_channel(n)
            assert n in chans, f"canale {n} assente in FACED ({chans})"
            idx.append(chans.index(n))
        return idx

    def trials(self, subjects, clips=None, channels=None, band=None, scale=None, dtype=np.float32):
        subjects = list(subjects)
        clips = self.stimuli() if clips is None else list(clips)
        scale = self.default_scale if scale is None else scale
        missing = set(subjects) - set(self.subjects())
        assert not missing, f"soggetti mancanti nel pacchetto: {sorted(missing)}"
        if self.synthetic:
            arr = np.stack([np.stack([synthetic_trial(s, c) for c in clips]) for s in subjects])
            if band is not None:
                from ..prep.filters import bandpass
                arr = bandpass(arr, band, self.fs)
        else:
            src = self._banded(band) if band is not None else self._data
            arr = np.asarray(src[np.asarray(subjects)][:, clips], dtype=np.float32)
        if channels is not None:
            arr = arr[:, :, list(channels)]
        if scale != 1.0:
            arr = arr * np.float32(scale)
        return arr.astype(dtype, copy=False)

    def _banded(self, band):
        """Versione filtrata dell'intero dataset, calcolata una volta e messa in cache su disco."""
        key = tuple(float(b) for b in band)
        if key in self._band_cache:
            return self._band_cache[key]
        from ..prep.filters import bandpass
        cache_dir = self.cache_dir or kaggle_env.cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        tag = hashlib.md5(json.dumps([key, self.meta.get("checksum_data")]).encode()).hexdigest()[:8]
        path = cache_dir / f"faced_bp{key[0]:g}-{key[1]:g}_{tag}_f16.npy"
        if not path.exists():
            print(f"[filtri] calcolo band-pass {key} Hz su tutto FACED -> {path.name}")
            import os
            tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")   # due worker: nessuna collisione
            out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=self._data.shape)
            for s in range(self._data.shape[0]):
                if s in set(self.subjects()):
                    out[s] = bandpass(np.asarray(self._data[s], dtype=np.float32), key, self.fs).astype(np.float16)
                else:
                    out[s] = np.nan
            out.flush()
            del out
            os.replace(tmp, path)
        self._band_cache[key] = np.load(path, mmap_mode="r")
        return self._band_cache[key]

    def clip_table(self) -> list[dict]:
        idx = FM.clip_index_in_category(self.meta["clip_labels"])
        return [{"clip": c, "label": self.meta["clip_labels"][c],
                 "category": FM.CATEGORIES[self.meta["clip_labels"][c]],
                 "valence": self.meta["clip_valence"][c], "idx_in_category": idx[c]}
                for c in self.stimuli()]


# ---------------------------------------------------------------- sintetico
def synthetic_meta(n_subjects: int = FM.N_SUBJECTS) -> dict:
    return {"dataset": "FACED", "synthetic": True, "fs": FM.FS_TARGET, "channels": FM.CHANNELS,
            "clip_labels": FM.CLIP_LABELS, "clip_valence": FM.CLIP_VALENCE,
            "categories": FM.CATEGORIES, "subjects_present": list(range(n_subjects)),
            "n_subjects_slots": n_subjects, "missing": []}


def synthetic_trial(subject: int, clip: int, n_samples: int = FM.N_SAMPLES_TARGET, fs: int = FM.FS_TARGET):
    """Trial sintetico deterministico (microvolt): rumore 1/f + ritmo dipendente dalla classe +
    firma della clip + guadagni per canale del soggetto (così l'allineamento ha qualcosa da fare)."""
    rng = np.random.default_rng(subject * 1000 + clip)
    n_ch = len(FM.CHANNELS)
    white = rng.standard_normal((n_ch, n_samples)).astype(np.float32)
    f = np.fft.rfftfreq(n_samples, 1 / fs)
    spec = np.fft.rfft(white, axis=-1) / np.maximum(f, 1.0) ** 0.5
    x = np.fft.irfft(spec, n=n_samples, axis=-1).astype(np.float32) * 10
    t = np.arange(n_samples) / fs
    lab = FM.CLIP_LABELS[clip]
    srng = np.random.default_rng(10_000 + subject)
    gains = srng.uniform(0.5, 2.0, n_ch).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi)
    freq = 6.0 + 2.5 * lab
    chans = np.arange(lab % 4, n_ch, 4)
    x[chans] += 6.0 * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
    crng = np.random.default_rng(50_000 + clip)
    x += (crng.standard_normal((n_ch, 1)) * 2.0 * np.sin(2 * np.pi * (3 + clip * 0.5) * t)).astype(np.float32)
    return x * gains[:, None]


# ---------------------------------------------------------------- stub per i dataset successivi
class _StubAdapter(DatasetAdapter):
    NOTES = ""

    def load(self):
        raise NotImplementedError(f"{self.name}: adapter non ancora implementato.\n{self.NOTES}")

    def subjects(self):
        raise NotImplementedError(self.name)

    def stimuli(self):
        raise NotImplementedError(self.name)

    def channel_names(self):
        raise NotImplementedError(self.name)

    def labels(self, task):
        raise NotImplementedError(self.name)

    def trials(self, *a, **k):
        raise NotImplementedError(self.name)


class MusinGAdapter(_StubAdapter):
    name = "musin_g"
    NOTES = """TODO MUSIN-G (OpenNeuro ds003774): 20 soggetti, EGI 128 canali -> mappare a 32 per nome 10-20;
fs eterogenea 250/1000 Hz (ricampionare a 200); 12 brani da ~1.5-2 min, ascolto unico, occhi chiusi;
rating 1-5 con 1 = massimo gradimento (INVERTIRE); presenza dei WAV non verificata."""


class NmedTAdapter(_StubAdapter):
    name = "nmed_t"
    NOTES = """TODO NMED-T (Stanford): 20 soggetti, 128 canali a 1 kHz; versione cleaned 125 Hz/125 canali con
low-pass 50 Hz; 10 brani da ~4.5 min; rating 1-9; audio non incluso."""


class EremusAdapter(_StubAdapter):
    name = "eremus"
    NOTES = """TODO EREMUS: 34 soggetti (26 con label), Emotiv EPOC Flex 32 canali, 128 Hz; 16 trial da ~90 s
(8 brani dalla playlist personale + 8 scelti da altri); 4 classi valenza x power; label di test nascoste
-> LOSO sui 26."""


class SeedAdapter(_StubAdapter):
    name = "seed"
    NOTES = """TODO SEED: 15 soggetti x 3 sessioni, 62 canali, 200 Hz nella versione distribuita; 3 classi =
categoria della clip (non self-report); subset per nome ai 32 canali; è nel pretraining di LaBraM ed
EEGPT (contaminazione: non usarli su SEED)."""


ADAPTERS = {"faced": FacedAdapter, "musin_g": MusinGAdapter, "nmed_t": NmedTAdapter,
            "eremus": EremusAdapter, "seed": SeedAdapter}
