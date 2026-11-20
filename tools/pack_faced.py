"""Impacchetta FACED processed (Synapse syn50614194, cartella Processed_data/) per Kaggle. Gira su CPU.

    python tools/pack_faced.py --src /percorso/Processed_data --out dist/faced

Produce in --out:
  faced_200hz_f16.npy   (123, 28, 32, 6000) float16, ~1.3 GB  (microvolt; la scala /100 è nel loader)
  meta.json             canali, label, mappa clip, soggetti presenti/mancanti, statistiche d'ampiezza
  checksums.json        sha256 dei file prodotti e dei .pkl sorgente

Ricampionamento 250 -> 200 Hz:
  --resample fft  (default) = scipy.signal.resample sull'intero segmento da 30 s, IDENTICO allo script
                    ufficiale CBraMod (preprocessing/preprocessing_faced.py) -> per il Gate 1;
  --resample poly = scipy.signal.resample_poly(up=4, down=5) (alternativa senza effetti di periodicità).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pianob.data import faced_meta as FM  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pkl(path: Path) -> np.ndarray:
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict):  # per sicurezza: alcune versioni potrebbero avvolgere l'array
        keys = [k for k, v in obj.items() if isinstance(v, np.ndarray)]
        assert len(keys) == 1, f"{path.name}: dict con chiavi {list(obj)}; serve un solo array"
        obj = obj[keys[0]]
    arr = np.asarray(obj)
    assert arr.shape == (FM.N_CLIPS, len(FM.CHANNELS), FM.N_SAMPLES_PROCESSED), \
        f"{path.name}: shape {arr.shape}, attesa (28, 32, 7500)"
    assert np.isfinite(arr).all(), f"{path.name}: valori non finiti"
    return arr


def resample(arr: np.ndarray, method: str) -> np.ndarray:
    if method == "fft":
        out = signal.resample(arr, FM.N_SAMPLES_TARGET, axis=-1)
    else:
        out = signal.resample_poly(arr, up=4, down=5, axis=-1)
    assert out.shape[-1] == FM.N_SAMPLES_TARGET
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="cartella Processed_data con sub000.pkl ... sub122.pkl")
    ap.add_argument("--out", default="dist/faced")
    ap.add_argument("--resample", choices=["fft", "poly"], default="fft")
    ap.add_argument("--keep-250hz", action="store_true", help="salva anche faced_250hz_f16.npy (~1.6 GB)")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {int(re.match(r"sub(\d+)\.pkl$", p.name).group(1)): p
             for p in sorted(src.glob("sub*.pkl")) if re.match(r"sub(\d+)\.pkl$", p.name)}
    assert files, f"nessun subXXX.pkl in {src}"
    n_slots = max(FM.N_SUBJECTS, max(files) + 1)
    missing = [s for s in range(n_slots) if s not in files]
    if missing:
        print(f"ATTENZIONE: soggetti mancanti {missing} (slot riempiti con NaN e registrati in meta.json)")

    data_path = out / "faced_200hz_f16.npy"
    X = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.float16,
                                  shape=(n_slots, FM.N_CLIPS, len(FM.CHANNELS), FM.N_SAMPLES_TARGET))
    X250 = None
    if a.keep_250hz:
        X250 = np.lib.format.open_memmap(out / "faced_250hz_f16.npy", mode="w+", dtype=np.float16,
                                         shape=(n_slots, FM.N_CLIPS, len(FM.CHANNELS), FM.N_SAMPLES_PROCESSED))
    stats, src_hash, dtypes = {}, {}, set()
    t0 = time.time()
    for s in range(n_slots):
        if s not in files:
            X[s] = np.nan
            if X250 is not None:
                X250[s] = np.nan
            continue
        arr = load_pkl(files[s])
        dtypes.add(str(arr.dtype))
        r = resample(arr.astype(np.float64), a.resample)
        mx = float(np.abs(r).max())
        assert mx < 6e4, f"sub{s:03d}: ampiezza {mx:.0f} fuori dal range float16"
        X[s] = r.astype(np.float16)
        if X250 is not None:
            X250[s] = arr.astype(np.float16)
        ch_std = arr.std(axis=(0, 2))
        stats[s] = {"std": float(arr.std()), "max_abs": float(np.abs(arr).max()),
                    "flat_channels": [FM.CHANNELS[i] for i in np.where(ch_std < 1e-6)[0]],
                    "f16_rel_err": float(np.abs(X[s].astype(np.float64) - r).max() / (mx + 1e-12))}
        src_hash[f"sub{s:03d}.pkl"] = sha256(files[s])
        if s % 10 == 0:
            print(f"  sub{s:03d}: std {stats[s]['std']:.2f} µV, max {stats[s]['max_abs']:.1f} ({time.time() - t0:.0f}s)")
    X.flush()
    del X
    stds = [stats[s]["std"] if s in stats else None for s in range(n_slots)]
    med = float(np.median([v for v in stds if v is not None]))
    unit_note = "microvolt (atteso)" if 1 < med < 500 else f"ATTENZIONE: std mediana {med:.3g}, unità da verificare"
    print(f"std mediana {med:.2f} -> {unit_note}")
    meta = {
        "dataset": "FACED", "version": 1, "created": time.strftime("%Y-%m-%d %H:%M"),
        "source": "Synapse syn50614194, Processed_data (ultimi 30 s, 0.05-47 Hz, ICA oculare, CAR)",
        "fs": FM.FS_TARGET, "fs_source": FM.FS_PROCESSED, "resample_method": a.resample,
        "shape": [n_slots, FM.N_CLIPS, len(FM.CHANNELS), FM.N_SAMPLES_TARGET], "dtype": "float16",
        "units": unit_note, "loader_scale_cbramod": 0.01,
        "channels": FM.CHANNELS, "channels_note": "ordine da TorchEEG; A1/A2 mastoidi, ordine [DA VERIFICARE]",
        "source_dtypes": sorted(dtypes),
        "clip_labels": FM.CLIP_LABELS, "clip_valence": FM.CLIP_VALENCE, "categories": FM.CATEGORIES,
        "clip_index_in_category": FM.clip_index_in_category(),
        "subjects_present": sorted(files), "missing": missing, "n_subjects_slots": n_slots,
        "per_subject_std": stds, "per_subject_stats": {f"sub{s:03d}": v for s, v in stats.items()},
        "ratings": None, "ratings_note": "i .pkl processed non contengono i self-report: da aggiungere dai raw",
    }
    meta_path = out / "meta.json"
    meta["checksum_data"] = sha256(data_path)
    meta_path.write_text(json.dumps(meta, indent=1))
    checks = {"faced_200hz_f16.npy": meta["checksum_data"], "meta.json": sha256(meta_path), "sources": src_hash}
    if a.keep_250hz:
        checks["faced_250hz_f16.npy"] = sha256(out / "faced_250hz_f16.npy")
    (out / "checksums.json").write_text(json.dumps(checks, indent=1))
    print(f"fatto in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
