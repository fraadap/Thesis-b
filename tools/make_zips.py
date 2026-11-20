"""Costruisce gli zip da caricare su Kaggle (in dist/).

    python tools/make_zips.py --data dist/faced --weights dist/weights   # dati (una volta) + codice
    python tools/make_zips.py --code-only                                # solo codice (a ogni modifica)
    python tools/make_zips.py --data dist/faced --weights dist/weights --single   # tutto in uno (sconsigliato)

pianob_faced_data.zip -> Dataset privato `pianob-faced-data`:
    faced/faced_200hz_f16.npy, faced/meta.json, faced/checksums.json,
    weights/cbramod_pretrained.pth, weights/cbramod_pretrained.sha256.json
pianob_code.zip -> Dataset `pianob-code` (si aggiorna con New Version):
    pianob/, notebooks/*.ipynb, queues/*.json, tools/, README_KAGGLE.md, STATO_PIANO_B.md, VERSION
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def version() -> str:
    try:
        h = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                    stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "pianob", "queues"],
                                        stderr=subprocess.DEVNULL).decode().strip()
        h += "+modifiche" if dirty else ""
    except Exception:
        h = "nogit"
    return f"{time.strftime('%Y-%m-%dT%H:%M')} {h}"


def add_tree(z: zipfile.ZipFile, base: Path, arc_prefix: str, patterns=("**/*",)):
    for pat in patterns:
        for f in sorted(base.glob(pat)):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix not in (".pyc",):
                z.write(f, f"{arc_prefix}/{f.relative_to(base)}")


def code_zip(z: zipfile.ZipFile, prefix: str = ""):
    p = (prefix + "/") if prefix else ""
    subprocess.check_call([sys.executable, str(ROOT / "tools" / "build_notebooks.py")])
    subprocess.check_call([sys.executable, "-m", "pianob.experiments.queue", "--out", str(ROOT / "queues")], cwd=ROOT)
    add_tree(z, ROOT / "pianob", f"{p}pianob")
    add_tree(z, ROOT / "notebooks", f"{p}notebooks", ("*.ipynb",))
    add_tree(z, ROOT / "queues", f"{p}queues", ("*.json",))
    add_tree(z, ROOT / "tools", f"{p}tools", ("*.py",))
    for f in ("README_KAGGLE.md", "STATO_PIANO_B.md"):
        if (ROOT / f).exists():
            z.write(ROOT / f, f"{p}{f}")
    z.writestr(f"{p}VERSION", version() + "\n")


def data_zip(z: zipfile.ZipFile, data: Path, weights: Path | None, prefix: str = ""):
    p = (prefix + "/") if prefix else ""
    for name in ("faced_200hz_f16.npy", "meta.json", "checksums.json"):
        f = data / name
        assert f.exists(), f"manca {f}: esegui prima tools/pack_faced.py"
        z.write(f, f"{p}faced/{name}", compress_type=zipfile.ZIP_STORED if f.suffix == ".npy" else zipfile.ZIP_DEFLATED)
    if weights is not None:
        w = weights / "cbramod_pretrained.pth"
        assert w.exists(), f"manca {w}: esegui prima tools/fetch_cbramod_weights.py"
        z.write(w, f"{p}weights/cbramod_pretrained.pth")
        s = weights / "cbramod_pretrained.sha256.json"
        if s.exists():
            z.write(s, f"{p}weights/{s.name}")
    else:
        print("ATTENZIONE: zip dei dati senza pesi CBraMod")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="cartella prodotta da pack_faced.py")
    ap.add_argument("--weights", default=None, help="cartella prodotta da fetch_cbramod_weights.py")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--code-only", action="store_true")
    ap.add_argument("--single", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = Path(a.weights) if a.weights else None
    if a.single:
        assert a.data, "--single richiede --data"
        zp = out / "pianob_all.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            code_zip(z, "code")
            data_zip(z, Path(a.data), weights, "data")
        print(f"{zp}: {zp.stat().st_size / 2**20:.1f} MB")
        return
    zp = out / "pianob_code.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        code_zip(z)
    print(f"{zp}: {zp.stat().st_size / 2**20:.2f} MB")
    if not a.code_only:
        assert a.data, "serve --data (oppure --code-only)"
        zp = out / "pianob_faced_data.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            data_zip(z, Path(a.data), weights)
        print(f"{zp}: {zp.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
