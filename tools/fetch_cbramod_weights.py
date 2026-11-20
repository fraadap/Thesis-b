"""Scarica i pesi pre-addestrati di CBraMod, ne annota lo sha256 e verifica che si carichino (strict).

    python tools/fetch_cbramod_weights.py --out dist/weights
    python tools/fetch_cbramod_weights.py --file /percorso/pretrained_weights.pth --out dist/weights

Fonte ufficiale (README di pretrained_weights/ nel repo wjq-learning/CBraMod):
  https://huggingface.co/weighting666/CBraMod  -> file pretrained_weights.pth  [nome file: DA VERIFICARE]
Il re-hosting `braindecode/cbramod-pretrained` usa probabilmente nomi di parametri diversi: il loader lo
rifiuta con un messaggio chiaro. Usa la fonte ufficiale.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

URLS = ["https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist/weights")
    ap.add_argument("--file", default=None, help="pesi già scaricati a mano")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "cbramod_pretrained.pth"
    if a.file:
        shutil.copy2(a.file, dst)
        source = str(a.file)
    else:
        source = None
        for url in URLS:
            try:
                print(f"scarico {url} ...")
                urllib.request.urlretrieve(url, dst)
                source = url
                break
            except Exception as e:
                print(f"  fallito: {e}")
        if source is None:
            sys.exit("Download fallito: scarica a mano pretrained_weights.pth dalla pagina Hugging Face e "
                     "rilancia con --file.")
    from pianob.models.cbramod import CBraModClassifier, sha256
    m = CBraModClassifier(32, 2000, 9, pretrained_path=dst)
    info = {"sha256": sha256(dst), "source": source, "size_mb": round(dst.stat().st_size / 2**20, 2),
            "n_tensors": m.pretrained_info["n_tensors"]}
    (out / "cbramod_pretrained.sha256.json").write_text(json.dumps(info, indent=1))
    print(f"OK: pesi caricati in modo strict nel backbone. {info}")


if __name__ == "__main__":
    main()
