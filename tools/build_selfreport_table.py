"""Passo 0 della Sessione 03: tabella dei voti di FACED (`selfreport.parquet` + `meta_selfreport.json`).

    python tools/build_selfreport_table.py --input /percorso/FACED --out analysis/selfreport
    python tools/build_selfreport_table.py --synthetic --out /tmp/sr

Ispeziona i file After_remarks (.mat o .pkl) dei soggetti, sceglie dai dati l'allineamento delle righe di
`score` (ordine di presentazione o canonico), verifica la mappa vid -> categoria e l'ordine delle colonne,
scrive la tabella e l'hash del contenuto. Si ferma (SelfReportError) se un controllo bloccante fallisce.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pianob.selfreport.table import SelfReportError, run_step0  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="*", default=None, help="cartelle in cui cercare i file (default: input Kaggle)")
    ap.add_argument("--out", default=None, help="cartella di uscita (default: stato/selfreport)")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args(argv)
    try:
        meta = run_step0(roots=[Path(p) for p in a.input] if a.input else None,
                         out_dir=Path(a.out) if a.out else None, synthetic=a.synthetic)
    except SelfReportError as e:
        print(f"\nPASSO 0 FERMATO: {e}", file=sys.stderr)
        sys.exit(3)
    print({k: meta[k] for k in ("alignment", "sha256", "n_subjects")})


if __name__ == "__main__":
    main()
