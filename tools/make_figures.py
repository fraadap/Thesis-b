"""Figure della tesi dai risultati di tutte le sessioni (zip `results_*.zip` o cartelle `pianob_state`).

    python tools/make_figures.py results_*.zip --out figures/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    from aggregate import load_inputs
    from pianob.figures import make_all
    df, jobs, _ = load_inputs(a.inputs)
    made = make_all(df, jobs, a.inputs, Path(a.out))
    print(f"-> {len(made)} figure in {a.out} (indice: {a.out}/INDEX.md)")


if __name__ == "__main__":
    main()
