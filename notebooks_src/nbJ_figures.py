# %% [markdown]
# # Piano B — Notebook J: figure della tesi
#
# Genera tutte le figure (in inglese, PDF e PNG) dai risultati di **tutte** le sessioni, più `INDEX.md` con
# una descrizione e i numeri principali di ogni figura. Nessun training: basta la CPU (circa 10-20 minuti).
# Le figure che dipendono da run non ancora fatte (notebook K, voti dei notebook H/I) si saltano da sole e
# compaiono quando i loro output vengono collegati.
#
# Impostazioni: Accelerator **None (CPU)**, Internet off. Input: `pianob-code` e **l'ultimo output di ogni
# notebook** (A, B, C, D, E, F, G, e quando ci sono H, I, K). Gli output sono cumulativi: basta l'ultimo di
# ciascun notebook; collegarne di più vecchi non fa danni (i duplicati si scartano). Output:
# `figures/` e `figures_thesis.zip`.

# %%
import os
os.environ["PIANOB_NO_RESTORE"] = "1"      # gli output collegati si leggono, non si copiano nello stato

# %% INCLUDE _setup_cell.py

# %%
from pathlib import Path
sys.path.insert(0, os.path.join(CODE, "tools"))
from aggregate import load_inputs
from pianob import figures

states = sorted({p for r in kaggle_env.input_roots() for p in r.rglob("pianob_state") if p.is_dir()})
zips = sorted({z for r in kaggle_env.input_roots() for z in r.rglob("*results_*.zip")
               if not any(s.parent == z.parent for s in states)})
INPUTS = [str(p) for p in states + zips]
print(f"{len(INPUTS)} input con risultati:")
for p in INPUTS:
    print("  ", p)
assert INPUTS or SYNTHETIC, "Nessun output collegato: aggiungi come input l'ultimo output di ogni notebook (A-G)."

# %%
OUT = kaggle_env.working_dir() / "figures"
if INPUTS:
    df, jobs, _ = load_inputs(INPUTS)
    made = figures.make_all(df, jobs, INPUTS, OUT)
    print((OUT / "INDEX.md").read_text())
    shutil.make_archive(str(kaggle_env.working_dir() / "figures_thesis"), "zip", OUT)
    print(f"\n{len(made)} figure -> figures_thesis.zip")
