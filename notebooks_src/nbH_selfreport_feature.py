# %% [markdown]
# # Piano B — Notebook H: label da self-report, feature e TSception (Sessione 03)
#
# 1. Passo 0: tabella dei voti dai file After_remarks (allineamento delle righe deciso dai dati, controlli
#    bloccanti, hash). Se un controllo fallisce il notebook si ferma con il motivo: riportalo.
# 2. H1: statistiche descrittive dei voti -> `report_selfreport_stage1.md` (scritto subito in /kaggle/working).
# 3. H2: rianalisi delle predizioni a 9 classi già salvate dalle sessioni precedenti (zero training).
# 4. Coda `H`: DE + z-score (worker solo-CPU, con le permutazioni entro clip) e TSception (GPU).
# 5. Analisi finale: tabelle 2 x 2, terza famiglia, `results/selfreport/`.
#
# Impostazioni: Accelerator **GPU T4 x2**, Internet off. Input: `pianob-faced-data` (con i file comportamentali),
# `pianob-code`, gli **output di tutte le sessioni precedenti** (servono alla rianalisi H2; restano in sola
# lettura) e, per riprendere, l'output della sessione precedente di questo notebook.

# %%
import os
os.environ["PIANOB_ACCOUNT"] = "nbH_selfreport_feature"   # la ripresa usa solo gli output di questo notebook

# %% INCLUDE _setup_cell.py

# %%
# Passo 0 + H1: tabella dei voti e descrittive (CPU, pochi minuti)
from pathlib import Path
from pianob.selfreport import analysis as SA, table as ST
ACCOUNT = os.environ["PIANOB_ACCOUNT"]
META = ST.run_step0(synthetic=SYNTHETIC)
VOTES = ST.load_votes()
DESC = SA.descriptives(VOTES, META, n_boot=200 if SYNTHETIC else 1000)
OUT = kaggle_env.state_dir() / "results" / "selfreport"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "descriptives.json").write_text(__import__("json").dumps(DESC, indent=1, default=float))
STAGE1 = SA.stage1_markdown(DESC)
(kaggle_env.working_dir() / "report_selfreport_stage1.md").write_text(STAGE1)
print(STAGE1)

# %%
# H2: rianalisi delle predizioni a 9 classi delle sessioni precedenti (input in sola lettura + stato)
SOURCES = [p for p in kaggle_env.input_roots()] + [kaggle_env.state_dir()]
if not (OUT / "reanalysis_h2.json").exists():
    H2 = SA.reanalyze_emo9(SOURCES, VOTES, n_boot=200 if SYNTHETIC else 2000, n_perm=200 if SYNTHETIC else 5000)
    (OUT / "reanalysis_h2.json").write_text(__import__("json").dumps(H2, indent=1, default=float))
    print("\n".join(SA.session03_markdown([], H2)))

# %%
nb_helpers.run_queue("H", gpus="auto", synthetic=SYNTHETIC, cpu_workers=2)
nb_helpers.queue_status("H")

# %%
# Analisi finale (predizioni di questo notebook): tabelle 2 x 2, terza famiglia
from pianob import report
RES = SA.run_all([kaggle_env.state_dir()], OUT, rows=report.load_rows(), h2=False,
                 n_boot=200 if SYNTHETIC else 2000, n_perm=200 if SYNTHETIC else 5000)
print("\n".join(RES.get("markdown", [])))

# %%
nb_helpers.finish(ACCOUNT)
