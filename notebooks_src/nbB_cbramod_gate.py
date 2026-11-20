# %% [markdown]
# # Piano B — Notebook B (Account 2): CBraMod, Gate 1
#
# Coda: CBraMod pre-addestrato con la ricetta ufficiale di fine-tuning su split dei foundation model
# (3 seed, **Gate 1**: BA pooled entro ~2 punti da 0.5509) → stesso modello da init casuale (3 seed) →
# Optuna per CBraMod con LoRA (30 trial).
#
# Impostazioni: Accelerator **GPU T4 x2**, Internet off. Avvio: **Save Version → Save & Run All (Commit)**.

# %% INCLUDE _setup_cell.py

# %%
ACCOUNT = "account2_nbB"
if not SYNTHETIC:
    nb_helpers.weights_check()
nb_helpers.run_queue("nbB_cbramod_gate", gpus="auto", synthetic=SYNTHETIC)
nb_helpers.queue_status("nbB_cbramod_gate")

# %%
from pianob import report
nb_helpers.banner(f"GATE 1: {report.gate1(report.load_jobs())}")
nb_helpers.finish(ACCOUNT)
