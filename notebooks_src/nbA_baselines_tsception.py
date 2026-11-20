# %% [markdown]
# # Piano B — Notebook A (Account 1): baseline e TSception
#
# Coda, in ordine di priorità: baseline su feature (CPU, con 100 permutazioni) → Optuna TSception (60 trial)
# ed EEGNet (40) → TSception/EEGNet con EA su split dei foundation model e leave-clip-out (5 seed), incluso
# BN-α offline/online → ablazioni senza EA → ablazione EMG con TSception → permutazioni per il Δstim →
# sonda della clip → valenza binaria.
#
# Impostazioni: Accelerator **GPU T4 x2**, Internet off. Avvio: **Save Version → Save & Run All (Commit)**.
# Per riprendere: aggiungi come input l'output della versione precedente di questo notebook e rilancia.

# %% INCLUDE _setup_cell.py

# %%
ACCOUNT = "account1_nbA"
nb_helpers.run_queue("nbA_baselines_tsception", gpus="auto", synthetic=SYNTHETIC)
nb_helpers.queue_status("nbA_baselines_tsception")

# %% [markdown]
# ## Riepilogo da incollare in chat
# Copia la tabella e tutto il blocco tra `PIANOB_JSON_BEGIN` e `PIANOB_JSON_END`.
# I file completi sono in `results_account1_nbA.zip` (scheda Output).

# %%
nb_helpers.finish(ACCOUNT)
