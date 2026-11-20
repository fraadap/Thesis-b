# %% [markdown]
# # Piano B — Notebook 0: smoke test (qualsiasi account, va bene anche solo CPU)
#
# Verifica del pacchetto dati (shape, mappa delle label, canali, ampiezze), dei pesi CBraMod, degli
# split con le asserzioni anti-leakage, poi 1 epoca su 3 soggetti per ogni modello. Durata < 15 min.
#
# Input richiesti: Dataset `pianob-faced-data` e `pianob-code`. Internet non necessario.

# %% INCLUDE _setup_cell.py

# %%
nb_helpers.data_check(synthetic=SYNTHETIC)
if not SYNTHETIC:
    nb_helpers.weights_check()

# %%
# 1 epoca su 3 soggetti per ogni modello (CPU o GPU). Il benchmark completo è nei notebook A/B/C.
nb_helpers.run_queue("nb0_smoke", gpus="auto", synthetic=SYNTHETIC, skip_bench=True)
nb_helpers.queue_status("nb0_smoke")

# %%
nb_helpers.finish("smoke")
