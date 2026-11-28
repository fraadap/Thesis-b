# %% [markdown]
# # Piano B — Notebook K: confronti aggiuntivi per la tesi
#
# Coda `K` (40 job), in ordine di priorità:
# 1. **Riconoscimento della clip**: quale delle 28 clip si sta guardando, su soggetti mai visti (split FM).
#    DE + z-score, EEGNet + EA, TSception + EA (4 s e 10 s), CBraMod. Serve alla figura F10.
# 2. **Protocollo entro soggetto** (`split_fm_within`): i soggetti di test entrano nel training con 2 dei 3
#    segmenti da 10 s di ogni clip e si testano sul terzo, come in molta letteratura. Figura F9.
# 3. **Lunghezza della finestra** (2 e 4 s contro 10 s) nel leave-clip-out: priorità bassa. Figura F13.
#
# Impostazioni: Accelerator **GPU T4 x2**, Internet off. Input: `pianob-faced-data`, `pianob-code`, i pesi di
# CBraMod (come per i notebook B, C, F) e, per riprendere, l'output della sessione precedente di questo notebook.

# %%
import os
os.environ["PIANOB_ACCOUNT"] = "nbK_confronti"   # la ripresa usa solo gli output di questo notebook

# %% INCLUDE _setup_cell.py

# %%
ACCOUNT = os.environ["PIANOB_ACCOUNT"]
nb_helpers.weights_check()
nb_helpers.run_queue("K", gpus="auto", synthetic=SYNTHETIC, cpu_workers=2)
nb_helpers.queue_status("K")

# %%
nb_helpers.finish(ACCOUNT)
