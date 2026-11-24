# %% [markdown]
# # Piano B — Notebook E (Account 3): EEGNet con la metodologia originale
#
# EEGNet come nel paper originale (Lawhern et al. 2018: dropout 0.25, max-norm, Adam senza weight decay, fino a
# 200 epoche con il modello migliore in validation) nel protocollo del benchmark CBraMod (finestre da 10 s a
# 200 Hz, scala /100, nessun allineamento), più EEGNet addestrato con il framework di fine-tuning di CBraMod.
#
# Domande: da dove viene il divario tra il nostro EEGNet (0.229) e quello riportato da CBraMod (0.409)?
# Anche un EEGNet ben addestrato è memorizzazione della clip (leave-clip-out)?
#
# Impostazioni: Accelerator **GPU T4 x2**, Internet off. Input: `pianob-faced-data`, `pianob-code`.

# %% INCLUDE _setup_cell.py

# %%
ACCOUNT = "account3_nbE"
nb_helpers.run_queue("nbE_eegnet", gpus="auto", synthetic=SYNTHETIC)
nb_helpers.queue_status("nbE_eegnet")

# %%
nb_helpers.finish(ACCOUNT)
