# %% [markdown]
# # Piano B — Notebook G: REVE-Base (riproduzione su FACED e tutti i protocolli di CBraMod)
#
# Sessione 02, Parte C. Prima il Gate R (R1: solo linear probing; R2: LP+FT, 5 seed, BA pooled 0.548-0.581),
# poi i blocchi nello stesso ordine di priorità del prompt: leave-clip-out, permutati, parità di dati, bande,
# canali, sonda della clip, valenza binaria, R3 e opzionali. Coda lunga: 2-3 sessioni con la ripresa.
#
# **Impostazioni:** Accelerator **GPU T4 x2**, **Internet ON** (serve per scaricare i pesi), Kaggle Secret
# `HF_TOKEN` (token Hugging Face dell'account che ha accettato le condizioni di `brain-bzh/reve-base`).
# Input: `pianob-faced-data`, `pianob-code` (+ l'output della sessione precedente di questo notebook).
#
# I pesi si scaricano in /tmp (fuori dagli output della sessione): non vanno ridistribuiti.

# %% INCLUDE _setup_cell.py

# %%
# Pesi REVE: token dal Kaggle Secret, revisione risolta e annotata (fissabile con PIANOB_REVE_REVISION)
from pianob.models.reve import weights as reve_weights
if SYNTHETIC:
    print("sintetico: niente download, pesi casuali")
else:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    REVE_INFO = reve_weights.download(HF_TOKEN)
    print({k: v for k, v in REVE_INFO.items() if k != "config"})

# %%
# Controlli obbligatori (Appendice R §R10): conteggi, forme, parametri allenabili, strict, niente NaN
import torch
from pianob.data.adapters import FacedAdapter
from pianob.models.reve.checks import preflight
PREFLIGHT = preflight(FacedAdapter(synthetic=SYNTHETIC).load(), device="cuda" if torch.cuda.is_available() else "cpu",
                      require_weights=not SYNTHETIC)

# %%
ACCOUNT = "nbG_reve"
nb_helpers.run_queue("G", gpus="auto", synthetic=SYNTHETIC)
nb_helpers.queue_status("G")

# %%
nb_helpers.finish(ACCOUNT)
