# %% [markdown]
# # Piano B — Notebook F: robustezza alla ricetta (CBraMod a due fasi) e pretraining su clip non viste
#
# Sessione 02, Parti A e B. In ordine:
# 1. Optuna `cbramod_twophase` (30 trial, validation sub080-099): testa (appiattita ufficiale / pooling per
#    canale / pooling globale), standardizzazione per soggetto degli embedding, epoche e lr delle due fasi.
# 2. Controllo di sanità: CBraMod a due fasi su split FM, 1 seed, BA di validation > 0.20. Se fallisce, i job
#    della Parte A non partono: fermarsi e diagnosticare.
# 3. Parte A: pre-addestrato e da zero, split FM e leave-clip-out (3 seed), 2 modelli permutati per fold.
# 4. Parte B: da zero con warm-up e pre-addestrato con warm-up sul leave-clip-out (3 seed).
#
# L'ablazione A.4 (coda `F_bis`) è solo preparata: si lancia solo se il CBraMod a due fasi supera il caso sulle
# clip non viste (tools/aggregate.py stampa la decisione).
#
# Impostazioni: Accelerator **GPU T4 x2**, Internet off. Input: `pianob-faced-data`, `pianob-code`
# (+ l'output della sessione precedente di questo notebook, per riprendere).

# %% INCLUDE _setup_cell.py

# %%
ACCOUNT = "nbF_ricetta_pretraining"
os.environ["PIANOB_REQUIRE_HPO"] = "0" if SYNTHETIC else "1"   # lo studio Optuna della coda è escluso dal controllo
nb_helpers.weights_check()
nb_helpers.run_queue("F", gpus="auto", synthetic=SYNTHETIC)
nb_helpers.queue_status("F")

# %%
nb_helpers.finish(ACCOUNT)
