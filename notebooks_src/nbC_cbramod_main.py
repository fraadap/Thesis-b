# %% [markdown]
# # Piano B — Notebook C (Account 3): CBraMod, esperimenti principali
#
# **Da lanciare solo dopo che il Gate 1 è passato** e con gli iperparametri Optuna di CBraMod LoRA
# disponibili (output del notebook B collegato come input, oppure file in `pianob/hpo/best/`).
#
# Coda: CBraMod LoRA su split dei foundation model (3 seed) e leave-clip-out (5 seed) → ricetta ufficiale su
# leave-clip-out (5 seed) → LoRA con EA → ablazione EMG (3 seed) → sonda della clip → permutazioni Δstim (3)
# → LoRA con z-score per soggetto.

# %% INCLUDE _setup_cell.py

# %%
ACCOUNT = "account3_nbC"
os.environ["PIANOB_REQUIRE_HPO"] = "0" if SYNTHETIC else "1"   # niente default silenziosi
ok = SYNTHETIC or nb_helpers.require_hpo("cbramod_lora_r2")
if not SYNTHETIC:
    nb_helpers.weights_check()
if ok:
    nb_helpers.run_queue("nbC_cbramod_main", gpus="auto", synthetic=SYNTHETIC)
    nb_helpers.queue_status("nbC_cbramod_main")

# %%
nb_helpers.finish(ACCOUNT)
