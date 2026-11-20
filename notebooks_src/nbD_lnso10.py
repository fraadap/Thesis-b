# %% [markdown]
# # Piano B — Notebook D (a rotazione): 10 fold sui soggetti
#
# Config finali su `lnso10` e `lnso10_lco` (3 seed) e gate di potenza. La lista delle config finali si
# cambia rigenerando la coda (`queues/nbD_lnso10.json`), non qui. Servono gli iperparametri Optuna
# (output dei notebook A/B collegati come input, oppure file in `pianob/hpo/best/`).

# %% INCLUDE _setup_cell.py

# %%
ACCOUNT = "rotation_nbD"
os.environ["PIANOB_REQUIRE_HPO"] = "0" if SYNTHETIC else "1"
nb_helpers.run_queue("nbD_lnso10", gpus="auto", synthetic=SYNTHETIC)
nb_helpers.queue_status("nbD_lnso10")

# %%
nb_helpers.finish(ACCOUNT)
