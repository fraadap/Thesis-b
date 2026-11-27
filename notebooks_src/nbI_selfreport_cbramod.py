# %% [markdown]
# # Piano B — Notebook I: label da self-report con CBraMod (e REVE se il Gate R è passato) (Sessione 03)
#
# 1. Passo 0: stessa tabella dei voti del notebook H (stesso hash: se differisce il notebook si ferma).
# 2. Coda `I`: CBraMod con la ricetta ufficiale (la configurazione finale non è ancora decisa), valenza e
#    arousal da self-report su split FM e leave-clip-out (3 seed, 1 permutato entro clip per fold), poi i 10 fold,
#    poi (opz.) CBraMod da zero e REVE. I job REVE partono solo se il Gate R risulta passato negli input
#    (collega l'output del notebook G) e se i pesi si scaricano (Internet ON, Secret HF_TOKEN); altrimenti si saltano.
# 3. Analisi finale: tabelle 2 x 2 in `results/selfreport/`.
#
# Impostazioni: Accelerator **GPU T4 x2**. Input: `pianob-faced-data` (con i file comportamentali), `pianob-code`
# (+ l'output della sessione precedente di questo notebook per riprendere; + l'output del notebook G per REVE).

# %%
import os
os.environ["PIANOB_ACCOUNT"] = "nbI_selfreport_cbramod"

# %% INCLUDE _setup_cell.py

# %%
from pianob.selfreport import analysis as SA, table as ST
ACCOUNT = os.environ["PIANOB_ACCOUNT"]
META = ST.run_step0(synthetic=SYNTHETIC)
nb_helpers.weights_check()

# %%
# REVE (blocco opzionale): pesi solo se il Gate R della Sessione 02 risulta passato
from pianob.run import _gate_r_passed
if not SYNTHETIC and _gate_r_passed():
    from pianob.models.reve import weights as reve_weights
    HF_TOKEN = None
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print(f"Secret HF_TOKEN non disponibile ({e!r}): provo senza token")
    try:
        print({k: v for k, v in reve_weights.download(HF_TOKEN).items() if k != "config"})
    except Exception as e:
        print(f"Pesi REVE non scaricati ({e!r}): i job REVE falliranno, gli altri proseguono")
else:
    print("Gate R non passato o non disponibile: i job REVE della coda I vengono saltati")

# %%
nb_helpers.run_queue("I", gpus="auto", synthetic=SYNTHETIC)
nb_helpers.queue_status("I")

# %%
from pianob import report
OUT = kaggle_env.state_dir() / "results" / "selfreport"
RES = SA.run_all([kaggle_env.state_dir()], OUT, rows=report.load_rows(), h2=False,
                 n_boot=200 if SYNTHETIC else 2000, n_perm=200 if SYNTHETIC else 5000)
print("\n".join(RES.get("markdown", [])))

# %%
nb_helpers.finish(ACCOUNT)
