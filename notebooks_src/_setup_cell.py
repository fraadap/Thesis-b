# %%
# Setup: trova il codice (Dataset "pianob-code"), lo copia fuori da /kaggle/working (così non finisce
# negli output) e lo aggiunge a sys.path. Non serve modificare nulla qui.
import glob, os, shutil, sys, time
os.environ.setdefault("PIANOB_T0", str(time.time()))      # inizio della sessione (guardia delle 12 h)
os.environ.setdefault("PIANOB_BUDGET_H", "11.3")           # stop pulito prima del limite Kaggle
SYNTHETIC = os.environ.get("PIANOB_SYNTHETIC") == "1"      # solo per prove locali


def _find_code():
    cands = []
    roots = ["/kaggle/input"] if os.path.exists("/kaggle/input") else []
    roots += [os.getcwd(), os.path.dirname(os.getcwd())]
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "pianob", "__init__.py"), recursive=True):
            base = os.path.dirname(os.path.dirname(p))
            if "pianob_state" not in base:
                cands.append(base)
    assert cands, "Codice non trovato: aggiungi il Dataset 'pianob-code' come input (Add Input)."
    # se ci sono più copie, usa quella con VERSION più recente
    def _ver(b):
        v = os.path.join(b, "VERSION")
        return open(v).read().strip() if os.path.exists(v) else ""
    return sorted(cands, key=_ver)[-1]


CODE_SRC = _find_code()
if os.path.exists("/kaggle/working"):
    CODE = "/tmp/pianob_code"
    shutil.rmtree(CODE, ignore_errors=True)
    shutil.copytree(CODE_SRC, CODE, ignore=shutil.ignore_patterns("notebooks", "*.zip", "local_work"))
else:
    CODE = CODE_SRC
sys.path.insert(0, CODE)
os.chdir(CODE)
print("codice da", CODE_SRC, "| versione", open(os.path.join(CODE, "VERSION")).read().strip()
      if os.path.exists(os.path.join(CODE, "VERSION")) else "(locale)")

from pianob import kaggle_env, nb_helpers  # noqa: E402
nb_helpers.print_env()
kaggle_env.restore_state_from_inputs()
