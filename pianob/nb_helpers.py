"""Funzioni usate dai notebook (così i notebook restano corti e non vanno toccati a mano)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import kaggle_env


def banner(title: str) -> None:
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80, flush=True)


def print_env() -> dict:
    info = kaggle_env.describe_env()
    banner("Ambiente")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"  versione codice: {kaggle_env.code_version()}")
    if info["kaggle"] and not info["gpus"]:
        print("  " + "!" * 70)
        print("  ATTENZIONE: nessuna GPU. Settings -> Accelerator -> GPU T4 x2.")
        print("  Solo lo smoke test gira su CPU: le altre code si fermeranno subito.")
        print("  " + "!" * 70)
    elif info["gpus"] and "P100" in info["gpus"][0]["name"]:
        print("  Nota: P100 (più lenta della T4 in fp16, niente tensor core). Funziona comunque.")
    return info


def run_queue(queue: str, gpus: str = "auto", synthetic: bool = False, skip_bench: bool = False) -> int:
    """Esegue `python -m pianob.run` in un sottoprocesso mostrando l'output riga per riga."""
    code_root = Path(__file__).resolve().parents[1]
    qpath = code_root / "queues" / f"{queue}.json"
    assert qpath.exists(), f"coda non trovata: {qpath}"
    check_queue_hpo(qpath)
    cmd = [sys.executable, "-u", "-m", "pianob.run", "--queue", str(qpath), "--gpus", gpus]
    if synthetic:
        cmd.append("--synthetic")
    if skip_bench:
        cmd.append("--skip-bench")
    banner(f"Esecuzione coda {queue}")
    p = subprocess.Popen(cmd, cwd=str(code_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=dict(os.environ))
    for line in p.stdout:
        if "  ep " in line and os.environ.get("PIANOB_QUIET_EPOCHS") == "1":
            continue
        print(line, end="", flush=True)
    return p.wait()


def queue_status(queue: str) -> None:
    from .experiments.queue import DONE_STATUSES, done_status, load_queue
    code_root = Path(__file__).resolve().parents[1]
    jobs = load_queue(code_root / "queues" / f"{queue}.json")
    done = done_status()
    n_ok = sum(done.get(j.job_id, {}).get("status") in DONE_STATUSES for j in jobs)
    n_fail = sum(done.get(j.job_id, {}).get("status") == "failed" for j in jobs)
    banner(f"Stato coda {queue}: {n_ok}/{len(jobs)} completati, {n_fail} falliti")
    for j in jobs:
        st = done.get(j.job_id, {}).get("status")
        if st == "failed":
            print(f"  FALLITO {j.job_id}: {done[j.job_id].get('error', '')[:300]}")
    rest = [j.job_id for j in jobs if done.get(j.job_id, {}).get("status") not in DONE_STATUSES]
    if rest:
        print(f"  Da fare ({len(rest)}): " + ", ".join(rest[:15]) + (" ..." if len(rest) > 15 else ""))
        print("  -> Per continuare: Save Version di nuovo con l'output di questa versione come input (vedi README).")


def data_check(synthetic: bool = False) -> None:
    """Verifica del pacchetto dati: shape, mappa delle label, canali, ampiezze, split e asserzioni."""
    import numpy as np
    from .data import faced_meta as FM
    from .data.adapters import FacedAdapter
    from .data.splits import PROTOCOLS, make_split, n_folds
    banner("Verifica del pacchetto dati")
    ad = FacedAdapter(synthetic=synthetic).load()
    m = ad.meta
    print(f"  sintetico: {synthetic}; soggetti presenti: {len(ad.subjects())}; mancanti: {m.get('missing', [])}")
    print(f"  fs: {ad.fs} Hz; canali ({len(ad.channel_names())}): {ad.channel_names()}")
    print(f"  ricampionamento: {m.get('resample_method')}; scala nel loader: x{ad.default_scale} (CBraMod: /100)")
    for row in ad.clip_table():
        print(f"    clip {row['clip']:2d}: {row['category']:<12} valenza {row['valence']:+d} (n. {row['idx_in_category']})")
    if not synthetic:
        x = ad.trials([0, 61, 122], [0, 27], scale=1.0)
        print(f"  ampiezza (µV) sub000/061/122: std {[round(float(v), 2) for v in x.std(axis=(1, 2, 3))]}")
        assert np.isfinite(x).all(), "valori non finiti nei dati"
        if "per_subject_std" in m:
            s = np.array([v for v in m["per_subject_std"] if v is not None])
            print(f"  std per soggetto: mediana {np.median(s):.2f}, min {s.min():.2f}, max {s.max():.2f}")
    for p in PROTOCOLS:
        for f in range(n_folds(p)):
            make_split(p, f, ad.task_clips("emo9"), ad.subjects())
            make_split(p, f, ad.task_clips("val2"), ad.subjects())
    print("  split: tutti i protocolli e fold superano le asserzioni anti-leakage")
    print("  OK")


def weights_check() -> None:
    banner("Pesi CBraMod")
    wp = kaggle_env.find_file("cbramod_pretrained.pth") or kaggle_env.find_file("pretrained_weights.pth")
    if wp is None:
        print("  pesi NON trovati: le config CBraMod pre-addestrate falliranno. Controlla il Dataset dati.")
        return
    from .models.cbramod import CBraModClassifier
    m = CBraModClassifier(32, 2000, 9, pretrained_path=wp)
    print(f"  caricati (strict) da {wp}: {m.pretrained_info}")
    exp = kaggle_env.find_file("cbramod_pretrained.sha256.json")
    if exp:
        want = json.loads(exp.read_text()).get("sha256")
        print(f"  checksum atteso {want[:16]}..., trovato {m.pretrained_info['sha256'][:16]}... -> "
              f"{'OK' if want == m.pretrained_info['sha256'] else 'DIVERSO!'}")


def check_queue_hpo(qpath) -> None:
    """Con PIANOB_REQUIRE_HPO=1 controlla PRIMA di partire che ogni config della coda trovi i suoi iperparametri
    Optuna: altrimenti si ferma subito, invece di far fallire i job uno a uno per tutta la sessione."""
    if os.environ.get("PIANOB_REQUIRE_HPO") != "1":
        return
    from .experiments.queue import load_queue
    from .experiments.registry import find_hpo_best, get_config
    jobs = load_queue(qpath)
    produced = {j.config for j in jobs if j.kind == "hpo"}   # studi Optuna eseguiti dalla coda stessa
    studies = {get_config(j.config).hpo_study for j in jobs if j.kind == "train"} - {None, ""} - produced
    missing = sorted(s for s in studies if find_hpo_best(s) is None)
    banner("Iperparametri Optuna richiesti dalla coda")
    for st in sorted(studies):
        print(f"  {st}: {'MANCANTE' if st in missing else find_hpo_best(st)}")
    if missing:
        raise RuntimeError(f"iperparametri Optuna mancanti: {missing}. Lo zip del codice è aggiornato? "
                           "Ricostruiscilo con tools/make_zips.py --code-only")


def require_hpo(study: str) -> bool:
    from .experiments.registry import find_hpo_best
    p = find_hpo_best(study)
    banner(f"Iperparametri Optuna '{study}'")
    if p is None:
        print("  NON trovati. Collega come input l'output del notebook che ha fatto lo studio Optuna, "
              "oppure incollami il JSON e aggiorno pianob/hpo/best/.")
        return False
    print(f"  trovati in {p}: {json.loads(p.read_text())}")
    return True


def finish(account: str) -> None:
    from . import report
    report.print_summary(account)
    report.make_results_zip(account)
    print(f"\nTempo totale della sessione: {kaggle_env.TimeGuard().elapsed_min():.0f} min")
