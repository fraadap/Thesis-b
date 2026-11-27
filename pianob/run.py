"""CLI: esegue una coda di job, con un worker per GPU.

    python -m pianob.run --queue queues/nbA_baselines_tsception.json --gpus auto
    python -m pianob.run --queue queues/nb0_smoke.json --synthetic --gpus none
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from . import kaggle_env


def _n_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _needs_gpu(job) -> bool:
    """Job che su CPU non sono praticabili: modelli neurali e studi Optuna (esclusi gli smoke test)."""
    if job.overrides.get("max_subjects"):
        return False
    if job.kind == "hpo":
        return True
    if job.kind != "train":
        return False
    from .experiments.registry import get_config
    return get_config(job.config).family == "neural"


def worker_loop(queue_path: str, worker: str, synthetic: bool, device: str) -> None:
    from .bench import empirical_minutes, estimate_job
    from .data.adapters import FacedAdapter
    from .experiments.jobs import Context, run_job
    from .experiments.queue import DONE_STATUSES, done_status, is_locked, load_queue, mark, release, try_claim

    jobs = load_queue(queue_path)
    guard = kaggle_env.TimeGuard()
    adapter = FacedAdapter(synthetic=synthetic, synth_subjects=123).load()
    wp = None
    for name in ("cbramod_pretrained.pth", "pretrained_weights.pth"):
        wp = kaggle_env.find_file(name)
        if wp:
            break
    log_path = kaggle_env.state_dir() / "logs" / f"{worker}.log"

    def log(msg):
        line = f"[{worker} {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    ctx = Context(adapter, device=device, worker=worker, guard=guard, log=log, weights_path=wp)
    bench_path = kaggle_env.state_dir() / "bench" / "bench.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else {}
    failed, skipped_time = set(), set()
    log(f"avvio: codice {kaggle_env.code_version()}, {len(jobs)} job in coda, device={device}, "
        f"pesi CBraMod={'sì' if wp else 'no'}, "
        f"tempo residuo {guard.remaining_min():.0f} min")
    while True:
        done = done_status()
        emp = empirical_minutes()
        ok = {k for k, v in done.items() if v["status"] in DONE_STATUSES}
        # job falliti in QUESTA sessione (anche da un altro worker): non si riprovano ora
        failed |= {k for k, v in done.items() if v["status"] == "failed" and v.get("t", 0) >= guard.t0}
        pending = [j for j in jobs if j.job_id not in ok and j.job_id not in failed and j.job_id not in skipped_time]
        if not pending:
            log("coda terminata")
            break
        started = False
        waiting = False
        for j in pending:
            if any(d not in ok for d in j.depends_on):
                if any(d in failed for d in j.depends_on):
                    continue
                waiting = True
                continue
            if is_locked(j):
                waiting = True
                continue
            est = estimate_job(j, bench, emp)
            if not guard.can_start(est):
                skipped_time.add(j.job_id)
                log(f"salto {j.job_id}: stima {est:.0f} min > tempo residuo {guard.remaining_min():.0f} min")
                continue
            if not try_claim(j, worker):
                continue
            started = True
            t0 = time.time()
            log(f"INIZIO {j.job_id} (stima {est:.0f} min)")
            try:
                status = run_job(j, ctx)
            except Exception as e:
                status = "failed"
                failed.add(j.job_id)
                log(f"ERRORE {j.job_id}: {e!r}\n{traceback.format_exc()}")
                mark(j, "failed", worker, (time.time() - t0) / 60, {"error": repr(e)[:500]})
            else:
                mark(j, status, worker, (time.time() - t0) / 60)
                log(f"FINE {j.job_id}: {status} in {(time.time() - t0) / 60:.1f} min")
            finally:
                release(j)
            if status == "interrupted":
                log("tempo della sessione quasi esaurito: mi fermo (il job riprenderà dal checkpoint)")
                return
            break
        if not started:
            if waiting and not guard.must_stop():
                time.sleep(20)
                continue
            log("nessun altro job eseguibile in questa sessione")
            break


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--gpus", default="auto", help="auto | none | 0,1")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--worker", default=None, help="(interno) id del worker")
    ap.add_argument("--skip-bench", action="store_true")
    a = ap.parse_args(argv)
    os.environ.setdefault("PIANOB_T0", str(time.time()))
    if a.synthetic:
        os.environ["PIANOB_SYNTHETIC"] = "1"
    synthetic = a.synthetic or os.environ.get("PIANOB_SYNTHETIC") == "1"

    if a.worker is not None:  # processo worker
        dev = "cuda" if _n_gpus() > 0 else "cpu"
        worker_loop(a.queue, a.worker, synthetic, dev)
        return

    kaggle_env.clear_locks()
    from .hpo.optuna_runner import fail_stale_trials
    fail_stale_trials()
    gpus = list(range(_n_gpus())) if a.gpus == "auto" else ([] if a.gpus == "none" else [int(g) for g in a.gpus.split(",")])
    if not gpus and kaggle_env.on_kaggle() and not synthetic and os.environ.get("PIANOB_ALLOW_CPU") != "1":
        from .experiments.queue import load_queue
        need = [j.job_id for j in load_queue(a.queue) if _needs_gpu(j)]
        if need:
            print("\n" + "!" * 80 + "\nERRORE: nessuna GPU in questa sessione, ma la coda contiene "
                  f"{len(need)} job che la richiedono (es. {need[0]}).\n"
                  "Su CPU un solo job può richiedere più delle 12 h della sessione: mi fermo subito.\n"
                  "Rimedio: Settings -> Accelerator -> GPU T4 x2, poi Save Version -> Save & Run All.\n"
                  "(Se la quota GPU settimanale è esaurita, Kaggle non assegna la GPU: controlla Settings -> Quotas.)\n"
                  + "!" * 80, flush=True)
            sys.exit(2)
    if not a.skip_bench:
        from .bench import print_estimates, run_bench
        from .experiments.queue import load_queue
        jobs = load_queue(a.queue)
        bench = run_bench(jobs, "cuda" if gpus else "cpu")
        print_estimates(jobs, bench, max(1, len(gpus)))
    if len(gpus) <= 1:
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        worker_loop(a.queue, "w0", synthetic, "cuda" if gpus else "cpu")
        return
    procs = []
    for g in gpus:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        cmd = [sys.executable, "-u", "-m", "pianob.run", "--queue", a.queue, "--worker", f"w{g}"]
        if synthetic:
            cmd.append("--synthetic")
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             cwd=str(Path(__file__).resolve().parents[1]))
        procs.append(p)
        time.sleep(5)  # sfasa l'avvio (claim dei job, caricamento dati)

    def pump(p):
        for line in p.stdout:
            print(line, end="", flush=True)

    threads = [threading.Thread(target=pump, args=(p,), daemon=True) for p in procs]
    for t in threads:
        t.start()
    for p in procs:
        p.wait()
    for t in threads:
        t.join(timeout=5)


if __name__ == "__main__":
    main()
