"""Micro-benchmark: stima dei minuti per job a partire da pochi step di training su dati casuali con
le shape reali. Il risultato alimenta la guardia del tempo."""
from __future__ import annotations

import json
import time

import numpy as np

from . import kaggle_env
from .config import TRAIN_DEFAULTS
from .data import faced_meta as FM
from .experiments.queue import Job
from .experiments.registry import resolve_config

N_TRAIN_SUBJ = 80
N_VAL_TEST_SUBJ = 43


def _n_windows(window_s: float, overlap: float) -> int:
    win = window_s
    stride = win * (1 - overlap)
    return int((30 - win) // stride + 1)


def bench_config(name: str, device: str, steps: int = 6) -> dict:
    import torch
    from .experiments.jobs import make_model
    from .experiments.jobs import Context
    from .data.adapters import FacedAdapter
    from .train.loop import build_optimizer
    try:
        cfg, _ = resolve_config(name)
    except RuntimeError:
        from .experiments.registry import get_config
        cfg = get_config(name)
    hp = {**TRAIN_DEFAULTS, **cfg.hparams}
    fs = FM.FS_TARGET
    n_ch = 28 if cfg.channel_layout == "hemispheric" else 32
    win = int(cfg.window_s * fs)
    ctx = Context(FacedAdapter(synthetic=True).load(), device=device)
    cfg_rand = cfg.__class__.from_dict({**cfg.to_dict(), "pretrained": False})
    model = make_model(ctx, cfg_rand, n_ch, win, 9).to(device)
    bs = int(hp["batch_size"])
    opt, _ = build_optimizer(model, hp, cfg.recipe, 10)
    use_amp = device.startswith("cuda")
    from .train.loop import make_scaler
    scaler = make_scaler(use_amp)
    x = torch.randn(bs, n_ch, win, device=device)
    y = torch.randint(0, 9, (bs,), device=device)
    crit = torch.nn.CrossEntropyLoss()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
        loss = crit(out.float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    model.train()
    for _ in range(2):
        step()
    if use_amp:
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        step()
    if use_amp:
        torch.cuda.synchronize()
    sec_step = (time.time() - t0) / steps
    n_tr = N_TRAIN_SUBJ * 28 * _n_windows(cfg.window_s, cfg.overlap_train)
    n_ev = 20 * 28 * _n_windows(cfg.window_s, 0)
    epoch_s = sec_step * np.ceil(n_tr / bs) + sec_step / 3 * np.ceil(n_ev / bs)
    exp_epochs = hp["epochs"] if not hp.get("patience") else min(hp["epochs"], 30)
    job_min = (exp_epochs * epoch_s + 60) / 60
    del model, opt
    if use_amp:
        torch.cuda.empty_cache()
    return {"sec_step": sec_step, "epoch_min": epoch_s / 60, "job_min_split_fm": job_min, "epochs_expected": exp_epochs}


PROTOCOL_FACTOR = {"split_fm": 1.0, "split_fm_lco": 0.75, "lnso10": 1.35, "lnso10_lco": 1.0,
                   "split_fm_sizematched": 0.75, "lnso10_sizematched": 1.0}


def empirical_minutes() -> dict:
    """Minuti mediani effettivi per (config, protocollo) dai job già completati (anche di sessioni
    precedenti): più affidabili del micro-benchmark, che sovrastima i job con early stopping."""
    from .experiments.queue import DONE_STATUSES, done_status
    acc: dict = {}
    for jid, r in done_status().items():
        parts = jid.split("__")
        if r["status"] in DONE_STATUSES and len(parts) >= 5 and r.get("minutes"):
            acc.setdefault((parts[0], parts[1]), []).append(float(r["minutes"]))
    return {k: float(np.median(v)) for k, v in acc.items() if len(v) >= 2}


def estimate_job(job: Job, bench: dict, empirical: dict | None = None) -> float:
    if empirical and job.kind == "train" and not job.overrides.get("max_subjects"):
        e = empirical.get((job.config, job.protocol)) \
            or empirical.get((job.config, job.protocol.replace("_sizematched", "_lco"))) \
            or empirical.get((job.config, "split_fm"))
        if e is not None:
            return 1.2 * e
    if job.kind == "report":
        return 1.0
    if job.overrides.get("max_subjects"):
        return 2.0
    if job.kind == "hpo":
        b = bench.get(job.config) or bench.get({"cbramod_ft_official_lr": "cbramod_ft_official"}.get(job.config, ""), {})
        return 0.6 * b.get("job_min_split_fm", 10)   # un trial (con pruning in media meno di un job)
    from .experiments.registry import get_config
    cfg = get_config(job.config)   # niente Optuna qui: la stima non deve fallire se mancano gli iperparametri
    if cfg.family == "const":
        return 0.5
    if cfg.family == "features":
        return 3.0 + 0.04 * job.n_perm
    b = bench.get(job.config, {})
    m = b.get("job_min_split_fm", 15.0) * PROTOCOL_FACTOR.get(job.protocol, 1.0)
    return m + (3.0 if cfg.tta_bnalpha else 0.0)


def run_bench(jobs: list[Job], device: str) -> dict:
    path = kaggle_env.state_dir() / "bench" / "bench.json"
    bench = json.loads(path.read_text()) if path.exists() else {}
    names = []
    for j in jobs:
        if j.kind == "hpo":
            from .hpo.optuna_runner import STUDY_CONFIG
            names.append(STUDY_CONFIG[j.config])
        elif j.kind == "train":
            from .experiments.registry import get_config
            cfg = get_config(j.config)
            if cfg.family == "neural" and cfg.model != "reve":
                names.append(j.config)
    gpu = (kaggle_env.gpu_info() or [{"name": "cpu"}])[0]["name"]
    for n in dict.fromkeys(names):
        key_ok = n in bench and bench[n].get("gpu") == gpu
        if key_ok:
            continue
        try:
            b = bench_config(n, device)
            b["gpu"] = gpu
            bench[n] = b
        except Exception as e:  # il benchmark non deve bloccare la coda
            print(f"[bench] {n}: errore {e!r}")
    kaggle_env.write_json(path, bench)
    return bench


def print_estimates(jobs: list[Job], bench: dict, n_workers: int) -> float:
    from .experiments.queue import done_status, DONE_STATUSES
    done = done_status()
    rows, tot = {}, 0.0
    for j in jobs:
        if done.get(j.job_id, {}).get("status") in DONE_STATUSES:
            continue
        m = estimate_job(j, bench)
        if j.kind == "hpo":
            m *= max(1, j.n_trials) / 2  # due shard
        key = (j.config, j.kind)
        r = rows.setdefault(key, [0, 0.0])
        r[0] += 1
        r[1] += m
        tot += m
    print("| config | tipo | job da fare | minuti stimati |\n|---|---|---|---|")
    for (c, k), (n, m) in rows.items():
        print(f"| {c} | {k} | {n} | {m:.0f} |")
    print(f"\nTotale stimato: {tot / 60:.1f} GPU-h; con {n_workers} worker ~{tot / 60 / max(1, n_workers):.1f} h di sessione")
    for name, b in bench.items():
        print(f"  bench {name}: {b['epoch_min']:.2f} min/epoca, ~{b['job_min_split_fm']:.0f} min/job su split_fm ({b.get('gpu')})")
    return tot
