"""Optuna: TPE multivariato + Hyperband sulle epoche, storage SQLite riprendibile.

Obiettivo: media per soggetto della BA (finestra) sui soggetti di VALIDATION di split_fm
(sub080-099). I soggetti di test (sub100-122) non vengono mai caricati qui.
"""
from __future__ import annotations

import json
import shutil
import time

import numpy as np

from .. import kaggle_env
from ..config import TRAIN_DEFAULTS
from ..experiments.queue import Job
from ..experiments.registry import get_config
from .search_spaces import N_TRIALS, apply_params, sample

# studio -> config base
STUDY_CONFIG = {"tsception_ea": "tsception_ea", "eegnet_ea": "eegnet_ea", "cbramod_lora_r2": "cbramod_lora_r2",
                "cbramod_ft_official_lr": "cbramod_ft_official"}


def storage_path(study: str):
    p = kaggle_env.state_dir() / "optuna" / f"{study}.db"
    if not p.exists():
        for cand in kaggle_env.find_all(f"{study}.db"):
            shutil.copy2(cand, p)
            print(f"[optuna] riprendo lo studio da {cand}")
            break
    return p


def _storage(study: str):
    import optuna
    url = f"sqlite:///{storage_path(study)}"
    return optuna.storages.RDBStorage(url, engine_kwargs={"connect_args": {"timeout": 120}})


def fail_stale_trials() -> None:
    """Da chiamare all'avvio (prima dei worker): i trial RUNNING di una sessione interrotta -> FAIL."""
    try:
        import optuna
    except ImportError:
        return
    for db in (kaggle_env.state_dir() / "optuna").glob("*.db"):
        st = optuna.storages.RDBStorage(f"sqlite:///{db}")
        for s in optuna.get_all_study_summaries(st):
            study = optuna.load_study(study_name=s.study_name, storage=st)
            for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
                st.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
                print(f"[optuna] {s.study_name}: trial {t.number} interrotto -> FAIL")


def n_finished(study) -> int:
    import optuna
    S = optuna.trial.TrialState
    return len(study.get_trials(deepcopy=False, states=(S.COMPLETE, S.PRUNED)))


def write_best(study, name: str) -> dict:
    import optuna
    S = optuna.trial.TrialState
    comp = study.get_trials(deepcopy=False, states=(S.COMPLETE,))
    if not comp:
        return {}
    best = study.best_trial
    out = {"study": name, "params": best.params, "value": best.value, "trial": best.number,
           "n_complete": len(comp), "n_pruned": len(study.get_trials(deepcopy=False, states=(S.PRUNED,))),
           "best_epoch": best.user_attrs.get("best_epoch")}
    kaggle_env.write_json(kaggle_env.state_dir() / "hpo_best" / f"{name}.json", out)
    return out


def run_hpo_job(job: Job, ctx) -> str:
    try:
        import optuna
    except ImportError as e:
        raise RuntimeError("optuna non disponibile nell'immagine: attiva internet e `pip install optuna`") from e
    from ..experiments.jobs import build_banks, make_model, prepare
    from ..train.loop import train_model
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    name = job.config
    n_target = job.n_trials or N_TRIALS[name]
    base = get_config(STUDY_CONFIG[name])
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=10,
                                         seed=1000 * (job.seed + 1) + job.shard)
    pruner = optuna.pruners.HyperbandPruner(min_resource=5, max_resource=50, reduction_factor=3)
    study = optuna.create_study(study_name=name, storage=_storage(name), direction="maximize",
                                sampler=sampler, pruner=pruner, load_if_exists=True)
    if name == "cbramod_ft_official_lr" and not study.trials:
        for lr in (5e-5, 1e-4, 2e-4):
            for ml in (True, False):
                study.enqueue_trial({"lr": lr, "multi_lr": ml})
    hpo_job = Job("train", base.name, "split_fm", 0, job.seed, overrides=dict(job.overrides))
    trial_minutes = []

    def objective(trial):
        params = sample(name, trial)
        cfg = apply_params(base, params)
        hp = {**TRAIN_DEFAULTS, **cfg.hparams, **job.overrides.get("hparams", {})}
        jd = prepare(hpo_job, ctx, cfg)
        assert not set(jd.split.test_subjects) & set(jd.split.val_subjects)
        win_len = int(round(cfg.window_s * ctx.adapter.fs))
        banks = build_banks(hpo_job, ctx, jd, win_len, need_train=True, need_test=False)  # test mai caricato
        import torch
        torch.manual_seed(trial.number)
        model = make_model(ctx, cfg, len(jd.channels), win_len, jd.n_classes)
        t0 = time.time()
        res = train_model(model, hp, cfg.recipe, banks["train"], banks["val"], jd.n_classes, ctx.device,
                          seed=trial.number, trial=trial, guard=ctx.guard,
                          log=lambda m: None)
        trial_minutes.append((time.time() - t0) / 60)
        trial.set_user_attr("best_epoch", res["best_epoch"])
        trial.set_user_attr("minutes", trial_minutes[-1])
        ctx.log(f"  [optuna {name}] trial {trial.number}: {res['status']} val BA {res['best_score']:.4f} "
                f"({trial_minutes[-1]:.1f} min) {params}")
        if res["status"] == "pruned":
            raise optuna.TrialPruned()
        if res["status"] == "interrupted":
            raise RuntimeError("interrotto dalla guardia del tempo")
        return res["best_score"]

    status = "completed"
    while n_finished(study) < n_target:
        est = np.median(trial_minutes) if trial_minutes else job.est_min
        if ctx.guard is not None and not ctx.guard.can_start(est or 0):
            status = "interrupted"
            break
        try:
            study.optimize(objective, n_trials=1, catch=())
        except RuntimeError as e:
            if "interrotto" in str(e):
                status = "interrupted"
                break
            raise
    best = write_best(study, name)
    ctx.log(f"[optuna {name}] trial finiti {n_finished(study)}/{n_target}; migliore: {best.get('value')} {best.get('params')}")
    return status
