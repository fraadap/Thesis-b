"""Esecuzione di un job: preparazione dei dati, training/fit, valutazione, scrittura dei risultati."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import kaggle_env
from ..config import TRAIN_DEFAULTS, ExpConfig
from ..data.adapters import FacedAdapter
from ..data.splits import Split, check_windows_in_split, make_split, permute_clip_labels
from ..eval.metrics import evaluate_predictions, subject_mean
from ..prep.alignment import align_subject
from ..prep.filters import kept_channels
from .queue import Job
from .registry import resolve_config


@dataclass
class Context:
    adapter: FacedAdapter
    device: str = "cpu"
    worker: str = "w0"
    guard: object = None
    log: callable = print
    weights_path: Path | None = None
    caches: dict = field(default_factory=dict)


# ------------------------------------------------------------------ preparazione
@dataclass
class JobData:
    cfg: ExpConfig
    split: Split
    clip_labels: np.ndarray        # label per clip (eventualmente permutate), -1 = esclusa
    n_classes: int
    channels: list[int]
    channel_names: list[str]


def prepare(job: Job, ctx: Context, cfg: ExpConfig) -> JobData:
    ad = ctx.adapter
    lab = ad.labels(job.task)
    task_clips = ad.task_clips(job.task)
    split = make_split(job.protocol, job.fold, task_clips, ad.subjects(), clip_labels=ad.meta["clip_labels"],
                       draw=job.draw or 0, max_subjects=job.overrides.get("max_subjects"))
    if job.perm is not None:
        lab = permute_clip_labels(lab, job.perm, task_clips)
    kept = kept_channels(ad.channel_names(), job.drop)
    if cfg.channel_layout == "hemispheric":
        from ..models.tsception import hemispheric_order
        kept = [kept[i] for i in hemispheric_order(kept)]
    idx = ad.channel_indices(kept)
    n_classes = int(ad.labels(job.task).max() + 1)
    return JobData(cfg, split, np.asarray(lab), n_classes, idx, kept)


def load_aligned(ctx: Context, jd: JobData, subjects, clips, fit_clips, band, win_len: int,
                 rescale: float | None = None) -> np.ndarray:
    """(n_subj * n_clips, C, T) float16, allineati per soggetto sulle sole clip `fit_clips`."""
    return load_aligned_per_subject(ctx, jd, [(s, list(clips), list(fit_clips)) for s in subjects],
                                    band, win_len, rescale)


def load_aligned_per_subject(ctx: Context, jd: JobData, plan, band, win_len: int,
                             rescale: float | None = None) -> np.ndarray:
    """plan = [(soggetto, clip, clip_per_l'allineamento)]: le clip possono variare per soggetto
    (protocolli *_sizematched). Trial concatenati nell'ordine del piano."""
    cfg = jd.cfg
    n = sum(len(c) for _, c, _ in plan)
    out, i = None, 0
    for s, clips, fit_clips in plan:
        x = ctx.adapter.trials([s], clips, channels=jd.channels, band=band)[0]      # (n_clip, C, T)
        if cfg.align != "none":
            fit = x[[clips.index(c) for c in fit_clips]]
            x = align_subject(x, cfg.align, win_len, cfg.align_ref, fit_trials=fit)
            if rescale is not None:
                x = x * np.float32(rescale)
        if out is None:
            out = np.empty((n, *x.shape[1:]), dtype=np.float16)
        out[i:i + len(clips)] = x.astype(np.float16)
        i += len(clips)
    return out


def _train_plan(sp: Split):
    return [(s, sp.subject_train_clips(s), sp.subject_train_clips(s)) for s in sp.train_subjects]


def _plan_meta(plan, clip_labels):
    s = np.concatenate([np.full(len(c), sub) for sub, c, _ in plan])
    c = np.concatenate([np.asarray(c) for _, c, _ in plan])
    return s, c, np.asarray(clip_labels)[c]


def _test_clips(sp: Split, conds: dict) -> list[int]:
    """Clip caricate per i soggetti di test. Per *_sizematched si carica l'intera sessione (come nel
    leave-clip-out), così l'allineamento dei soggetti di test è identico e gli esempi T_k coincidono."""
    ev = sorted({c for cl in conds.values() for c in cl})
    return sorted(set(ev) | set(sp.all_clips)) if sp.is_sizematched else ev


def tk_conditions(sp: Split) -> dict:
    """Valutazione aggiuntiva dei protocolli con clip condivise (split_fm, lnso10) sui sottoinsiemi T_k:
    serve come BA_full nella scomposizione quantità di dati / novità dello stimolo."""
    if sp.is_lco or sp.is_sizematched or sp.protocol not in ("split_fm", "lnso10"):
        return {}
    from ..data.splits import heldout_sets
    return {f"tk{k}": [c for c in v if c in sp.train_clips]
            for k, v in heldout_sets(task_clips=sp.all_clips or sp.train_clips).items()}


def _trial_meta(subjects, clips, clip_labels):
    s = np.repeat(np.asarray(subjects), len(clips))
    c = np.tile(np.asarray(clips), len(subjects))
    return s, c, np.asarray(clip_labels)[c]


def raw_rms(ctx: Context, jd: JobData, subjects, clips, band) -> float:
    x = ctx.adapter.trials(subjects[:5], clips, channels=jd.channels, band=band)
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


# ------------------------------------------------------------------ righe dei risultati
def result_rows(job: Job, config_name: str, ev: dict, cond: str, variant: str, extra: dict) -> list[dict]:
    rows = []
    for s, m in ev["per_subject"].items():
        rows.append({"config": config_name, "protocol": job.protocol, "fold": job.fold, "seed": job.seed,
                     "variant": variant, "test_condition": cond, "task": job.task, "subject": s,
                     **{k: (None if isinstance(v, float) and np.isnan(v) else round(v, 5) if isinstance(v, float) else v)
                        for k, v in m.items()}, **extra})
    return rows


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def env_extra(ctx: Context, minutes: float) -> dict:
    gpus = kaggle_env.gpu_info()
    return {"gpu": gpus[0]["name"] if gpus else "cpu", "minutes": round(minutes, 2),
            "version": kaggle_env.code_version()}


# ------------------------------------------------------------------ dispatcher
def run_job(job: Job, ctx: Context) -> str:
    if job.kind == "hpo":
        from ..hpo.optuna_runner import run_hpo_job
        return run_hpo_job(job, ctx)
    cfg, hp_source = resolve_config(job.config)
    if cfg.family == "const":
        return run_const(job, ctx, cfg)
    if cfg.family == "features":
        return run_features(job, ctx, cfg)
    return run_neural(job, ctx, cfg, hp_source)


def _write(job, ctx, rows, record):
    st = kaggle_env.state_dir()
    append_jsonl(st / "results" / f"rows_{ctx.worker}.jsonl", rows)
    append_jsonl(st / "results" / f"jobs_{ctx.worker}.jsonl", [record])


def _conditions(jd: JobData):
    return jd.split.eval_conditions()


# ------------------------------------------------------------------ predittore costante
def run_const(job: Job, ctx: Context, cfg: ExpConfig) -> str:
    t0 = time.time()
    jd = prepare(job, ctx, cfg)
    sp = jd.split
    n_win = int(30 // cfg.window_s)
    train_lab = np.repeat(jd.clip_labels[sp.train_clips], n_win * len(sp.train_subjects))
    maj = int(np.bincount(train_lab, minlength=jd.n_classes).argmax())
    rows, rec = [], {"job_id": job.job_id, "config": cfg.name, "majority_class": maj}
    for cond, clips in _conditions(jd).items():
        s, c, l = _trial_meta(sp.test_subjects, clips, jd.clip_labels)
        s, c, l = (np.repeat(a, n_win) for a in (s, c, l))
        P = np.zeros((len(l), jd.n_classes), dtype=np.float32)
        P[:, maj] = 1
        ev = evaluate_predictions(P, s, c, l)
        rows += result_rows(job, cfg.name, ev, cond, job.variant, env_extra(ctx, (time.time() - t0) / 60))
        rec[f"pooled_{cond}"] = ev["pooled"]
    _write(job, ctx, rows, rec)
    return "completed"


# ------------------------------------------------------------------ baseline su feature (CPU)
def _feature_cache(ctx: Context, cfg: ExpConfig, jd: JobData, band, subjects) -> dict:
    from ..features.spectral import feature_matrix
    key = f"feat_{cfg.features}_w{cfg.window_s:g}_{'bp%g-%g' % tuple(band) if band else 'full'}_{len(jd.channels)}ch_{hash(tuple(jd.channels)) % 10**6}"
    d = kaggle_env.cache_dir() / key
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in subjects:
        f = d / f"sub{s:03d}.npy"
        if f.exists():
            out[s] = np.load(f)
            continue
        x = ctx.adapter.trials([s], None, channels=jd.channels, band=band, scale=1.0)[0]   # microvolt
        F = feature_matrix(cfg.features, x, jd.channel_names, ctx.adapter.fs, cfg.window_s)
        F = F.reshape(x.shape[0], -1, F.shape[-1])                                          # (28, n_win, F)
        tmp = d / f"sub{s:03d}.{os.getpid()}.tmp.npy"
        np.save(tmp, F.astype(np.float32))
        os.replace(tmp, f)
        out[s] = F
    return out


def _xy(feats, subjects, clips, clip_labels, zsub_clips=None):
    """`clips` / `zsub_clips`: lista comune oppure funzione soggetto -> lista (protocolli *_sizematched)."""
    X, S, Cc = [], [], []
    for s in subjects:
        F = feats[s]
        cl = clips(s) if callable(clips) else clips
        zc = zsub_clips(s) if callable(zsub_clips) else zsub_clips
        if zc is not None:  # z-score per soggetto sulle clip disponibili (label-free)
            ref = F[zc].reshape(-1, F.shape[-1])
            F = (F - ref.mean(0)) / (ref.std(0) + 1e-8)
        Fx = F[cl]
        X.append(Fx.reshape(-1, F.shape[-1]))
        S.append(np.full(Fx.shape[0] * Fx.shape[1], s))
        Cc.append(np.repeat(cl, Fx.shape[1]))
    X, S, Cc = np.concatenate(X), np.concatenate(S), np.concatenate(Cc)
    return X, S, Cc, np.asarray(clip_labels)[Cc]


def _fit_clf(kind, C, X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    clf = LogisticRegression(C=C, max_iter=3000) if kind == "lr" else LinearSVC(C=C, max_iter=5000, random_state=seed)
    return make_pipeline(StandardScaler(), clf).fit(X, y)


def _proba(model, X, n_classes):
    est = model[-1]
    cls = est.classes_
    if hasattr(est, "predict_proba"):
        p = model.predict_proba(X)
    else:
        z = model.decision_function(X)
        if z.ndim == 1:
            z = np.stack([-z, z], 1)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    out = np.zeros((len(X), n_classes), dtype=np.float32)
    out[:, cls] = p
    return out


def run_features(job: Job, ctx: Context, cfg: ExpConfig) -> str:
    t0 = time.time()
    jd = prepare(job, ctx, cfg)
    sp = jd.split
    conds = {**_conditions(jd), **tk_conditions(sp)}
    all_eval = _test_clips(sp, conds)
    subs = sp.train_subjects + sp.val_subjects + sp.test_subjects
    feats = _feature_cache(ctx, cfg, jd, job.band, subs)
    z = cfg.feature_zsub

    def data(lab):
        tr = _xy(feats, sp.train_subjects, sp.subject_train_clips, lab, sp.subject_train_clips if z else None)
        va = _xy(feats, sp.val_subjects, sp.train_clips, lab, sp.train_clips if z else None)
        te = {cond: _xy(feats, sp.test_subjects, cl, lab, all_eval if z else None) for cond, cl in conds.items()}
        return tr, va, te

    tr, va, te = data(jd.clip_labels)
    allowed = sorted(set().union(*[sp.subject_train_clips(s) for s in sp.train_subjects]))
    check_windows_in_split(tr[1], tr[2], sp.train_subjects, allowed, "train")
    if sp.is_sizematched:
        _check_train_size(sp, len(tr[1]) // max(1, sp.n_train_trials()), len(tr[1]), job)
    best = (-1, None, None)
    grid = []
    for kind in cfg.hp("clf", ["lr"]):
        for C in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            m = _fit_clf(kind, C, tr[0], tr[3], job.seed)
            ev = evaluate_predictions(_proba(m, va[0], jd.n_classes), va[1], va[2], va[3])
            score = subject_mean(ev["per_subject"])
            grid.append({"clf": kind, "C": C, "val_ba": round(score, 4)})
            if score > best[0]:
                best = (score, kind, C)
    _, kind, C = best
    model = _fit_clf(kind, C, tr[0], tr[3], job.seed)
    rows, rec = [], {"job_id": job.job_id, "config": cfg.name, "clf": kind, "C": C, "val_ba": best[0], "grid": grid,
                     **_size_info(sp)}
    extra = {**env_extra(ctx, 0), **_draw_extra(sp)}
    preds = {}
    for cond, (X, S, Cc, y) in te.items():
        P = _proba(model, X, jd.n_classes)
        ev = evaluate_predictions(P, S, Cc, y)
        rows += result_rows(job, cfg.name, ev, cond, job.variant, extra)
        rec[f"pooled_{cond}"] = ev["pooled"]
        rec[f"confusion_{cond}"] = ev["confusion_win"]
        preds[cond] = (P, S, Cc, y)
    # Permutazioni a livello di clip con riaddestramento (stessi iperparametri).
    if job.n_perm and job.perm is None:
        task_clips = ctx.adapter.task_clips(job.task)
        null = {cond: [] for cond in conds}
        for k in range(job.n_perm):
            lab_k = permute_clip_labels(jd.clip_labels, k, task_clips)
            trk, _, tek = data(lab_k)
            mk = _fit_clf(kind, C, trk[0], trk[3], job.seed)
            for cond, (X, S, Cc, y) in tek.items():
                ev = evaluate_predictions(_proba(mk, X, jd.n_classes), S, Cc, y)
                null[cond].append(subject_mean(ev["per_subject"]))
                if sp.is_lco or sp.is_sizematched:  # serve per la correzione del Δstim, per soggetto
                    rows += result_rows(job, cfg.name, ev, cond, _join(job.variant, f"perm{k}"), extra)
        for cond in conds:
            obs = subject_mean(evaluate_predictions(*preds[cond])["per_subject"])
            nl = np.asarray(null[cond])
            rec[f"perm_{cond}"] = {"observed": obs, "null_mean": float(nl.mean()),
                                   "null_p95": float(np.percentile(nl, 95)),
                                   "p": float((1 + np.sum(nl >= obs)) / (1 + len(nl))), "n_perm": len(nl)}
    minutes = (time.time() - t0) / 60
    for r in rows:
        r["minutes"] = round(minutes, 2)
    rec["minutes"] = minutes
    _write(job, ctx, rows, rec)
    return "completed"


def _check_train_size(sp: Split, windows_per_trial: int, n_windows: int, job: Job) -> None:
    """Numero di finestre di training identico al fold leave-clip-out corrispondente."""
    from ..data.splits import make_split
    lco = make_split(sp.protocol.replace("_sizematched", "_lco"), sp.fold, sp.all_clips, None,
                     max_subjects=job.overrides.get("max_subjects"))
    lco_trials = sum(len(lco.subject_train_clips(s)) for s in lco.train_subjects if s in sp.train_subjects)
    assert n_windows == lco_trials * windows_per_trial, \
        f"finestre di training {n_windows} != leave-clip-out {lco_trials * windows_per_trial}"


def _size_info(sp: Split) -> dict:
    if not sp.is_sizematched:
        return {}
    from ..data.splits import coverage
    cov = coverage(sp)
    return {"draw": sp.draw, "heldout_clips": sp.heldout_clips, "coverage": cov,
            "coverage_frac_mean": float(np.mean(list(cov.values()))) / max(1, len(sp.train_subjects))}


def _draw_extra(sp: Split) -> dict:
    return {"draw": sp.draw} if sp.is_sizematched else {}


def _join(a, b):
    return b if a == "base" else f"{a}_{b}"


# ------------------------------------------------------------------ modelli neurali
def build_banks(job: Job, ctx: Context, jd: JobData, win_len: int, need_train=True, need_test=True):
    from ..train.loop import TrialBank
    cfg, sp = jd.cfg, jd.split
    conds = _conditions(jd)
    all_eval = _test_clips(sp, conds)
    plan = _train_plan(sp)
    rescale = raw_rms(ctx, jd, sp.train_subjects, sp.train_clips, job.band) if cfg.align_rescale else None
    key = (job.protocol, job.fold, job.task, tuple(job.band or ()), job.drop, cfg.align, cfg.align_ref,
           win_len, rescale is not None, tuple(jd.channels), tuple(sp.train_subjects), need_test, sp.draw)
    if key in ctx.caches:
        arrs = ctx.caches[key]
    else:
        while len(ctx.caches) >= int(os.environ.get("PIANOB_CACHE_ENTRIES", 3)):
            ctx.caches.pop(next(iter(ctx.caches)))
        arrs = {
            "train": load_aligned_per_subject(ctx, jd, plan, job.band, win_len, rescale) if need_train else None,
            "val": load_aligned(ctx, jd, sp.val_subjects, sp.train_clips, sp.train_clips, job.band, win_len, rescale),
            "test": load_aligned(ctx, jd, sp.test_subjects, all_eval, all_eval, job.band, win_len, rescale)
            if need_test else None,
        }
        ctx.caches[key] = arrs
    dev = ctx.device
    banks = {}
    if need_train:
        s, c, l = _plan_meta(plan, jd.clip_labels)
        banks["train"] = TrialBank(arrs["train"], s, c, l, win_len, jd.cfg.overlap_train, dev)
        ws = banks["train"].ws
        for sub in sp.train_subjects:   # nessuna finestra fuori dalle clip di training del suo soggetto
            m = ws.subject == sub
            check_windows_in_split(ws.subject[m], ws.clip[m], [sub], sp.subject_train_clips(sub), "train")
        if sp.is_sizematched:
            per_trial = len(ws.win_trial) // max(1, len(ws.trial_subject))
            _check_train_size(sp, per_trial, banks["train"].n, job)
    s, c, l = _trial_meta(sp.val_subjects, sp.train_clips, jd.clip_labels)
    banks["val"] = TrialBank(arrs["val"], s, c, l, win_len, 0.0, dev)
    check_windows_in_split(banks["val"].ws.subject, banks["val"].ws.clip, sp.val_subjects, sp.train_clips, "val")
    if need_test:
        s, c, l = _trial_meta(sp.test_subjects, all_eval, jd.clip_labels)
        banks["test"] = TrialBank(arrs["test"], s, c, l, win_len, 0.0, dev)
        check_windows_in_split(banks["test"].ws.subject, banks["test"].ws.clip, sp.test_subjects, all_eval, "test")
    return banks


def make_model(ctx: Context, cfg: ExpConfig, n_ch: int, win_len: int, n_classes: int):
    from ..models.heads import build_model
    wp = ctx.weights_path
    if cfg.model == "cbramod" and cfg.pretrained and wp is None:
        if ctx.adapter.synthetic:
            ctx.log("  [sintetico] pesi CBraMod assenti: uso pesi casuali")
            cfg = ExpConfig.from_dict({**cfg.to_dict(), "pretrained": False})
        else:
            raise FileNotFoundError("Pesi CBraMod non trovati (weights/cbramod_pretrained.pth nel Dataset dati).")
    return build_model(cfg, n_ch, win_len, n_classes, ctx.adapter.fs, pretrained_path=wp)


def run_neural(job: Job, ctx: Context, cfg: ExpConfig, hp_source: str) -> str:
    import torch
    from ..models.lora import count_trainable
    from ..train.loop import predict, train_model
    from ..train.tta import bnalpha_predict, collect_subject_bn_stats, has_bn
    t0 = time.time()
    hp = {**TRAIN_DEFAULTS, **cfg.hparams, **job.overrides.get("hparams", {})}
    torch.manual_seed(job.seed)
    np.random.seed(job.seed % 2**32)
    jd = prepare(job, ctx, cfg)
    sp = jd.split
    win_len = int(round(cfg.window_s * ctx.adapter.fs))
    banks = build_banks(job, ctx, jd, win_len)
    model = make_model(ctx, cfg, len(jd.channels), win_len, jd.n_classes)
    params = count_trainable(model)
    ctx.log(f"  {job.job_id}: train {banks['train'].n} fin., val {banks['val'].n}, test {banks['test'].n}; "
            f"parametri allenabili {params['trainable']:,}/{params['total']:,}; hp da {hp_source}")
    st = kaggle_env.state_dir()
    ckpt = st / "ckpt" / f"{job.job_id}.pt"
    res = train_model(model, hp, cfg.recipe, banks["train"], banks["val"], jd.n_classes, ctx.device,
                      job.seed, ckpt_path=ckpt, guard=ctx.guard, log=ctx.log)
    if res["status"] == "interrupted":
        return "interrupted"
    model.load_state_dict(res["best_state"])
    test = banks["test"]
    P = predict(model, test, ctx.device, amp=hp.get("amp", True))
    minutes = (time.time() - t0) / 60
    extra = {**env_extra(ctx, minutes), "hp_source": hp_source, "best_epoch": res["best_epoch"]}
    rows = []
    rec = {"job_id": job.job_id, "config": cfg.name, "status": res["status"], "best_epoch": res["best_epoch"],
           "val_best": res["best_score"], "history": res["history"], "params": params, "hp": hp,
           "hp_source": hp_source, "cfg": cfg.to_dict(), "minutes": minutes,
           "pretrained": getattr(model, "pretrained_info", None)}
    from ..eval.shortcuts import test_label_permutation
    conds = _conditions(jd)
    rec.update(_size_info(sp))
    extra.update(_draw_extra(sp))
    for cond, clips in {**conds, **tk_conditions(sp)}.items():
        m = np.isin(test.ws.clip, clips)
        ev = evaluate_predictions(P[m], test.ws.subject[m], test.ws.clip[m], test.ws.label[m])
        rows += result_rows(job, cfg.name, ev, cond, job.variant, extra)
        rec[f"pooled_{cond}"] = ev["pooled"]
        rec[f"confusion_{cond}"] = ev["confusion_win"]
        if job.perm is None and cond in conds:
            rec[f"testperm_{cond}"] = test_label_permutation(P[m], test.ws.subject[m], test.ws.clip[m],
                                                             test.ws.label[m], n_perm=1000, seed=job.seed)
    # predizioni salvate (servono a permutazioni/analisi successive)
    pdir = st / "artifacts" / "preds"
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pdir / f"{job.job_id}.npz", probs=P.astype(np.float16), subject=test.ws.subject,
                        clip=test.ws.clip, label=test.ws.label)

    # BN-α (regime offline e online), n0 scelto sulla validation
    if cfg.tta_bnalpha and has_bn(model) and job.perm is None and not job.overrides.get("skip_tta") \
            and not sp.is_sizematched:
        val_out = bnalpha_predict(model, banks["val"], ctx.device, n0_list=(8, 32, 128), first_n=(), stream=False)
        vs = {k: subject_mean(evaluate_predictions(v, banks["val"].ws.subject, banks["val"].ws.clip,
                                                   banks["val"].ws.label)["per_subject"]) for k, v in val_out.items()}
        n0_sel = int(max(vs, key=vs.get).split("n0")[-1])
        rec["bnalpha_val"] = vs
        rec["bnalpha_n0_selected"] = n0_sel
        tout = bnalpha_predict(model, test, ctx.device, n0_list=(0, 8, 32, 128))
        for tv, Pt in tout.items():
            for cond, clips in conds.items():
                m = np.isin(test.ws.clip, clips)
                ev = evaluate_predictions(Pt[m], test.ws.subject[m], test.ws.clip[m], test.ws.label[m])
                rows += result_rows(job, f"{cfg.name}_bnalpha", ev, cond, _join(job.variant, tv), extra)
                if tv.endswith(f"_n0{n0_sel}"):  # copia con nome stabile per n0 scelto in validation
                    rows += result_rows(job, f"{cfg.name}_bnalpha", ev, cond,
                                        _join(job.variant, tv.replace(f"_n0{n0_sel}", "_n0sel")), extra)
    # sonda dell'identità della clip (solo split senza LCO)
    if job.probe and not sp.is_lco:
        from ..eval.shortcuts import clip_probe
        tr = banks["train"]
        idx_tr = np.where(np.isin(tr.ws.win_start % win_len, [0]))[0]   # finestre non sovrapposte
        _, Etr = predict(model, tr, ctx.device, idx=idx_tr, return_embed=True)
        _, Ete = predict(model, test, ctx.device, return_embed=True)
        pr = clip_probe(Etr, tr.ws.clip[idx_tr], Ete, test.ws.clip, test.ws.label, jd.clip_labels,
                        test_subj=test.ws.subject)
        rec["clip_probe"] = {k: v for k, v in pr.items() if k != "per_subject"}
        rec["model_ba_subject_mean"] = subject_mean(evaluate_predictions(P, test.ws.subject, test.ws.clip, test.ws.label)["per_subject"])
        for s, d in pr["per_subject"].items():
            rows.append({"config": cfg.name, "protocol": job.protocol, "fold": job.fold, "seed": job.seed,
                         "variant": _join(job.variant, "clipprobe"), "test_condition": "all", "task": job.task,
                         "subject": s, "ba_win": round(d["ba_from_clip"], 5), "clip_acc": round(d["clip_acc"], 5),
                         **extra})
    # hook Proposta 3: statistiche BN per soggetto di training
    if has_bn(model) and job.protocol == "split_fm" and job.perm is None and not job.band and not job.drop:
        bdir = st / "artifacts" / "bnstats"
        bdir.mkdir(parents=True, exist_ok=True)
        torch.save(collect_subject_bn_stats(model, banks["train"], ctx.device), bdir / f"{job.job_id}.pt")
    if cfg.model != "cbramod" or job.overrides.get("save_model"):
        (st / "models").mkdir(exist_ok=True)
        torch.save(res["best_state"], st / "models" / f"{job.job_id}.pt")
    rec["minutes"] = (time.time() - t0) / 60
    _write(job, ctx, rows, rec)
    if ckpt.exists():
        ckpt.unlink()
    return res["status"]
