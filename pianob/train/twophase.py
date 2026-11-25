"""Ricetta a due fasi per CBraMod (Sessione 02, Parte A): linear probing, poi fine-tuning (LP-FT).

Riferimento del metodo: Kumar et al., ICLR 2022, "Fine-Tuning can Distort Pretrained Features and
Underperform Out-of-Distribution". Dettagli della nostra versione (dal prompt della Sessione 02):

- Fase LP: backbone congelato (requires_grad=False, backbone in eval), si allena solo la testa.
- Fase FT: tutta la rete, due gruppi di parametri (backbone, testa) a lr diverse; l'ottimizzatore si
  ricrea all'inizio della fase FT; gradient clipping a 1.0.
- In entrambe: AdamW (weight decay 0.05), one-cycle per fase con frazione di warm-up `warmup_frac`,
  label smoothing, batch 64, AMP fp16 + GradScaler.
- Selezione: BA media per soggetto in validation sulle epoche della fase FT, patience 10 (FT).
  Le curve di entrambe le fasi restano nella history.
- Standardizzazione per soggetto degli embedding (teste a pooling): statistiche ricalcolate una volta
  in LP (backbone fermo), all'inizio di ogni epoca FT (train) e prima di ogni validation; sulla rete
  finale prima del test le ricalcola chi chiama (experiments/jobs.py).

Checkpoint e ripresa come `train_model`: si salva la fase corrente, l'epoca e gli stati di ottimizzatore,
scheduler e scaler.
"""
from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .loop import (_rng_state, _set_rng_state, batch_subjects, compute_subject_stats, forward_model, make_scaler,
                   validate)

TWOPHASE_DEFAULTS = {
    "lp_epochs": 5, "ft_epochs": 20, "lr_head_lp": 1e-3, "lr_backbone_ft": 3e-5, "lr_head_ft": 3e-4,
    "warmup_frac": 0.1, "label_smoothing": 0.1, "weight_decay": 0.05, "batch_size": 64, "clip_grad_ft": 1.0,
    "patience": 10, "select_metric": "ba_subject_mean", "amp": True,
}


def _split_params(model):
    back = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    return back, head


def set_phase(model, phase: str) -> None:
    """LP: solo la testa allenabile; FT: tutto allenabile."""
    back, head = _split_params(model)
    for p in back:
        p.requires_grad = phase == "ft"
    for p in head:
        p.requires_grad = True


def train_mode(model, phase: str) -> None:
    model.train()
    if phase == "lp":
        model.backbone.eval()      # backbone congelato anche nei layer con comportamento train/eval


def make_phase_optimizer(model, hp: dict, phase: str, steps_per_epoch: int):
    back, head = _split_params(model)
    if phase == "lp":
        groups = [{"params": head, "lr": hp["lr_head_lp"]}]
        epochs = int(hp["lp_epochs"])
    else:
        groups = [{"params": back, "lr": hp["lr_backbone_ft"]}, {"params": head, "lr": hp["lr_head_ft"]}]
        epochs = int(hp["ft_epochs"])
    opt = torch.optim.AdamW(groups, weight_decay=hp["weight_decay"])
    total = max(1, epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=total,
                                                pct_start=float(hp["warmup_frac"]), anneal_strategy="cos")
    return opt, sched


def train_twophase(model: nn.Module, hp: dict, train_bank, val_bank, n_classes: int, device, seed: int,
                   ckpt_path: Path | None = None, guard=None, trial=None, log=print,
                   ckpt_every_min: float = 10.0) -> dict:
    hp = {**TWOPHASE_DEFAULTS, **hp}
    device = torch.device(device)
    model.to(device)
    use_amp = bool(hp.get("amp", True)) and device.type == "cuda"
    bs = int(hp["batch_size"])
    steps = math.ceil(train_bank.n / bs)
    crit = nn.CrossEntropyLoss(label_smoothing=float(hp["label_smoothing"]))
    select = hp.get("select_metric", "ba_subject_mean")
    patience = int(hp.get("patience", 10))
    needs_subj = getattr(model, "needs_subject", False)

    state = {"phase": "lp", "epoch": 0, "global_epoch": 0, "best_score": -1e9, "best_epoch": -1,
             "best_state": None, "history": [], "bad": 0, "phase_minutes": {"lp": 0.0, "ft": 0.0}}
    rng = np.random.default_rng(seed)
    resume = None
    if ckpt_path is not None and Path(ckpt_path).exists():
        resume = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        state.update(resume["state"])
        rng = resume["np_rng"]
        _set_rng_state(resume["rng"])
        log(f"  [ripresa] fase {state['phase']}, epoca {state['epoch']}")

    status = "completed"
    last_ckpt = time.time()
    phases = [p for p in ("lp", "ft") if int(hp[f"{p}_epochs"]) > 0]
    for phase in phases:
        if phase == "lp" and state["phase"] == "ft":
            continue                                  # ripresa a fase FT già iniziata
        if state["phase"] != phase:                   # transizione LP -> FT
            state.update(phase=phase, epoch=0, bad=0)
        set_phase(model, phase)
        opt, sched = make_phase_optimizer(model, hp, phase, steps)
        scaler = make_scaler(use_amp)
        if resume is not None and resume["state"]["phase"] == phase:
            opt.load_state_dict(resume["opt"])
            sched.load_state_dict(resume["sched"])
            scaler.load_state_dict(resume["scaler"])
            resume = None
        params = [p for g in opt.param_groups for p in g["params"]]
        if needs_subj and phase == "lp":              # backbone fermo: statistiche una volta sola
            compute_subject_stats(model, train_bank, device, amp=use_amp)
            compute_subject_stats(model, val_bank, device, amp=use_amp)

        def save_ckpt():
            if ckpt_path is None:
                return
            tmp = Path(str(ckpt_path) + ".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "state": state, "np_rng": rng, "rng": _rng_state()}, tmp)
            os.replace(tmp, ckpt_path)

        n_ep = int(hp[f"{phase}_epochs"])
        while state["epoch"] < n_ep:
            t0 = time.time()
            if needs_subj and phase == "ft":
                compute_subject_stats(model, train_bank, device, amp=use_amp)
            train_mode(model, phase)
            perm = rng.permutation(train_bank.n)
            losses = []
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                x, y = train_bank.get(b, device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    out = forward_model(model, x, batch_subjects(train_bank, b, device))
                loss = crit(out.float(), y)
                scaler.scale(loss).backward()
                if phase == "ft" and float(hp["clip_grad_ft"]) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, float(hp["clip_grad_ft"]))
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(loss.item())
            if not np.isfinite(np.mean(losses)):
                raise FloatingPointError(f"loss non finita (fase {phase}, epoca {state['epoch']})")
            if needs_subj and phase == "ft":
                compute_subject_stats(model, val_bank, device, amp=use_amp)
            v = validate(model, val_bank, device, use_amp, n_classes)
            score = v[select]
            sec = time.time() - t0
            state["phase_minutes"][phase] += sec / 60
            state["history"].append({"phase": phase, "epoch": state["epoch"], "global_epoch": state["global_epoch"],
                                     "loss": float(np.mean(losses)), **v, "sec": sec,
                                     "lr": [g["lr"] for g in opt.param_groups]})
            improved = False
            if phase == "ft":
                improved = score > state["best_score"]
                if improved:
                    state.update(best_score=score, best_epoch=state["global_epoch"], bad=0)
                    state["best_state"] = {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}
                else:
                    state["bad"] += 1
            log(f"  {phase} ep {state['epoch']:3d} loss {np.mean(losses):.4f} val BA {v['ba_subject_mean']:.4f} "
                f"kappa {v['kappa_pooled']:.4f} {'*' if improved else ''} ({sec:.0f}s)")
            if trial is not None:
                trial.report(v["ba_subject_mean"], state["global_epoch"])
            state["epoch"] += 1
            state["global_epoch"] += 1
            if trial is not None and trial.should_prune():
                status = "pruned"
                break
            if phase == "ft" and patience and state["bad"] >= patience:
                status = "early_stopped"
                break
            if guard is not None and guard.must_stop():
                save_ckpt()
                status = "interrupted"
                break
            if ckpt_path is not None and (time.time() - last_ckpt) / 60 >= ckpt_every_min:
                save_ckpt()
                last_ckpt = time.time()
        if status in ("pruned", "interrupted"):
            break
    if state["best_state"] is None:                    # nessuna epoca FT (es. ft_epochs=0 o potatura in LP)
        state["best_state"] = copy.deepcopy(model.state_dict())
        last = state["history"][-1] if state["history"] else {select: -1e9}
        state["best_score"] = float(last.get(select, -1e9))
        state["best_epoch"] = state["global_epoch"] - 1
    return {"best_state": state["best_state"], "best_epoch": state["best_epoch"],
            "best_score": float(state["best_score"]), "history": state["history"], "status": status,
            "phase_minutes": state["phase_minutes"],
            "phase_epochs": {p: sum(1 for h in state["history"] if h["phase"] == p) for p in ("lp", "ft")}}
