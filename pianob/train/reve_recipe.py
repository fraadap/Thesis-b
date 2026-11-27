"""Ricetta ufficiale di REVE (`src/dt.py`, commit 06a7059a), portata nella pipeline pianob (Appendice R §R6).

- Due fasi nella stessa esecuzione: LP (encoder congelato; si allenano testa e query cls), poi FT (tutto
  scongelato) che parte dagli ULTIMI pesi di LP (dt.py non ricarica model_best.pth).
- Per fase: StableAdamW nuovo (betas 0.92/0.999, eps 1e-9, wd 0.01, nessun gruppo), warm-up esponenziale per
  step lr = lr_max * (10^(step/total) - 1) / 9 (primo step a lr 0), ReduceLROnPlateau (max, fattore 0.5,
  patience 6) con `step(val_acc)` una volta per epoca solo per epoch > warmup_epochs.
- Mixup: un solo lambda ~ U(0, 1) per batch (random.random()) e una sola permutazione;
  loss = lambda * CE(y) + (1 - lambda) * CE(y[perm]). Cross-entropy senza pesi né label smoothing.
- Batch 64 diviso in micro-batch (`micro_batch`, default 32) con accumulo del gradiente: su T4 (14.5 GB) il
  fine-tuning completo con batch 64 esaurisce la memoria (notebook G, 2026-11-27). Il mixup si calcola sul batch
  intero (un lambda, una permutazione) e la loss di ogni micro-batch è pesata per la sua frazione del batch:
  stesso gradiente del batch intero (il modello non ha BatchNorm e il dropout è 0).
- Gradient clipping a norma 2.0 su tutti i parametri; autocast fp16; GradScaler spento (variante A,
  fedele al codice: la chiave torch_dtype era letta al livello sbagliato) o acceso (variante B).
- Selezione ed early stopping sulla BA di validation pooled (balanced_accuracy_score su tutte le finestre):
  si aggiorna il migliore solo se migliora in senso stretto, altrimenti patience += 1; stop quando
  patience > 10 (11 epoche consecutive senza miglioramento); il conteggio riparte a ogni fase.
- Modello migliore = epoca con la BA di validation più alta della fase FT (di LP se training_mode='lp').
"""
from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .loop import _rng_state, _set_rng_state, make_scaler, validate
from .stable_adamw import StableAdamW


def exponential_warmup(total_steps: int):
    def f(step):
        return (10 ** (step / total_steps) - 1) / 9 if step < total_steps else 1.0
    return f


def phases_for(mode: str) -> list[str]:
    return {"lp": ["lp"], "ft": ["ft"], "lp+ft": ["lp", "ft"], "lora": ["lp", "lora"]}[mode]


def train_reve_official(model: nn.Module, hp: dict, train_bank, val_bank, n_classes: int, device, seed: int,
                        ckpt_path: Path | None = None, guard=None, trial=None, log=print,
                        ckpt_every_min: float = 10.0) -> dict:
    device = torch.device(device)
    model.to(device)
    use_amp = device.type == "cuda"
    bs = int(hp["batch_size"])
    micro = int(hp.get("micro_batch") or bs)
    steps = math.ceil(train_bank.n / bs)
    crit = nn.CrossEntropyLoss()
    select = hp.get("select_metric", "ba_pooled")
    plateau_metric = hp.get("plateau_metric", "acc_pooled")
    random.seed(seed)                                # lambda del mixup (dt.py: random.random())
    rng = np.random.default_rng(seed)                # ordine dei batch (shuffle seminato)
    phases = phases_for(hp.get("training_mode", "lp+ft"))
    state = {"phase": phases[0], "epoch": 0, "global_epoch": 0, "best_val": 0.0, "best_epoch": -1,
             "best_state": None, "patience": 0, "history": [], "phase_minutes": {p: 0.0 for p in phases},
             "phase_best": {}}
    resume = None
    if ckpt_path is not None and Path(ckpt_path).exists():
        resume = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        state.update(resume["state"])
        rng = resume["np_rng"]
        random.setstate(resume["py_rng"])
        _set_rng_state(resume["rng"])
        log(f"  [ripresa] fase {state['phase']}, epoca {state['epoch']}")

    status = "completed"
    last_ckpt = time.time()
    for phase in phases:
        if phases.index(phase) < phases.index(state["phase"]):
            continue
        if state["phase"] != phase:                      # nuova fase: best e patience ripartono (dt.py)
            state.update(phase=phase, epoch=0, best_val=0.0, patience=0, best_state=None)
        ph = "ft" if phase == "lora" else phase
        model.set_phase(phase)
        model.set_dropout(hp.get(f"dropout_{ph}", 0.0))
        params = [p for p in model.parameters() if p.requires_grad]
        opt = StableAdamW(params, lr=hp[f"lr_{ph}"], betas=tuple(hp["betas"]), weight_decay=hp["weight_decay"],
                          eps=hp["eps"])
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=hp["plateau_factor"],
                                                             patience=hp["plateau_patience"])
        warm_ep = int(hp[f"{ph}_warmup_epochs"])
        warm = torch.optim.lr_scheduler.LambdaLR(opt, exponential_warmup(steps * warm_ep)) if warm_ep > 0 else None
        scaler = make_scaler(use_amp and bool(hp.get("grad_scaler", False)))
        if resume is not None and resume["state"]["phase"] == phase:
            opt.load_state_dict(resume["opt"])
            plateau.load_state_dict(resume["plateau"])
            if warm is not None and resume.get("warm") is not None:
                warm.load_state_dict(resume["warm"])
            scaler.load_state_dict(resume["scaler"])
            resume = None

        def save_ckpt():
            if ckpt_path is None:
                return
            tmp = Path(str(ckpt_path) + ".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "plateau": plateau.state_dict(),
                        "warm": warm.state_dict() if warm is not None else None, "scaler": scaler.state_dict(),
                        "state": state, "np_rng": rng, "py_rng": random.getstate(), "rng": _rng_state()}, tmp)
            os.replace(tmp, ckpt_path)

        n_ep = int(hp[f"{ph}_epochs"])
        while state["epoch"] < n_ep:
            t0 = time.time()
            epoch = state["epoch"]
            model.train()
            perm = rng.permutation(train_bank.n)
            losses = []
            for i in range(0, len(perm), bs):
                x, y = train_bank.get(perm[i:i + bs], device)
                opt.zero_grad(set_to_none=True)
                if hp.get("mixup", True):
                    mm = random.random()
                    pm = torch.randperm(x.shape[0], device=x.device)
                    x, y_b = mm * x + (1 - mm) * x[pm], y[pm]
                else:
                    mm, y_b = 1.0, y
                n, loss_sum = x.shape[0], 0.0
                for c in range(0, n, micro):
                    xc, ya, yb = x[c:c + micro], y[c:c + micro], y_b[c:c + micro]
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        out = model(xc).float()
                        lc = mm * crit(out, ya) + (1 - mm) * crit(out, yb) if mm != 1.0 else crit(out, ya)
                    lc = lc * (xc.shape[0] / n)
                    scaler.scale(lc).backward()
                    loss_sum += float(lc.detach())
                    del out, lc
                loss = torch.tensor(loss_sum)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(hp["clip_grad"]))
                scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
                skipped = scaler.is_enabled() and scale != scaler.get_scale()
                if warm is not None and epoch < warm_ep and not skipped:
                    warm.step()
            if not np.isfinite(np.mean(losses)):
                raise FloatingPointError(f"loss non finita (fase {phase}, epoca {epoch}): prova la variante "
                                         "con GradScaler (reve_base_gradscaler), scelta in validation")
            v = validate(model, val_bank, device, use_amp, n_classes)
            sec = time.time() - t0
            state["phase_minutes"][phase] += sec / 60
            improved = state["best_val"] < v[select]
            if improved:
                state["best_val"], state["best_epoch"], state["patience"] = v[select], state["global_epoch"], 0
                state["best_state"] = {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}
                state["phase_best"][phase] = {"epoch": epoch, "val": v[select]}
            else:
                state["patience"] += 1
            state["history"].append({"phase": phase, "epoch": epoch, "global_epoch": state["global_epoch"],
                                     "loss": float(np.mean(losses)), **v, "sec": sec,
                                     "lr": opt.param_groups[0]["lr"], "patience": state["patience"]})
            log(f"  {phase} ep {epoch:3d} loss {np.mean(losses):.4f} val BA {v[select]:.4f} "
                f"acc {v[plateau_metric]:.4f} lr {opt.param_groups[0]['lr']:.2e} {'*' if improved else ''} ({sec:.0f}s)")
            if epoch > warm_ep:
                plateau.step(v[plateau_metric])
            state["epoch"] += 1
            state["global_epoch"] += 1
            if state["patience"] > int(hp["patience"]):
                log(f"  fase {phase} terminata per patience")
                break
            if guard is not None and guard.must_stop():
                save_ckpt()
                status = "interrupted"
                break
            if ckpt_path is not None and (time.time() - last_ckpt) / 60 >= ckpt_every_min:
                save_ckpt()
                last_ckpt = time.time()
        if status == "interrupted":
            break
    if state["best_state"] is None:
        state["best_state"] = {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}
    return {"best_state": state["best_state"], "best_epoch": state["best_epoch"], "best_score": float(state["best_val"]),
            "history": state["history"], "status": status if status == "interrupted" else "completed",
            "phase_minutes": state["phase_minutes"], "phase_best": state["phase_best"],
            "phase_epochs": {p: sum(1 for h in state["history"] if h["phase"] == p) for p in phases}}
