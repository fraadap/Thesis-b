"""Training dei modelli neurali: AMP fp16 (T4/P100, niente bf16), selezione sulla validation per
soggetto, checkpoint a intervalli di tempo, ripresa dopo interruzione."""
from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..data.windows import WindowSet, gather_windows_torch, make_windows
from ..eval.metrics import evaluate_predictions, subject_mean


class TrialBank:
    """Trial (n_trials, C, T) in fp16, su GPU se ci stanno, + indice delle finestre."""

    def __init__(self, data: np.ndarray, subj, clip, label, win_len: int, overlap: float = 0.0,
                 device="cpu", max_gpu_gb: float = 4.0):
        assert data.ndim == 3 and data.shape[0] == len(subj)
        nbytes = data.size * 2
        dev = torch.device(device)
        if dev.type == "cuda" and nbytes / 2**30 > max_gpu_gb:
            dev = torch.device("cpu")
        self.data = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float16)).to(dev)
        if self.data.device.type == "cpu" and torch.cuda.is_available():
            self.data = self.data.pin_memory()
        self.ws: WindowSet = make_windows(np.asarray(subj), np.asarray(clip), np.asarray(label),
                                          data.shape[-1], win_len, overlap)
        self.win_label = torch.as_tensor(self.ws.label, dtype=torch.long)

    @property
    def n(self):
        return self.ws.n

    def get(self, idx, device) -> torch.Tensor:
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        x = gather_windows_torch(self.data, torch.as_tensor(self.ws.win_trial[idx]),
                                 torch.as_tensor(self.ws.win_start[idx]), self.ws.win_len)
        return x.to(device, non_blocking=True).float(), self.win_label[idx_t].to(device)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def class_weights(labels, n_classes: int) -> torch.Tensor:
    """omega_c proporzionale a 1/n_c, normalizzati a media 1."""
    cnt = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    w = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), 0.0)
    w = w / w[cnt > 0].mean()
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(model, bank: TrialBank, device, batch_size: int = 128, amp: bool = True,
            idx=None, return_embed: bool = False):
    model.eval()
    idx = np.arange(bank.n) if idx is None else np.asarray(idx)
    probs, embs = [], []
    use_amp = amp and torch.device(device).type == "cuda"
    for i in range(0, len(idx), batch_size):
        x, _ = bank.get(idx[i:i + batch_size], device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            if return_embed:
                e = model.embed(x)
                embs.append(e.float().cpu())
                out = _head_from_embed(model, x, e)
            else:
                out = model(x)
        probs.append(torch.softmax(out.float(), dim=1).cpu())
    P = torch.cat(probs).numpy()
    return (P, torch.cat(embs).numpy()) if return_embed else P


def _head_from_embed(model, x, e):
    return model(x)


def build_optimizer(model, hp: dict, recipe: str, steps_per_epoch: int):
    params_head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    params_back = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    wd = hp["weight_decay"]
    if recipe == "cbramod_official":
        # finetune_trainer.py: backbone lr, resto 0.001*(bs/256)**0.5, AdamW, cosine per step, eta_min 1e-6
        groups = [{"params": params_back, "lr": hp["lr"]},
                  {"params": params_head, "lr": 0.001 * (hp["batch_size"] / 256) ** 0.5}] \
            if hp.get("multi_lr", True) else [{"params": params_back + params_head, "lr": hp["lr"]}]
    elif hp.get("lora_rank"):
        groups = [{"params": params_head, "lr": hp.get("lr_head", hp["lr"])},
                  {"params": params_back, "lr": hp.get("lr_lora", hp["lr"])}]
    else:
        groups = [{"params": params_back + params_head, "lr": hp["lr"]}]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    total = max(1, hp["epochs"] * steps_per_epoch)
    warm = int(round(float(hp.get("lr_warmup_epochs", 0)) * steps_per_epoch))
    if hp.get("scheduler", "cosine_step") == "cosine_step" and warm > 0:
        # warm-up lineare delle LR (variante diagnostica per CBraMod da zero), poi coseno come la ricetta
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warm),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total - warm), eta_min=1e-6)],
            milestones=[warm])
    elif hp.get("scheduler", "cosine_step") == "cosine_step":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-6)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    return opt, sched


def validate(model, bank, device, amp, n_classes):
    P = predict(model, bank, device, amp=amp)
    ev = evaluate_predictions(P, bank.ws.subject, bank.ws.clip, bank.ws.label)
    return {"ba_subject_mean": subject_mean(ev["per_subject"], "ba_win"),
            "ba_trial_subject_mean": subject_mean(ev["per_subject"], "ba_trial"),
            "kappa_pooled": ev["pooled"]["kappa_win"], "ba_pooled": ev["pooled"]["ba_win"]}


def _rng_state():
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def train_model(model: nn.Module, hp: dict, recipe: str, train_bank: TrialBank, val_bank: TrialBank,
                n_classes: int, device, seed: int, ckpt_path: Path | None = None, guard=None,
                trial=None, log=print, ckpt_every_min: float = 10.0) -> dict:
    """Ritorna {'best_state', 'best_epoch', 'best_score', 'history', 'status'} con status in
    {'completed', 'early_stopped', 'interrupted', 'pruned'}."""
    device = torch.device(device)
    model.to(device)
    use_amp = bool(hp.get("amp", True)) and device.type == "cuda"
    bs = int(hp["batch_size"])
    steps = math.ceil(train_bank.n / bs)
    opt, sched = build_optimizer(model, hp, recipe, steps)
    scaler = make_scaler(use_amp)
    if recipe == "cbramod_official" or not hp.get("class_weighted", True):
        crit = nn.CrossEntropyLoss(label_smoothing=hp["label_smoothing"])
    else:
        crit = nn.CrossEntropyLoss(weight=class_weights(train_bank.ws.label, n_classes).to(device),
                                   label_smoothing=hp["label_smoothing"])
    select = hp.get("select_metric", "ba_subject_mean")
    patience = int(hp.get("patience", 0))
    warm = int(hp.get("warmup_head_epochs", 0))
    clip = float(hp.get("clip_grad", 0))

    start_epoch, best_score, best_epoch, best_state, history, bad = 0, -1e9, -1, None, [], 0
    rng = np.random.default_rng(seed)
    if ckpt_path is not None and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_score, best_epoch = ck["epoch"] + 1, ck["best_score"], ck["best_epoch"]
        best_state, history, bad = ck["best_state"], ck["history"], ck["bad"]
        rng = ck["np_rng"]
        _set_rng_state(ck["rng"])
        log(f"  [ripresa] dal checkpoint, epoca {start_epoch}")

    def save_ckpt(epoch):
        if ckpt_path is None:
            return
        tmp = Path(str(ckpt_path) + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_score": best_score,
                    "best_epoch": best_epoch, "best_state": best_state, "history": history, "bad": bad,
                    "np_rng": rng, "rng": _rng_state()}, tmp)
        os.replace(tmp, ckpt_path)

    last_ckpt = time.time()
    status = "completed"
    lora_params = {id(p) for n, p in model.named_parameters() if ".parametrizations." in n}
    for epoch in range(start_epoch, int(hp["epochs"])):
        t0 = time.time()
        model.train()
        perm = rng.permutation(train_bank.n)
        losses = []
        for i in range(0, len(perm), bs):
            x, y = train_bank.get(perm[i:i + bs], device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(x)
            loss = crit(out.float(), y)
            scaler.scale(loss).backward()
            if epoch < warm:  # warm-up: solo la testa (i parametri senza grad sono saltati da AdamW)
                for g in opt.param_groups:
                    for p in g["params"]:
                        if id(p) in lora_params:
                            p.grad = None
            if clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], clip)
            scaler.step(opt)
            scaler.update()
            if hasattr(model, "apply_constraints"):   # es. max-norm di EEGNet
                model.apply_constraints()
            sched.step()
            losses.append(loss.item())
        if not np.isfinite(np.mean(losses)):
            raise FloatingPointError(f"loss non finita all'epoca {epoch}: dati o training numericamente instabili")
        v = validate(model, val_bank, device, use_amp, n_classes)
        score = v[select]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **v, "sec": time.time() - t0})
        improved = score > best_score
        if improved:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}
        else:
            bad += 1
        log(f"  ep {epoch:3d} loss {np.mean(losses):.4f} val BA {v['ba_subject_mean']:.4f} "
            f"kappa {v['kappa_pooled']:.4f} {'*' if improved else ''} ({time.time() - t0:.0f}s)")
        if trial is not None:
            trial.report(v["ba_subject_mean"], epoch)
            if trial.should_prune():
                status = "pruned"
                break
        if patience and bad >= patience:
            status = "early_stopped"
            break
        if guard is not None and guard.must_stop():
            save_ckpt(epoch)
            status = "interrupted"
            break
        if ckpt_path is not None and (time.time() - last_ckpt) / 60 >= ckpt_every_min:
            save_ckpt(epoch)
            last_ckpt = time.time()
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    return {"best_state": best_state, "best_epoch": best_epoch, "best_score": float(best_score),
            "history": history, "status": status}
