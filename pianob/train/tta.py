"""Test-time adaptation BN-α (solo modelli con BatchNorm).

mu_test = a_n * mu_hat_s + (1 - a_n) * mu_src,  a_n = n / (n + n0)   (idem per la varianza)

Regimi:
- offline 'session': statistiche da tutte le finestre del soggetto di test
- online 'first{n}': statistiche dalle prime n finestre (ordine naturale delle clip), valutazione su tutte
- online 'stream'  : causale, la finestra i usa le statistiche delle finestre 0..i
- n0 = 0: ristima pura (AdaBN), controllo
Hook del prior (Proposta 3): `prior_fn(subject) -> lista di (mean, var)` sostituisce mu_src.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def bn_layers(model) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, BN_TYPES)]


def has_bn(model) -> bool:
    return len(bn_layers(model)) > 0


def get_stats(model):
    return [(m.running_mean.detach().clone(), m.running_var.detach().clone()) for m in bn_layers(model)]


def set_stats(model, stats):
    for m, (mu, var) in zip(bn_layers(model), stats):
        m.running_mean.copy_(mu)
        m.running_var.copy_(var)


def mix_stats(target, source, alpha: float):
    return [(alpha * mt + (1 - alpha) * ms, alpha * vt + (1 - alpha) * vs)
            for (mt, vt), (ms, vs) in zip(target, source)]


@torch.no_grad()
def estimate_stats(model, x: torch.Tensor, batch: int = 32):
    """Statistiche BN dei dati x (n, C, T): media cumulativa delle statistiche per batch,
    BN in modalità train (momentum=None), dropout disattivato. Il modello non viene modificato."""
    m = copy.deepcopy(model)
    m.eval()
    for bn in bn_layers(m):
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    for i in range(0, len(x), batch):
        m(x[i:i + batch])
    return get_stats(m)


@torch.no_grad()
def cumulative_window_stats(model, x: torch.Tensor) -> list:
    """Per ogni i: media delle statistiche BN delle finestre 0..i (ciascuna stimata da sola)."""
    m = copy.deepcopy(model)
    m.eval()
    bns = bn_layers(m)
    for bn in bns:
        bn.momentum = None
        bn.train()
    out, acc = [], None
    for i in range(len(x)):
        for bn in bns:
            bn.reset_running_stats()
        m(x[i:i + 1])
        st = get_stats(m)
        acc = st if acc is None else [((ma * i + mi) / (i + 1), (va * i + vi) / (i + 1))
                                      for (ma, va), (mi, vi) in zip(acc, st)]
        out.append(acc)
    return out


@torch.no_grad()
def collect_subject_bn_stats(model, bank, device, batch: int = 64) -> dict:
    """Hook: statistiche BN per soggetto di training, salvate a fine training (Proposta 3)."""
    out = {}
    subj = bank.ws.subject
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]
        x, _ = bank.get(idx, device)
        out[int(s)] = [(a.cpu(), b.cpu()) for a, b in estimate_stats(model, x, batch)]
    return out


@torch.no_grad()
def bnalpha_predict(model, bank, device, n0_list=(0, 8, 32, 128), first_n=(1, 8, 32, 128),
                    stream: bool = True, prior_fn=None, subjects=None) -> dict:
    """Ritorna {variante: probs (n_win, K)} allineate alle finestre del bank."""
    model = model.to(device).eval()
    src = get_stats(model)
    subj = bank.ws.subject
    subjects = np.unique(subj) if subjects is None else subjects
    K = None
    out: dict[str, np.ndarray] = {}

    def put(name, idx, probs):
        nonlocal K
        K = probs.shape[1]
        if name not in out:
            out[name] = np.full((bank.n, K), np.nan, dtype=np.float32)
        out[name][idx] = probs

    work = copy.deepcopy(model)
    for s in subjects:
        idx = np.where(subj == s)[0]          # già nell'ordine naturale (clip, tempo)
        x, _ = bank.get(idx, device)
        prior = prior_fn(int(s)) if prior_fn is not None else src
        prior = [(a.to(device), b.to(device)) for a, b in prior]
        full = estimate_stats(model, x)
        for n0 in n0_list:
            a = len(idx) / (len(idx) + n0)
            set_stats(work, mix_stats(full, prior, a))
            put(f"bnalpha_session_n0{n0}", idx, torch.softmax(work(x).float(), 1).cpu().numpy())
            for n in first_n:
                n_eff = min(n, len(idx))
                st = estimate_stats(model, x[:n_eff])
                set_stats(work, mix_stats(st, prior, n_eff / (n_eff + n0)))
                put(f"bnalpha_first{n}_n0{n0}", idx, torch.softmax(work(x).float(), 1).cpu().numpy())
            if stream:
                if n0 == n0_list[0]:
                    cum = cumulative_window_stats(model, x)
                probs = []
                for i in range(len(idx)):
                    set_stats(work, mix_stats(cum[i], prior, (i + 1) / (i + 1 + n0)))
                    probs.append(torch.softmax(work(x[i:i + 1]).float(), 1).cpu().numpy())
                put(f"bnalpha_stream_n0{n0}", idx, np.concatenate(probs))
    set_stats(work, src)
    return out
