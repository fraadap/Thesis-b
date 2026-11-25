"""StableAdamW per la ricetta ufficiale di REVE (Sessione 02, Parte C).

Porting di `src/utils/stable_adamw.py` del repo REVE (commit 06a7059a; a sua volta modificato da optimi di
Benjamin Warner, licenza MIT). Stessa matematica del ramo foreach ufficiale, scritta per tensore senza API
private di PyTorch (`_group_tensors_by_device_and_dtype`, `_default_to_fused_or_foreach`), che cambiano tra
versioni e potrebbero mancare nell'immagine Kaggle. Parametri in fp32: la somma di Kahan (solo per pesi
fp16/bf16) non si attiva, come nel codice ufficiale con pesi fp32.

Aggiornamento per parametro, al passo t (beta debiasati):
  m <- lerp(m, g, 1 - b1hat);  v <- b2hat * v + (1 - b2hat) * g^2
  lr_p = lr / max(1, sqrt(mean(g^2 / max(v, eps^2))))          (stabilizzazione di tipo Adafactor)
  p <- p * (1 - lr_p * wd)                                      (weight decay disaccoppiato)
  p <- p - lr_p * m / (sqrt(v) + eps)
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def debias_beta(beta: float, step: int) -> float:
    return (beta ** step - beta) / (beta ** step - 1)


class StableAdamW(Optimizer):
    def __init__(self, params, lr: float, betas=(0.9, 0.99), weight_decay: float = 1e-2, eps: float = 1e-6,
                 decouple_lr: bool = False, max_lr: float | None = None):
        if not 0.0 <= lr:
            raise ValueError(f"lr non valido: {lr}")
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps, "weight_decay": weight_decay,
                    "decouple_lr": decouple_lr, "max_lr": max_lr if max_lr is not None else (lr if decouple_lr else None)}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            group["step"] = group.get("step", 0) + 1          # contatore per gruppo, come l'originale
            t = group["step"]
            beta1_comp = 1 - debias_beta(group["beta1"], t)
            beta2_hat = debias_beta(group["beta2"], t)
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            for p in params:
                g = p.grad
                st = self.state[p]
                if "exp_avg" not in st:
                    st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["eps_sq"] = torch.tensor(eps ** 2, dtype=p.dtype, device=p.device)
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.lerp_(g, beta1_comp)
                v.mul_(beta2_hat).addcmul_(g, g, value=1 - beta2_hat)
                r = g.pow(2).div_(torch.maximum(v, st["eps_sq"]))
                neg_lr = -lr / max(1.0, r.mean().sqrt().item())
                if wd != 0:
                    if group["decouple_lr"]:
                        p.mul_(1 + (neg_lr / group["max_lr"]) * wd)
                    else:
                        p.mul_(1 + neg_lr * wd)
                p.addcdiv_(m, v.sqrt().add_(eps), value=neg_lr)
        return loss
