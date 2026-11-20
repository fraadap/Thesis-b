"""LoRA (Hu et al. 2021) via torch.nn.utils.parametrize, così funziona anche con
nn.MultiheadAttention (che usa `in_proj_weight` impacchettato Q|K|V direttamente).

W' = W + (alpha / r) * B @ A, con B inizializzata a zero (uscita identica al modello base).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class LoRADelta(nn.Module):
    """Parametrizzazione di un peso (out, in) con `n_blocks` blocchi lungo out (3 per Q|K|V)."""

    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float,
                 dropout: float = 0.0, n_blocks: int = 1):
        super().__init__()
        assert out_features % n_blocks == 0
        self.n_blocks, self.rank = n_blocks, rank
        self.scale = alpha / rank
        bo = out_features // n_blocks
        self.A = nn.Parameter(torch.empty(n_blocks, rank, in_features))
        self.B = nn.Parameter(torch.zeros(n_blocks, bo, rank))
        for i in range(n_blocks):
            nn.init.kaiming_uniform_(self.A[i], a=math.sqrt(5))
        # dropout sui fattori (variante "weight-space": l'input non è accessibile alla parametrizzazione)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        delta = torch.bmm(self.B, self.drop(self.A)).reshape(W.shape)
        return W + self.scale * delta.to(W.dtype)


def apply_lora(model: nn.Module, rank: int = 2, alpha: float | None = None, dropout: float = 0.0,
               targets=("attn_qkv", "attn_out", "ffn"), root_name: str = "backbone") -> list[str]:
    """Aggiunge LoRA ai pesi del transformer di `model.<root_name>`. Ritorna i nomi dei pesi adattati."""
    alpha = float(alpha if alpha is not None else rank)
    root = getattr(model, root_name)
    done = []
    for name, mod in root.named_modules():
        if isinstance(mod, nn.MultiheadAttention):
            if "attn_qkv" in targets:
                W = mod.in_proj_weight
                parametrize.register_parametrization(
                    mod, "in_proj_weight", LoRADelta(W.shape[0], W.shape[1], rank, alpha, dropout, 3))
                done.append(f"{name}.in_proj_weight")
            if "attn_out" in targets:
                W = mod.out_proj.weight
                parametrize.register_parametrization(
                    mod.out_proj, "weight", LoRADelta(W.shape[0], W.shape[1], rank, alpha, dropout))
                done.append(f"{name}.out_proj.weight")
        elif isinstance(mod, nn.Linear) and "ffn" in targets and name.split(".")[-1] in ("linear1", "linear2"):
            W = mod.weight
            parametrize.register_parametrization(mod, "weight", LoRADelta(W.shape[0], W.shape[1], rank, alpha, dropout))
            done.append(f"{name}.weight")
    return done


def mark_trainable_lora(model: nn.Module, head_name: str = "classifier") -> None:
    """Solo LoRA (A, B) e la testa restano allenabili."""
    for n, p in model.named_parameters():
        p.requires_grad = (".parametrizations." in n and (n.endswith(".A") or n.endswith(".B"))) \
            or n.startswith(head_name + ".")


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if ".parametrizations." in n and (n.endswith(".A") or n.endswith(".B"))]


def count_trainable(model: nn.Module) -> dict:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for p in lora_parameters(model))
    return {"total": tot, "trainable": tr, "lora": lora}
