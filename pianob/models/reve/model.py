"""REVE-Base (El Ouahidi et al., brain-bzh) vendorizzato senza dipendenze da transformers/einops/flash-attn.

Fonte: https://github.com/elouayas/reve_eeg, commit 06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7 (licenza MIT,
`LICENSE_REVE` in questa cartella). Moduli copiati da `src/models/encoder.py` (classe REVE),
`src/models/backbone.py` e `src/models/classifier.py` (ReveClassifier con pooling "no"), con gli stessi nomi
dei sottomoduli, così lo state_dict del checkpoint Hugging Face `brain-bzh/reve-base` si carica con strict=True.

Differenze dichiarate rispetto al codice ufficiale:
- attenzione sempre con `F.scaled_dot_product_attention` (il codice usa flash-attn se installato, altrimenti
  SDPA; matematicamente equivalenti);
- `einops.rearrange` sostituito da reshape/permute equivalenti;
- il rumore sulle posizioni non modifica il tensore in ingresso (nel codice `pos += rumore` è in place);
- la scala d'ingresso è dentro il modello (`input_scale`): il nostro loader restituisce dati/100 (convenzione
  CBraMod), REVE vuole dati/1000, quindi il modello moltiplica per 0.1.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REVE_BASE = {"embed_dim": 512, "depth": 22, "heads": 8, "head_dim": 64, "mlp_dim_ratio": 2.66, "use_geglu": True,
             "freqs": 4, "noise_ratio": 0.0025, "patch_size": 200, "patch_overlap": 20}


# ------------------------------------------------------------------ backbone (src/models/backbone.py)
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, geglu):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, hidden_dim * 2 if geglu else hidden_dim, bias=False),
                                 GEGLU() if geglu else nn.GELU(), nn.Linear(hidden_dim, dim, bias=False))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, head_dim=64):
        super().__init__()
        inner = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        b, n, _ = q.shape
        q, k, v = (t.reshape(b, n, self.heads, -1).transpose(1, 2) for t in (q, k, v))   # b h n d
        out = F.scaled_dot_product_attention(q, k, v)
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))


class TransformerBackbone(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, geglu):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([nn.ModuleList([Attention(dim, heads, head_dim), FeedForward(dim, mlp_dim, geglu)])
                                     for _ in range(depth)])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ------------------------------------------------------------------ encoder (src/models/encoder.py)
class FourierEmb4D(nn.Module):
    def __init__(self, dimension: int, freqs: int, increment_time=0.1, margin: float = 0.4):
        super().__init__()
        self.dimension, self.freqs, self.increment_time, self.margin = dimension, freqs, increment_time, margin

    def forward(self, positions_):
        positions = positions_.clone()
        positions[:, :, -1] *= self.increment_time
        *U, _ = positions.shape
        freqs_w = torch.arange(self.freqs).to(positions)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        p_z = 2 * math.pi * freqs_z / width
        p_w = 2 * math.pi * freqs_w / width
        positions = positions[..., None, None, None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y + positions[..., 2] * p_z
               + positions[..., 3] * p_w).view(*U, -1)
        if self.dimension != 512:
            _, _, hd = loc.shape
            loc = loc[:, :, :-(hd - self.dimension // 2)]
        return torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)

    @staticmethod
    def add_time_patch(pos, num_patches):
        B, C, _ = pos.shape
        pos_rep = pos.unsqueeze(2).repeat(1, 1, num_patches, 1)
        t = torch.arange(0, num_patches, 1, device=pos.device).float().view(1, 1, num_patches, 1).expand(B, C, num_patches, 1)
        return torch.cat((pos_rep, t), dim=-1).view(B, C * num_patches, 4)


class ReveEncoder(nn.Module):
    def __init__(self, embed_dim=512, depth=22, heads=8, head_dim=64, mlp_dim_ratio=2.66, use_geglu=True,
                 freqs=4, patch_size=200, patch_overlap=20, noise_ratio=0.0025, **_):
        super().__init__()
        self.transformer = TransformerBackbone(embed_dim, depth, heads, head_dim, int(embed_dim * mlp_dim_ratio),
                                               use_geglu)
        self.embed_dim, self.freqs = embed_dim, freqs
        self.patch_size, self.overlap_size, self.noise_ratio = patch_size, patch_overlap, noise_ratio
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_size, embed_dim))
        self.fourier4d = FourierEmb4D(embed_dim, freqs=freqs)
        self.mlp4d = nn.Sequential(nn.Linear(4, embed_dim, bias=False), nn.GELU(), nn.LayerNorm(embed_dim))
        self.ln = nn.LayerNorm(embed_dim)

    def n_patches(self, n_samples: int) -> int:
        """Formula di `get_flattened_output_dim` (src/utils/model_utils.py): +1 se la divisione è esatta."""
        step = self.patch_size - self.overlap_size
        n = math.ceil((n_samples - self.patch_size) / step)
        return n + 1 if (n_samples - self.patch_size) % step == 0 else n

    def forward(self, eeg, pos):
        eeg = eeg.float()
        patches = eeg.unfold(dimension=2, size=self.patch_size, step=self.patch_size - self.overlap_size)
        b, c, h, _ = patches.shape
        if self.training:   # rumore sulle posizioni, unico per canale, condiviso nel batch, generato con numpy
            noise = np.random.normal(loc=0, scale=self.noise_ratio, size=(c, 3))
            pos = pos + torch.from_numpy(noise).to(pos)
        pos = FourierEmb4D.add_time_patch(pos, h)
        pos_embed = self.ln(self.fourier4d(pos) + self.mlp4d(pos))
        x = self.to_patch_embedding(patches).reshape(b, c * h, self.embed_dim) + pos_embed
        return self.transformer(x)


# ------------------------------------------------------------------ classificatore (src/models/classifier.py)
class ReveClassifier(nn.Module):
    """ReveClassifier con pooling "no": query `cls` -> contesto, cat([contesto, token]) -> flatten ->
    RMSNorm -> Dropout -> Linear. Le posizioni (C, 3) sono un buffer fisso del job (canali del montaggio)."""

    extra_embeddings = ("token_mean",)

    def __init__(self, positions: torch.Tensor, n_samples: int, n_classes: int, dropout: float = 0.0,
                 input_scale: float = 0.1, arch: dict | None = None):
        super().__init__()
        arch = {**REVE_BASE, **(arch or {})}
        assert positions.dim() == 2 and positions.shape[1] == 3, positions.shape
        self.encoder = ReveEncoder(**arch)
        E = self.encoder.embed_dim
        self.cls_query_token = nn.Parameter(torch.randn(1, 1, E))
        self.register_buffer("positions", positions.float().clone(), persistent=False)
        C = positions.shape[0]
        self.n_tokens = C * self.encoder.n_patches(n_samples)
        self.out_shape = (self.n_tokens + 1) * E
        self.linear_head = nn.Sequential(nn.Flatten(1), RMSNorm(self.out_shape), nn.Dropout(dropout),
                                         nn.Linear(self.out_shape, n_classes))
        self.input_scale = float(input_scale)
        self.embed_mode = None
        self.pretrained_info = None
        self.variant_info = None

    # ---- fasi (src/utils/model_utils.py: freeze_model / unfreeze_model)
    def set_phase(self, phase: str) -> None:
        if phase == "lp":
            for p in self.parameters():
                p.requires_grad = False
            for p in self.linear_head.parameters():
                p.requires_grad = True
            self.cls_query_token.requires_grad = True
        elif phase == "ft":
            for p in self.parameters():
                p.requires_grad = True
        elif phase == "lora":          # §R8.4: solo LoRA + testa + query
            for n, p in self.named_parameters():
                p.requires_grad = (".parametrizations." in n and (n.endswith(".A") or n.endswith(".B"))) \
                    or n.startswith("linear_head.") or n == "cls_query_token"
        else:
            raise ValueError(phase)

    def set_dropout(self, p: float) -> None:
        self.linear_head[2].p = float(p)

    # ---- forward
    def tokens(self, x):
        pos = self.positions.unsqueeze(0).expand(x.shape[0], -1, -1)
        return self.encoder(x * self.input_scale, pos)                       # (B, C*h, E)

    def _context(self, x):
        b = x.shape[0]
        q = self.cls_query_token.expand(b, -1, -1)
        w = torch.softmax(torch.matmul(q, x.transpose(-1, -2)) / (self.encoder.embed_dim ** 0.5), dim=-1)
        return torch.matmul(w, x)                                            # (B, 1, E)

    def forward(self, x):
        t = self.tokens(x)
        return self.linear_head(torch.cat([self._context(t), t], dim=-2))

    def embed(self, x):
        """Sonda della clip: contesto della query cls (512) o, con embed_mode='token_mean', media dei token."""
        t = self.tokens(x)
        if self.embed_mode == "token_mean":
            return t.mean(dim=1)
        return self._context(t).squeeze(1)


def apply_reve_lora(model: ReveClassifier, rank: int = 16, alpha: float | None = None) -> list[str]:
    """LoRA rango 16 sulle proiezioni dell'attenzione (to_qkv, to_out), come `get_lora_config` del codice
    legacy (apply_to=attention; alpha = rango). Con peft to_qkv è un unico Linear: un solo blocco."""
    from torch.nn.utils import parametrize

    from ..lora import LoRADelta
    alpha = float(alpha if alpha is not None else rank)
    done = []
    for i, layer in enumerate(model.encoder.transformer.layers):
        for name in ("to_qkv", "to_out"):
            lin = getattr(layer[0], name)
            W = lin.weight
            parametrize.register_parametrization(lin, "weight", LoRADelta(W.shape[0], W.shape[1], rank, alpha))
            done.append(f"encoder.transformer.layers.{i}.0.{name}.weight")
    return done
