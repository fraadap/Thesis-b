"""CBraMod (Wang et al., ICLR 2025): backbone vendorizzato + teste di classificazione + loader dei pesi.

Le teste replicano `models/model_for_faced.py` del repo ufficiale senza einops
(`Rearrange('b c s d -> b (c s d)')` == `reshape(b, -1)` su tensore contiguo).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from .cbramod import CBraMod

PATCH = 200  # campioni per patch (1 s a 200 Hz)


class SubjectEmbedNorm(nn.Module):
    """z-score per soggetto di ogni dimensione dell'embedding (ricetta a due fasi, Sessione 02).

    Statistiche calcolate SENZA etichette sulle finestre del soggetto (`compute_subject_stats` in
    train/loop.py) e tenute in una cache indicizzata dall'ID del soggetto (buffer, quindi salvate nello
    state_dict). mode: none | session (tutte le finestre del soggetto, trasduttivo come EA) | heldout
    (solo le clip fuori dal fold di test: il calcolo lo restringe chi chiama, con una maschera)."""

    def __init__(self, dim: int, mode: str = "none", n_slots: int = 128, eps: float = 1e-5):
        super().__init__()
        assert mode in ("none", "session", "heldout"), mode
        self.mode, self.eps = mode, eps
        self.register_buffer("mean", torch.zeros(n_slots, dim))
        self.register_buffer("std", torch.ones(n_slots, dim))
        self.register_buffer("valid", torch.zeros(n_slots, dtype=torch.bool))

    @torch.no_grad()
    def set_stats(self, subjects, mean: torch.Tensor, std: torch.Tensor) -> None:
        idx = torch.as_tensor(list(subjects), dtype=torch.long, device=self.mean.device)
        self.mean[idx] = mean.to(self.mean)
        self.std[idx] = std.to(self.std)
        self.valid[idx] = True

    def forward(self, e: torch.Tensor, subj: torch.Tensor | None) -> torch.Tensor:
        if self.mode == "none":
            return e
        assert subj is not None, "subj_embed_norm richiede gli ID dei soggetti nel batch"
        subj = subj.to(self.mean.device).long()
        if not bool(self.valid[subj].all()):
            raise RuntimeError(f"statistiche dell'embedding mancanti per i soggetti {sorted(set(subj[~self.valid[subj]].tolist()))}")
        return (e - self.mean[subj].to(e.dtype)) / (self.std[subj].to(e.dtype) + self.eps)


class CBraModClassifier(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int, head: str = "all_patch_reps",
                 dropout: float = 0.1, pretrained_path: str | Path | None = None, n_layer: int = 12,
                 subj_embed_norm: str = "none"):
        super().__init__()
        assert n_samples % PATCH == 0, f"la finestra deve essere multipla di {PATCH} campioni"
        self.n_patches = n_samples // PATCH
        self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                                seq_len=30, n_layer=n_layer, nhead=8)
        self.pretrained_info = None
        if pretrained_path is not None:
            self.pretrained_info = load_pretrained(self.backbone, pretrained_path)
        self.backbone.proj_out = nn.Identity()
        C, S, D = n_channels, self.n_patches, 200
        self.head_type = head
        self.norm = None
        if head in ("pool_channel", "pool_global"):
            # Sessione 02: media sui patch (per canale, oppure su patch e canali) -> MLP compatto
            E = C * D if head == "pool_channel" else D
            self.embed_dim = E
            self.norm = SubjectEmbedNorm(E, subj_embed_norm)
            self.classifier = nn.Sequential(nn.Linear(E, 256), nn.ELU(), nn.Dropout(dropout), nn.Linear(256, n_classes))
        elif subj_embed_norm not in ("none", None):
            raise ValueError("subj_embed_norm vale solo per le teste a pooling (pool_channel, pool_global)")
        elif head in ("all_patch_reps", "flat_official"):   # flat_official = nome della Sessione 02
            self.classifier = nn.Sequential(
                nn.Flatten(1), nn.Linear(C * S * D, S * D), nn.ELU(), nn.Dropout(dropout),
                nn.Linear(S * D, D), nn.ELU(), nn.Dropout(dropout), nn.Linear(D, n_classes))
        elif head == "all_patch_reps_onelayer":
            self.classifier = nn.Sequential(nn.Flatten(1), nn.Linear(C * S * D, n_classes))
        elif head == "avgpooling_patch_reps":
            self.classifier = _AvgPoolHead(D, n_classes)
        elif head == "pooled_mlp":  # testa leggera per LoRA: media sulle patch, poi MLP sui canali
            self.classifier = _PooledMLPHead(C, D, n_classes, dropout)
        else:
            raise ValueError(head)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        return self.backbone(x.reshape(B, C, T // PATCH, PATCH))      # (B, C, S, D)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).mean(dim=(1, 2))

    @property
    def needs_subject(self) -> bool:
        return self.norm is not None and self.norm.mode != "none"

    def pooled_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding delle teste a pooling PRIMA della standardizzazione per soggetto."""
        f = self.features(x)                                   # (B, C, S, D)
        if self.head_type == "pool_channel":
            return f.mean(dim=2).flatten(1)                    # (B, C*D)
        return f.mean(dim=(1, 2))                              # (B, D)

    def forward(self, x: torch.Tensor, subj: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm is not None:
            return self.classifier(self.norm(self.pooled_embedding(x), subj))
        return self.classifier(self.features(x))


class _AvgPoolHead(nn.Module):
    def __init__(self, D, n_classes):
        super().__init__()
        self.fc = nn.Linear(D, n_classes)

    def forward(self, f):
        return self.fc(f.mean(dim=(1, 2)))


class _PooledMLPHead(nn.Module):
    def __init__(self, C, D, n_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(1), nn.Linear(C * D, 256), nn.ELU(), nn.Dropout(dropout),
                                 nn.Linear(256, n_classes))

    def forward(self, f):
        return self.net(f.mean(dim=2))


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pretrained(backbone: nn.Module, path) -> dict:
    """Carica i pesi pre-addestrati (strict). Accetta lo state_dict ufficiale
    (`pretrained_weights.pth`) e wrapper comuni ('state_dict', 'model', prefissi 'module.'/'backbone.')."""
    path = Path(path)
    assert path.exists(), f"pesi CBraMod non trovati: {path}"
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch vecchio
        sd = torch.load(path, map_location="cpu")
    for k in ("state_dict", "model", "model_state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    for pref in ("module.", "backbone.", "model."):
        if all(k.startswith(pref) for k in sd):
            sd = {k[len(pref):]: v for k, v in sd.items()}
    own = backbone.state_dict()
    missing = sorted(set(own) - set(sd))
    unexpected = sorted(set(sd) - set(own))
    if missing or unexpected:
        raise RuntimeError(
            "I pesi non corrispondono al backbone CBraMod ufficiale.\n"
            f"mancanti ({len(missing)}): {missing[:10]}\ninattesi ({len(unexpected)}): {unexpected[:10]}\n"
            "Se hai scaricato il re-hosting braindecode, usa invece i pesi del repo ufficiale "
            "(huggingface.co/weighting666/CBraMod, file pretrained_weights.pth).")
    backbone.load_state_dict(sd, strict=True)
    return {"path": str(path), "sha256": sha256(path), "n_tensors": len(sd)}
