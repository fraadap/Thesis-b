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


class CBraModClassifier(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int, head: str = "all_patch_reps",
                 dropout: float = 0.1, pretrained_path: str | Path | None = None, n_layer: int = 12):
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
        if head == "all_patch_reps":
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
