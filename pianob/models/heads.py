"""Costruzione dei modelli a partire dalla config + stub REVE."""
from __future__ import annotations

import torch.nn as nn


def build_model(cfg, n_channels: int, n_samples: int, n_classes: int, fs: int = 200,
                pretrained_path=None) -> nn.Module:
    m = cfg.model
    if m == "eegnet":
        return EEGNet_(cfg, n_channels, n_samples, n_classes, fs)
    if m == "tsception":
        from .tsception import TSception
        return TSception(n_channels, n_samples, n_classes, fs, num_T=cfg.hp("num_T", 15),
                         num_S=cfg.hp("num_S", 15), hidden=cfg.hp("hidden", 32), dropout=cfg.hp("dropout", 0.5))
    if m == "cbramod":
        from .cbramod import CBraModClassifier
        model = CBraModClassifier(n_channels, n_samples, n_classes, head=cfg.hp("head", "all_patch_reps"),
                                  dropout=cfg.hp("dropout", 0.1),
                                  pretrained_path=pretrained_path if cfg.pretrained else None,
                                  n_layer=cfg.hp("n_layer", 12),
                                  subj_embed_norm=cfg.hp("subj_embed_norm", "none"))
        if not cfg.pretrained and cfg.hp("init", "official") == "default":
            # variante diagnostica: init di default di PyTorch al posto di kaiming_normal(fan_out) del codice
            # ufficiale, che con la ricetta ufficiale fa divergere/collassare il training da zero
            for mod in model.backbone.modules():
                if isinstance(mod, nn.Linear):
                    mod.reset_parameters()
        if cfg.hp("lora_rank", 0):
            from .lora import apply_lora, mark_trainable_lora
            apply_lora(model, rank=cfg.hp("lora_rank"), alpha=cfg.hp("lora_alpha", None),
                       dropout=cfg.hp("lora_dropout", 0.0))
            mark_trainable_lora(model)
        return model
    if m == "reve":
        raise NotImplementedError(
            "REVE-Base rinviato: 69M parametri, codice remoto (trust_remote_code), accesso al checkpoint "
            "da verificare. Stub predisposto in models/heads.py.")
    raise ValueError(m)


def EEGNet_(cfg, n_channels, n_samples, n_classes, fs):
    from .eegnet import EEGNet
    return EEGNet(n_channels, n_samples, n_classes, fs, F1=cfg.hp("F1", 8), D=cfg.hp("D", 2),
                  dropout=cfg.hp("dropout", 0.5), max_norm=bool(cfg.hp("max_norm", False)))
