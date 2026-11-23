"""Nomi delle configurazioni -> ExpConfig. Gli iperparametri trovati da Optuna sovrascrivono i default."""
from __future__ import annotations

import json
from pathlib import Path

from .. import kaggle_env
from ..config import ExpConfig

_CBRAMOD_OFFICIAL = {  # finetune_main.py (repo ufficiale), verificato
    "epochs": 50, "batch_size": 64, "lr": 1e-4, "weight_decay": 5e-2, "clip_grad": 1.0, "dropout": 0.1,
    "head": "all_patch_reps", "label_smoothing": 0.1, "multi_lr": True, "class_weighted": False,
    "select_metric": "kappa_pooled", "patience": 0, "scheduler": "cosine_step",
}
_LORA = {
    "epochs": 50, "batch_size": 64, "lora_rank": 2, "lora_alpha": 2, "lora_dropout": 0.0,
    "lr_head": 1e-3, "lr_lora": 5e-4, "lr": 5e-4, "weight_decay": 1e-2, "label_smoothing": 0.1,
    "warmup_head_epochs": 5, "head": "all_patch_reps", "dropout": 0.1, "patience": 10, "clip_grad": 1.0,
}
_TSC = {"epochs": 50, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-2, "dropout": 0.5, "num_T": 15,
        "num_S": 15, "hidden": 32, "label_smoothing": 0.1, "patience": 10}
_EEGNET = {"epochs": 50, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-3, "dropout": 0.5, "F1": 8, "D": 2,
           "label_smoothing": 0.1, "patience": 10}


def _all() -> dict[str, ExpConfig]:
    C = ExpConfig
    cfgs = [
        C("const_majority", "const", notes="livello di caso effettivo"),
        C("logpower5_lr", "features", "lr", window_s=10, features="logpower5",
          hparams={"clf": ["lr"]}, notes="baseline spettrale minima"),
        C("de_asym_lr", "features", "lr", window_s=10, features="de_asym",
          hparams={"clf": ["lr", "svm"]}, notes="DE + DASM + RASM, LR o SVM lineare scelto in validation"),
        C("de_asym_lr_zsub", "features", "lr", window_s=10, features="de_asym", feature_zsub=True,
          hparams={"clf": ["lr", "svm"]}, notes="z-score delle feature per soggetto (trasduttivo)"),
        C("eegnet_ea", "neural", "eegnet", window_s=4, overlap_train=0.5, align="ea", hparams=dict(_EEGNET),
          hpo_study="eegnet_ea", tta_bnalpha=True),
        C("eegnet_noalign", "neural", "eegnet", window_s=4, overlap_train=0.5, align="none",
          hparams=dict(_EEGNET), hpo_study="eegnet_ea", notes="ablazione: EEGNet senza EA"),
        C("tsception_ea", "neural", "tsception", window_s=4, overlap_train=0.5, align="ea",
          channel_layout="hemispheric", hparams=dict(_TSC), hpo_study="tsception_ea", tta_bnalpha=True,
          notes="produce anche le righe tsception_ea_bnalpha (TTA a fine training)"),
        C("tsception_noalign", "neural", "tsception", window_s=4, overlap_train=0.5, align="none",
          channel_layout="hemispheric", hparams=dict(_TSC), hpo_study="tsception_ea",
          notes="ablazione: TSception senza EA"),
        C("cbramod_ft_official", "neural", "cbramod", window_s=10, pretrained=True, recipe="cbramod_official",
          hparams=dict(_CBRAMOD_OFFICIAL), notes="Gate 1: ricetta ufficiale, full fine-tuning"),
        C("cbramod_scratch_ft", "neural", "cbramod", window_s=10, pretrained=False, recipe="cbramod_official",
          hparams=dict(_CBRAMOD_OFFICIAL), notes="init casuale, stessa ricetta (cfr. issue #22 del repo)"),
        # Diagnostica dell'init casuale (la ricetta esatta collassa sulla classe più frequente, cfr. issue #22):
        # una sola modifica alla volta rispetto a cbramod_scratch_ft.
        C("cbramod_scratch_ft_definit", "neural", "cbramod", window_s=10, pretrained=False, recipe="cbramod_official",
          hparams={**_CBRAMOD_OFFICIAL, "init": "default"}, notes="init casuale di default PyTorch"),
        C("cbramod_scratch_ft_warmup", "neural", "cbramod", window_s=10, pretrained=False, recipe="cbramod_official",
          hparams={**_CBRAMOD_OFFICIAL, "lr_warmup_epochs": 3}, notes="warm-up lineare delle LR per 3 epoche"),
        C("cbramod_scratch_ft_definit_warmup", "neural", "cbramod", window_s=10, pretrained=False,
          recipe="cbramod_official", hparams={**_CBRAMOD_OFFICIAL, "init": "default", "lr_warmup_epochs": 3},
          notes="entrambe le modifiche"),
        C("cbramod_lora_r2", "neural", "cbramod", window_s=10, pretrained=True, hparams=dict(_LORA),
          hpo_study="cbramod_lora_r2"),
        C("cbramod_lora_r2_ea", "neural", "cbramod", window_s=10, pretrained=True, align="ea",
          align_rescale=True, hparams=dict(_LORA), hpo_study="cbramod_lora_r2"),
        C("cbramod_lora_r2_zsub", "neural", "cbramod", window_s=10, pretrained=True, align="zscore",
          align_rescale=True, hparams=dict(_LORA), hpo_study="cbramod_lora_r2"),
        C("reve_base_lora", "neural", "reve", window_s=10, pretrained=True,
          notes="RINVIATO: 69M parametri, codice remoto, accesso al checkpoint da verificare"),
    ]
    return {c.name: c for c in cfgs}


REGISTRY = _all()


def get_config(name: str) -> ExpConfig:
    if name not in REGISTRY:
        raise KeyError(f"config sconosciuta: {name}. Disponibili: {sorted(REGISTRY)}")
    import copy
    return copy.deepcopy(REGISTRY[name])


def find_hpo_best(study: str) -> Path | None:
    """Ordine: stato corrente -> output di sessioni precedenti (input) -> file nel codice."""
    local = kaggle_env.state_dir() / "hpo_best" / f"{study}.json"
    if local.exists():
        return local
    for p in kaggle_env.find_all(f"{study}.json"):
        if p.parent.name == "hpo_best":
            return p
    pkg = Path(__file__).resolve().parents[1] / "hpo" / "best" / f"{study}.json"
    return pkg if pkg.exists() else None


def resolve_config(name: str, require_hpo: bool = False) -> tuple[ExpConfig, str]:
    """Config con gli iperparametri Optuna applicati (se disponibili). Ritorna (cfg, sorgente_hp)."""
    import os
    require_hpo = require_hpo or os.environ.get("PIANOB_REQUIRE_HPO") == "1"
    cfg = get_config(name)
    if not cfg.hpo_study:
        return cfg, "registry"
    p = find_hpo_best(cfg.hpo_study)
    if p is None:
        if require_hpo:
            raise RuntimeError(f"iperparametri Optuna '{cfg.hpo_study}' non trovati")
        print(f"[attenzione] {name}: nessun risultato Optuna '{cfg.hpo_study}', uso i default del registro")
        return cfg, "defaults"
    best = json.loads(p.read_text())
    from ..hpo.search_spaces import apply_params
    cfg = apply_params(cfg, best["params"])
    return cfg, f"optuna:{cfg.hpo_study}:{best.get('n_complete', '?')}trial"
