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
# Lawhern et al. 2018 (vedi le config eegnet_lawhern*)
_LAWHERN = {"F1": 8, "D": 2, "dropout": 0.25, "max_norm": True, "epochs": 200, "batch_size": 64, "lr": 1e-3,
            "weight_decay": 0.0, "label_smoothing": 0.0, "class_weighted": False, "scheduler": "constant",
            "clip_grad": 0.0, "select_metric": "ba_subject_mean", "patience": 50}
# TSception con la metodologia originale (Ding et al., IEEE TAC 2022; repo deepBrains/TSception): num_T = num_S = 15,
# hidden 32, dropout 0.5, Adam lr 1e-3 senza weight decay, batch 64, label smoothing 0.1, LR costante, molte
# epoche con early stopping in validation. [DA VERIFICARE sul repo: numero massimo di epoche, patience e smoothing.]
# Stesso budget di EEGNet originale (200 epoche, patience 50) per confrontarli alla pari.
_TSC_DING = {"num_T": 15, "num_S": 15, "hidden": 32, "dropout": 0.5, "epochs": 200, "batch_size": 64, "lr": 1e-3,
             "weight_decay": 0.0, "label_smoothing": 0.1, "class_weighted": False, "scheduler": "constant",
             "clip_grad": 0.0, "select_metric": "ba_subject_mean", "patience": 50}
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
        # --- EEGNet con la metodologia originale (notebook E) -------------------------------------------------
        # Lawhern et al. 2018: F1=8, D=2, F2=16, kernel temporale fs/2, separabile fs/8, dropout 0.25 (cross-
        # soggetto), max-norm (1.0 sui filtri spaziali, 0.25 sul classificatore), Adam lr 1e-3 senza weight decay,
        # batch 64, niente label smoothing né pesi di classe, molte epoche con il modello migliore in validation.
        # Protocollo del benchmark CBraMod: finestre da 10 s a 200 Hz, scala /100, nessun allineamento.
        C("eegnet_lawhern", "neural", "eegnet", window_s=10, align="none", hparams=dict(_LAWHERN),
          notes="EEGNet originale (Lawhern 2018) nel protocollo CBraMod"),
        C("eegnet_lawhern_ea", "neural", "eegnet", window_s=10, align="ea", hparams=dict(_LAWHERN),
          notes="come eegnet_lawhern + allineamento EA per soggetto"),
        # Verifiche dopo il notebook E (0.455): una modifica alla volta rispetto a eegnet_lawhern_ea.
        C("eegnet_lawhern_ea_w4", "neural", "eegnet", window_s=4, align="ea", hparams=dict(_LAWHERN),
          notes="finestre da 4 s (come eegnet_ea): il salto 0.229 -> 0.455 viene dalla finestra o dalla ricetta?"),
        C("eegnet_lawhern_ea_long", "neural", "eegnet", window_s=10, align="ea",
          hparams={**_LAWHERN, "epochs": 400},
          notes="400 epoche: con 200 la validation di 2 seed su 3 migliora ancora all'ultima epoca"),
        C("eegnet_cbramod_recipe", "neural", "eegnet", window_s=10, align="none", recipe="cbramod_official",
          hparams={**_CBRAMOD_OFFICIAL, "F1": 8, "D": 2, "dropout": 0.25, "max_norm": False},
          notes="EEGNet addestrato con il framework di fine-tuning CBraMod (come le baseline del paper) "
                "[DA VERIFICARE: la ricetta esatta delle baseline CBraMod non è nel repo]"),
        C("tsception_ea", "neural", "tsception", window_s=4, overlap_train=0.5, align="ea",
          channel_layout="hemispheric", hparams=dict(_TSC), hpo_study="tsception_ea", tta_bnalpha=True,
          notes="produce anche le righe tsception_ea_bnalpha (TTA a fine training)"),
        C("tsception_noalign", "neural", "tsception", window_s=4, overlap_train=0.5, align="none",
          channel_layout="hemispheric", hparams=dict(_TSC), hpo_study="tsception_ea",
          notes="ablazione: TSception senza EA"),
        # --- TSception con la metodologia originale nel protocollo CBraMod (finestre da 10 s), notebook E -----------
        C("tsception_ding", "neural", "tsception", window_s=10, align="none", channel_layout="hemispheric",
          hparams=dict(_TSC_DING), notes="TSception originale (Ding 2022) nel protocollo CBraMod"),
        C("tsception_ding_ea", "neural", "tsception", window_s=10, align="ea", channel_layout="hemispheric",
          hparams=dict(_TSC_DING), notes="come tsception_ding + allineamento EA per soggetto"),
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
        C("cbramod_ft_official_warmup", "neural", "cbramod", window_s=10, pretrained=True, recipe="cbramod_official",
          hparams={**_CBRAMOD_OFFICIAL, "lr_warmup_epochs": 3},
          notes="pre-addestrato con la stessa modifica del da-zero funzionante: confronto H1 a ricetta identica"),
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
