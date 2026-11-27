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
# Sessione 02: ricetta a due fasi (default prima di Optuna; Optuna sovrascrive)
_TWOPHASE = {"lp_epochs": 5, "ft_epochs": 20, "lr_head_lp": 1e-3, "lr_backbone_ft": 3e-5, "lr_head_ft": 3e-4,
             "warmup_frac": 0.1, "label_smoothing": 0.1, "weight_decay": 0.05, "batch_size": 64,
             "clip_grad_ft": 1.0, "patience": 10, "dropout": 0.1, "head": "pool_channel",
             "subj_embed_norm": "session", "select_metric": "ba_subject_mean"}
# REVE, ricetta ufficiale (dt.py + faced.yaml, commit 06a7059a; Appendice R §R6). Varianti A di §R8:
# dropout 0.0, GradScaler spento, niente LoRA.
_REVE_OFFICIAL = {"training_mode": "lp+ft", "lp_epochs": 20, "ft_epochs": 200, "lp_warmup_epochs": 3,
                  "ft_warmup_epochs": 5, "lr_lp": 5e-3, "lr_ft": 1e-4, "batch_size": 64, "micro_batch": 32, "patience": 10,
                  "plateau_patience": 6, "plateau_factor": 0.5, "mixup": True, "clip_grad": 2.0,
                  "weight_decay": 0.01, "betas": (0.92, 0.999), "eps": 1e-9, "dropout_lp": 0.0,
                  "dropout_ft": 0.0, "grad_scaler": False, "input_scale": 0.001,
                  "select_metric": "ba_pooled", "plateau_metric": "acc_pooled"}
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
        # Sessione 03, §5: feature residualizzate sulla clip (media della clip tolta entro training, val e test,
        # ciascuno dai propri soggetti, senza etichette; trasduttivo sul test): il modello non riapprende mu
        C("de_asym_lr_zsub_clipres", "features", "lr", window_s=10, features="de_asym", feature_zsub=True,
          feature_clipres=True, hparams={"clf": ["lr", "svm"]},
          notes="DE + z-score per soggetto, feature meno la media della clip (Sessione 03)"),
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
          notes="RINVIATO (sostituito dalle config reve_base* della Sessione 02)"),
        # --- Sessione 02, Parte A: CBraMod con fine-tuning a due fasi (LP poi FT), iperparametri da Optuna ----
        C("cbramod_twophase_pre", "neural", "cbramod", window_s=10, pretrained=True, recipe="twophase",
          hparams=dict(_TWOPHASE), hpo_study="cbramod_twophase",
          notes="LP-FT (Kumar et al. 2022), testa e standardizzazione per soggetto scelte in validation"),
        C("cbramod_twophase_scratch", "neural", "cbramod", window_s=10, pretrained=False, recipe="twophase",
          hparams=dict(_TWOPHASE), hpo_study="cbramod_twophase",
          notes="stessi iperparametri da inizializzazione casuale: la fase LP su backbone casuale tiene solo "
                "identico lo schedule"),
        # Ablazione A.4 (preparata, da lanciare solo se cbramod_twophase_pre supera il caso sulle clip nuove):
        # una modifica alla volta rispetto al migliore Optuna (le hparams qui sovrascrivono quelle di Optuna)
        C("cbramod_twophase_pre_flathead", "neural", "cbramod", window_s=10, pretrained=True, recipe="twophase",
          hparams={**_TWOPHASE, "head": "flat_official", "subj_embed_norm": "none",
                   "hpo_fixed": ["head", "subj_embed_norm"]}, hpo_study="cbramod_twophase",
          notes="A.4: testa ufficiale appiattita (se Optuna sceglie già flat_official, si usa la testa a pooling migliore: "
                "vedi queue_F_bis)"),
        C("cbramod_twophase_pre_nonorm", "neural", "cbramod", window_s=10, pretrained=True, recipe="twophase",
          hparams={**_TWOPHASE, "subj_embed_norm": "none", "hpo_fixed": ["subj_embed_norm"]},
          hpo_study="cbramod_twophase", notes="A.4"),
        C("cbramod_twophase_pre_bestpool", "neural", "cbramod", window_s=10, pretrained=True, recipe="twophase",
          hparams={**_TWOPHASE, "hpo_variant": "best_pool_head"}, hpo_study="cbramod_twophase",
          notes="A.4 se Optuna sceglie flat_official: migliore configurazione con testa a pooling"),
        C("cbramod_twophase_pre_heldoutnorm", "neural", "cbramod", window_s=10, pretrained=True, recipe="twophase",
          hparams={**_TWOPHASE, "subj_embed_norm": "heldout", "hpo_fixed": ["subj_embed_norm"]},
          hpo_study="cbramod_twophase", notes="A.4"),
        C("cbramod_official_poolhead", "neural", "cbramod", window_s=10, pretrained=True, recipe="cbramod_official",
          hparams={**_CBRAMOD_OFFICIAL, "hpo_keys": ["head", "subj_embed_norm", "dropout"]},
          hpo_study="cbramod_twophase",
          notes="A.4: ricetta ufficiale con la testa (e la standardizzazione) migliori di Optuna; gli altri "
                "iperparametri restano quelli ufficiali (resolve_config applica solo head/subj_embed_norm/dropout)"),
        # --- Sessione 02, Parte C: REVE-Base (Appendice R del prompt) -------------------------------------------
        C("reve_base", "neural", "reve", window_s=10, pretrained=True, recipe="reve_official",
          hparams=dict(_REVE_OFFICIAL), notes="REVE-Base, LP+FT con la ricetta ufficiale (dt.py, commit 06a7059a)"),
        C("reve_base_lp", "neural", "reve", window_s=10, pretrained=True, recipe="reve_official",
          hparams={**_REVE_OFFICIAL, "training_mode": "lp"}, notes="Gate R1: solo linear probing, senza pooling"),
        C("reve_base_gradscaler", "neural", "reve", window_s=10, pretrained=True, recipe="reve_official",
          hparams={**_REVE_OFFICIAL, "grad_scaler": True},
          notes="variante B di §R8.2 (GradScaler attivo): solo se il training fp16 è instabile, scelta in validation"),
        C("reve_base_yamldropout", "neural", "reve", window_s=10, pretrained=True, recipe="reve_official",
          hparams={**_REVE_OFFICIAL, "dropout_lp": 0.05, "dropout_ft": 0.1},
          notes="variante B di §R8.3 (dropout dello YAML): solo diagnostica, scelta in validation"),
        C("reve_base_scratch", "neural", "reve", window_s=10, pretrained=False, recipe="reve_official",
          hparams=dict(_REVE_OFFICIAL), notes="opzionale: REVE da inizializzazione casuale, ricetta ufficiale"),
        C("reve_base_lora16", "neural", "reve", window_s=10, pretrained=True, recipe="reve_official",
          hparams={**_REVE_OFFICIAL, "lora_rank": 16, "lora_alpha": 16, "training_mode": "lora"},
          notes="opzionale (§R8.4): LoRA rango 16 sull'attenzione, testa allenabile"),
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


def resolve_config(name: str, require_hpo: bool = False, params: dict | None = None) -> tuple[ExpConfig, str]:
    """Config con gli iperparametri Optuna applicati (se disponibili). Ritorna (cfg, sorgente_hp).
    `params`: parametri espliciti di un trial (candidati della riselezione su più seed), al posto del migliore."""
    import os
    require_hpo = require_hpo or os.environ.get("PIANOB_REQUIRE_HPO") == "1"
    cfg = get_config(name)
    if params is not None:
        from ..hpo.search_spaces import apply_params
        params = dict(params)
        if params.get("head") == "flat_official":
            params["subj_embed_norm"] = "none"
        return apply_params(cfg, params), "candidato esplicito"
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
    params = dict(best["params"])
    src = f"optuna:{cfg.hpo_study}:{best.get('n_complete', '?')}trial"
    if cfg.hp("hpo_variant") == "best_pool_head":
        # Sessione 02, A.4: migliore trial con testa a pooling (se Optuna sceglie la testa appiattita)
        pools = {h: v for h, v in best.get("best_by_head", {}).items() if h != "flat_official"}
        if not pools:
            raise RuntimeError("nessun trial completato con testa a pooling nello studio cbramod_twophase")
        h = max(pools, key=lambda k: pools[k]["value"])
        params = dict(pools[h]["params"])
        src += f":best_{h}"
    if cfg.hp("hpo_keys"):                 # solo alcuni parametri dallo studio (es. testa per la ricetta ufficiale)
        params = {k: v for k, v in params.items() if k in cfg.hp("hpo_keys")}
    fixed = set(cfg.hp("hpo_fixed", ()))   # parametri fissati dalla config (ablazioni: una modifica alla volta)
    params = {k: v for k, v in params.items() if k not in fixed}
    if params.get("head") == "flat_official" and "subj_embed_norm" not in params:
        params["subj_embed_norm"] = "none"
    cfg = apply_params(cfg, params)
    if cfg.hp("head") == "flat_official" and cfg.hp("subj_embed_norm", "none") != "none":
        cfg.hparams["subj_embed_norm"] = "none"   # la standardizzazione vale solo per le teste a pooling
    return cfg, src
