"""Spazi di ricerca Optuna (§8) e applicazione dei parametri alla config."""
from __future__ import annotations

import copy

N_TRIALS = {"tsception_ea": 60, "eegnet_ea": 40, "cbramod_lora_r2": 30, "cbramod_ft_official_lr": 10}


def sample(study: str, trial) -> dict:
    if study == "tsception_ea":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.6),
            "num_T": trial.suggest_categorical("num_T", [9, 15, 21]),
            "num_S": trial.suggest_categorical("num_S", [6, 9, 15]),
            "window_s": trial.suggest_categorical("window_s", [2, 4, 6]),
            "label_smoothing": trial.suggest_categorical("label_smoothing", [0.0, 0.1, 0.2]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
    if study == "eegnet_ea":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.6),
            "F1": trial.suggest_categorical("F1", [8, 16]),
            "D": trial.suggest_categorical("D", [2, 4]),
            "window_s": trial.suggest_categorical("window_s", [2, 4]),
        }
    if study == "cbramod_lora_r2":
        return {
            "lr_head": trial.suggest_float("lr_head", 3e-4, 3e-3, log=True),
            "lr_lora": trial.suggest_float("lr_lora", 1e-4, 2e-3, log=True),
            "lora_rank": trial.suggest_categorical("lora_rank", [1, 2, 4, 8]),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.3),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
            "label_smoothing": trial.suggest_categorical("label_smoothing", [0.0, 0.1, 0.2]),
            "warmup_head_epochs": trial.suggest_categorical("warmup_head_epochs", [0, 3, 5]),
            # aggiunta rispetto al §8: la testa ufficiale ha ~128M parametri; quella leggera rende LoRA
            # davvero efficiente. Scelta lasciata alla validation.
            "head": trial.suggest_categorical("head", ["all_patch_reps", "pooled_mlp"]),
        }
    if study == "cbramod_ft_official_lr":
        return {"lr": trial.suggest_categorical("lr", [5e-5, 1e-4, 2e-4]),
                "multi_lr": trial.suggest_categorical("multi_lr", [True, False])}
    raise KeyError(study)


CFG_FIELDS = {"window_s"}


def apply_params(cfg, params: dict):
    cfg = copy.deepcopy(cfg)
    for k, v in params.items():
        if k in CFG_FIELDS:
            setattr(cfg, k, v)
        else:
            cfg.hparams[k] = v
    if "lora_rank" in params:
        cfg.hparams["lora_alpha"] = params["lora_rank"]
    return cfg
