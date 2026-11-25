"""REVE-Base nella pipeline pianob (Sessione 02, Parte C). Vedi model.py (codice vendorizzato) e weights.py.

Ordine dei canali: il codice ufficiale di REVE (`configs/task/faced.yaml`) e quello di CBraMod da cui deriva
assegnano alle ultime due righe dei .pkl di FACED i nomi ['A2', 'A1']. Il nostro `faced_meta.CHANNELS` le
chiama ['A1', 'A2'] (ordine mai verificato). Per REVE l'ordine conta (le posizioni 3D entrano nel modello):
si usa l'ordine del codice di REVE, con cui è stato ottenuto 0.5646. `faced_meta.CHANNELS` NON si cambia,
per non modificare il comportamento delle config esistenti (TSception usa la coppia A1/A2 nel layout
emisferico): la corrispondenza nome-nostro -> nome-REVE è in REVE_NAME.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import REVE_BASE, ReveClassifier, apply_reve_lora

_POS_FILE = Path(__file__).with_name("positions_faced.json")
# indice del canale nel file .pkl -> nome secondo il codice REVE (faced.yaml, electrodes)
REVE_ELECTRODES = ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'CZ', 'C3', 'C4', 'T3',
                   'T4', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3', 'P4', 'T5', 'T6', 'PO3', 'PO4', 'OZ', 'O1', 'O2',
                   'A2', 'A1']


def _name_map():
    from ...data import faced_meta as FM
    assert len(FM.CHANNELS) == len(REVE_ELECTRODES)
    return dict(zip(FM.CHANNELS, REVE_ELECTRODES))


REVE_NAME = _name_map()


def positions_for(channel_names) -> torch.Tensor:
    """Posizioni 3D (metri) dei canali del job, dai nomi del nostro meta tramite REVE_NAME
    (posizioni copiate da hf/reve-positions/positions.json del repo ufficiale; T3=T7 e T5=P7 coincidono)."""
    from ...data.faced_meta import canonical_channel
    bank = json.loads(_POS_FILE.read_text())
    names = [REVE_NAME[canonical_channel(c)] for c in channel_names]
    pos = torch.tensor([bank[n] for n in names], dtype=torch.float32)
    assert pos.shape == (len(channel_names), 3), pos.shape
    return pos


def build_reve(cfg, n_channels: int, n_samples: int, n_classes: int, channel_names=None, synthetic: bool = False,
               log=print) -> ReveClassifier:
    from .weights import find_weights, load_into
    if channel_names is None:
        from ...data.faced_meta import CHANNELS
        channel_names = CHANNELS[:n_channels]
    assert len(channel_names) == n_channels
    pos = positions_for(channel_names)
    model = ReveClassifier(pos, n_samples, n_classes, dropout=cfg.hp("dropout_lp", 0.0),
                           input_scale=cfg.hp("input_scale", 0.001) / 0.01)   # loader: dati/100 -> REVE: dati/1000
    if cfg.pretrained:
        wp = find_weights()
        if wp is None:
            if synthetic:
                log("  [sintetico] pesi REVE assenti: uso pesi casuali")
            else:
                raise FileNotFoundError("Pesi REVE non trovati: esegui la cella di download del notebook G "
                                        "(Kaggle Secret HF_TOKEN, internet attivo)")
        else:
            model.pretrained_info = load_into(model, wp)
    model.variant_info = {"grad_scaler": bool(cfg.hp("grad_scaler", False)),
                          "dropout": [cfg.hp("dropout_lp", 0.0), cfg.hp("dropout_ft", 0.0)],
                          "lora": cfg.hp("lora_rank", 0) or None, "pretrained": model.pretrained_info is not None,
                          "channel_order": "REVE (A2, A1 nelle ultime due righe)", "seed_numpy": True,
                          "input_scale": "1/1000"}
    if cfg.hp("lora_rank", 0):
        apply_reve_lora(model, rank=cfg.hp("lora_rank"), alpha=cfg.hp("lora_alpha", None))
    return model
