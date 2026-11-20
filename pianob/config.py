"""Config degli esperimenti (dataclass serializzabili in JSON)."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class ExpConfig:
    name: str
    family: str                     # "const" | "features" | "neural"
    model: str = ""                 # eegnet | tsception | cbramod | lr | svm
    window_s: float = 10.0
    overlap_train: float = 0.0      # sovrapposizione SOLO in training
    align: str = "none"             # none | ea | zscore   (per soggetto, label-free)
    align_ref: str = "euclid"       # euclid | riemann
    align_rescale: bool = False     # dopo EA/z-score riporta l'RMS a quello dei dati grezzi (per i FM)
    features: str = ""              # logpower5 | de_asym
    feature_zsub: bool = False      # z-score delle feature per soggetto
    channel_layout: str = "all"     # all | hemispheric (TSception)
    pretrained: bool = False
    recipe: str = "default"         # default | cbramod_official
    hparams: dict = field(default_factory=dict)
    hpo_study: str = ""             # nome dello studio Optuna da cui prendere gli iperparametri
    tta_bnalpha: bool = False       # valuta anche BN-α (offline/online) a fine training
    seeds_dev: int = 3
    seeds_final: int = 5
    notes: str = ""

    def hp(self, key, default=None):
        return self.hparams.get(key, default)

    def with_hparams(self, **kw) -> "ExpConfig":
        c = copy.deepcopy(self)
        c.hparams.update(kw)
        return c

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ExpConfig":
        return ExpConfig(**d)

    def digest(self) -> str:
        return hashlib.md5(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:8]


# Iperparametri di default del training neurale (sovrascritti da registry / Optuna).
TRAIN_DEFAULTS = {
    "epochs": 50,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "label_smoothing": 0.1,
    "class_weighted": True,
    "clip_grad": 1.0,
    "patience": 10,                 # early stopping sulla validation (0 = nessuno)
    "select_metric": "ba_subject_mean",   # oppure kappa_pooled (ricetta ufficiale CBraMod)
    "scheduler": "cosine_step",     # cosine per step con eta_min=1e-6 (come CBraMod)
    "multi_lr": False,
    "warmup_head_epochs": 0,        # epoche iniziali con solo la testa allenabile (LoRA)
    "amp": True,
}
