"""Protocolli di valutazione deterministici con asserzioni anti-leakage.

- split_fm      : train sub000-079, val 080-099, test 100-122; tutte le clip ovunque (stimoli condivisi)
- split_fm_lco  : stessi soggetti x 3 fold di clip; test su clip viste (baseline appaiata) e non viste
- lnso10        : 10 fold sequenziali sui soggetti (12 per fold, 15 nell'ultimo), val = 10% del train
- lnso10_lco    : lnso10 x 3 fold di clip (fold = 3 * fold_soggetti + fold_clip)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import faced_meta as FM

PROTOCOLS = ("split_fm", "split_fm_lco", "lnso10", "lnso10_lco")
N_CLIP_FOLDS = 3
N_SUBJ_FOLDS = 10


@dataclass
class Split:
    protocol: str
    fold: int
    train_subjects: list[int]
    val_subjects: list[int]
    test_subjects: list[int]
    train_clips: list[int]           # clip usate in training E in validation
    unseen_clips: list[int] = field(default_factory=list)  # vuoto se non leave-clip-out
    shared_stimuli: bool = True      # True se i soggetti di test vedono clip presenti in training

    @property
    def is_lco(self) -> bool:
        return len(self.unseen_clips) > 0

    def eval_conditions(self) -> dict[str, list[int]]:
        """Condizioni di test: 'all' (split senza LCO) oppure 'seen' / 'unseen'."""
        if self.is_lco:
            return {"seen": list(self.train_clips), "unseen": list(self.unseen_clips)}
        return {"all": list(self.train_clips)}

    def to_dict(self):
        return asdict(self)


def n_folds(protocol: str) -> int:
    return {"split_fm": 1, "split_fm_lco": N_CLIP_FOLDS, "lnso10": N_SUBJ_FOLDS,
            "lnso10_lco": N_SUBJ_FOLDS * N_CLIP_FOLDS}[protocol]


def clip_fold_assignment(clip_labels=FM.CLIP_LABELS, n_folds: int = N_CLIP_FOLDS) -> np.ndarray:
    """Fold di clip: la k-esima clip di ogni categoria va nel fold k; con 4 clip (neutre) l'indice i
    va nel fold i % 3 (ripartizione 2/1/1)."""
    idx = FM.clip_index_in_category(clip_labels)
    return np.array([i % n_folds for i in idx])


def _lnso_folds(subject_ids: list[int], n: int = N_SUBJ_FOLDS) -> list[list[int]]:
    """Partizione sequenziale: n_per = floor(N/n); l'ultimo fold prende il resto (123 -> 9x12 + 15).
    Stesso schema del codice CLISA (THU-EP) degli autori FACED; per FACED: [DA VERIFICARE] sulla
    partizione esatta usata da Xu & Plataniotis."""
    ids = sorted(subject_ids)
    n_per = len(ids) // n
    return [ids[k * n_per:(k + 1) * n_per] if k < n - 1 else ids[k * n_per:] for k in range(n)]


def make_split(protocol: str, fold: int = 0, task_clips: list[int] | None = None,
               subjects_present: list[int] | None = None, clip_labels=FM.CLIP_LABELS,
               n_subjects: int = FM.N_SUBJECTS) -> Split:
    task_clips = list(range(len(clip_labels))) if task_clips is None else sorted(task_clips)
    all_ids = list(range(n_subjects))
    present = set(all_ids if subjects_present is None else subjects_present)
    assert 0 <= fold < n_folds(protocol), f"fold {fold} fuori range per {protocol}"

    if protocol in ("split_fm", "split_fm_lco"):
        assert n_subjects == FM.N_SUBJECTS, "split_fm è definito solo per i 123 soggetti di FACED"
        tr, va, te = all_ids[:80], all_ids[80:100], all_ids[100:]
        clip_fold = fold if protocol == "split_fm_lco" else None
    else:
        sf = fold // N_CLIP_FOLDS if protocol == "lnso10_lco" else fold
        folds = _lnso_folds(all_ids)
        te = folds[sf]
        rest = [s for s in all_ids if s not in te]
        rng = np.random.default_rng(7919 + sf)
        n_val = int(round(0.1 * len(rest)))
        va = sorted(rng.choice(rest, size=n_val, replace=False).tolist())
        tr = [s for s in rest if s not in va]
        clip_fold = fold % N_CLIP_FOLDS if protocol == "lnso10_lco" else None

    tr, va, te = ([s for s in x if s in present] for x in (tr, va, te))
    if clip_fold is None:
        train_clips, unseen = task_clips, []
    else:
        cf = clip_fold_assignment(clip_labels)
        unseen = [c for c in task_clips if cf[c] == clip_fold]
        train_clips = [c for c in task_clips if cf[c] != clip_fold]
    sp = Split(protocol, fold, tr, va, te, train_clips, unseen, shared_stimuli=True)
    check_split(sp, clip_labels)
    return sp


def check_split(sp: Split, clip_labels=FM.CLIP_LABELS) -> None:
    tr, va, te = set(sp.train_subjects), set(sp.val_subjects), set(sp.test_subjects)
    assert tr and te, "split vuoto"
    assert not (tr & va), f"LEAKAGE soggetti train/val: {sorted(tr & va)}"
    assert not (tr & te), f"LEAKAGE soggetti train/test: {sorted(tr & te)}"
    assert not (va & te), f"LEAKAGE soggetti val/test: {sorted(va & te)}"
    if sp.is_lco:
        inter = set(sp.train_clips) & set(sp.unseen_clips)
        assert not inter, f"LEAKAGE clip train/unseen: {sorted(inter)}"
        lab = np.asarray(clip_labels)
        # ogni classe presente nelle clip non viste deve avere anche clip viste (altrimenti la classe
        # non sarebbe mai appresa e il Δstim sarebbe falsato)
        assert set(lab[sp.unseen_clips]) <= set(lab[sp.train_clips]), "classe senza clip di training"


def check_windows_in_split(win_subjects, win_clips, allowed_subjects, allowed_clips, where: str) -> None:
    """Nessuna finestra fuori dal suo split (incluse finestre 'a cavallo')."""
    ws, wc = np.asarray(win_subjects), np.asarray(win_clips)
    bad_s = set(np.unique(ws)) - set(allowed_subjects)
    bad_c = set(np.unique(wc)) - set(allowed_clips)
    assert not bad_s, f"LEAKAGE finestre {where}: soggetti non ammessi {sorted(bad_s)}"
    assert not bad_c, f"LEAKAGE finestre {where}: clip non ammesse {sorted(bad_c)}"


def permute_clip_labels(clip_labels, seed: int, task_clips: list[int] | None = None) -> np.ndarray:
    """Permutazione a livello di clip: rimescola la mappa clip -> classe preservando il numero di
    clip per classe. Le clip escluse dal task (label -1) restano escluse."""
    lab = np.asarray(clip_labels).copy()
    idx = np.array(task_clips if task_clips is not None else np.where(lab >= 0)[0])
    rng = np.random.default_rng(424_242 + seed)
    lab[idx] = lab[idx][rng.permutation(len(idx))]
    return lab
