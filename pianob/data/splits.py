"""Protocolli di valutazione deterministici con asserzioni anti-leakage.

- split_fm      : train sub000-079, val 080-099, test 100-122; tutte le clip ovunque (stimoli condivisi)
- split_fm_lco  : stessi soggetti x 3 fold di clip; test su clip viste (baseline appaiata) e non viste
- lnso10        : 10 fold sequenziali sui soggetti (12 per fold, 15 nell'ultimo), val = 10% del train
- lnso10_lco    : lnso10 x 3 fold di clip (fold = 3 * fold_soggetti + fold_clip)
- split_fm_sizematched / lnso10_sizematched : controllo della quantità di dati per il leave-clip-out.
    Stessi soggetti e stesso numero di finestre di training del fold LCO k, ma ogni soggetto di training
    contribuisce un sottoinsieme casuale di clip (diverso per soggetto) con la stessa composizione per
    classe; le clip T_k sono quindi viste da circa 2/3 dei soggetti. Test sugli stessi esempi della
    condizione 'unseen' del fold k (stessi soggetti, clip T_k, stesso allineamento).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import faced_meta as FM

PROTOCOLS = ("split_fm", "split_fm_lco", "lnso10", "lnso10_lco", "split_fm_sizematched", "lnso10_sizematched")
SIZEMATCHED = ("split_fm_sizematched", "lnso10_sizematched")
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
    # solo protocolli *_sizematched: clip di training per soggetto e clip T_k di test
    train_clips_per_subject: dict | None = None
    heldout_clips: list[int] = field(default_factory=list)
    draw: int | None = None
    all_clips: list[int] = field(default_factory=list)   # clip del task (per l'allineamento dei soggetti di test)

    @property
    def is_sizematched(self) -> bool:
        return self.train_clips_per_subject is not None

    def subject_train_clips(self, s: int) -> list[int]:
        return list(self.train_clips_per_subject[s]) if self.is_sizematched else list(self.train_clips)

    def n_train_trials(self) -> int:
        return sum(len(self.subject_train_clips(s)) for s in self.train_subjects)

    @property
    def is_lco(self) -> bool:
        return len(self.unseen_clips) > 0

    def eval_conditions(self) -> dict[str, list[int]]:
        """Condizioni di test: 'all' (split senza LCO) oppure 'seen' / 'unseen'."""
        if self.is_lco:
            return {"seen": list(self.train_clips), "unseen": list(self.unseen_clips)}
        if self.is_sizematched:
            return {f"tk{self.clip_fold}": list(self.heldout_clips)}
        return {"all": list(self.train_clips)}

    @property
    def clip_fold(self) -> int:
        return self.fold % N_CLIP_FOLDS if self.protocol.startswith("lnso10") else self.fold

    def to_dict(self):
        return asdict(self)


def n_folds(protocol: str) -> int:
    return {"split_fm": 1, "split_fm_lco": N_CLIP_FOLDS, "lnso10": N_SUBJ_FOLDS,
            "lnso10_lco": N_SUBJ_FOLDS * N_CLIP_FOLDS, "split_fm_sizematched": N_CLIP_FOLDS,
            "lnso10_sizematched": N_SUBJ_FOLDS * N_CLIP_FOLDS}[protocol]


def heldout_sets(clip_labels=FM.CLIP_LABELS, task_clips=None) -> dict[int, list[int]]:
    """T_k per ogni fold di clip k (le clip 'unseen' del leave-clip-out)."""
    cf = clip_fold_assignment(clip_labels)
    tc = range(len(clip_labels)) if task_clips is None else task_clips
    return {k: [c for c in sorted(tc) if cf[c] == k] for k in range(N_CLIP_FOLDS)}


def sizematched_subsets(subjects, pool: list[int], n_per_stratum: dict, strata: dict, heldout: list[int],
                        seed_key: tuple, max_tries: int = 1000) -> tuple[dict, int]:
    """Sottoinsieme casuale di stimoli per soggetto, con `n_per_stratum[st]` stimoli per strato
    (classe per FACED; un unico strato per i brani musicali). Diverso per soggetto, deterministico dato
    `seed_key`. Si ripete l'estrazione (rejection sampling) finché ogni stimolo di `heldout` compare in
    almeno un soggetto: con 80 soggetti non succede mai, serve solo ai test con pochi soggetti.
    Generico: usato anche per leave-song-out vs brani condivisi (MUSIN-G 9/12, NMED-T 8/10)."""
    by_st: dict = {}
    for c in pool:
        by_st.setdefault(strata[c], []).append(c)
    for st, n in n_per_stratum.items():
        assert len(by_st.get(st, [])) >= n, f"strato {st}: {len(by_st.get(st, []))} stimoli < {n}"
    for attempt in range(max_tries):
        out = {}
        for s in subjects:
            rng = np.random.default_rng((97531, *seed_key, attempt, int(s)))
            pick = [c for st, n in n_per_stratum.items() for c in rng.choice(sorted(by_st[st]), n, replace=False)]
            out[int(s)] = sorted(int(c) for c in pick)
        covered = set().union(*out.values()) if out else set()
        if set(heldout) <= covered:
            return out, attempt
    raise AssertionError("copertura di T_k impossibile: troppi pochi soggetti")


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
               n_subjects: int = FM.N_SUBJECTS, draw: int = 0, max_subjects: int | None = None) -> Split:
    """`max_subjects` (solo smoke test) tronca train/val/test PRIMA del sottocampionamento per soggetto."""
    if protocol in SIZEMATCHED:
        return _make_sizematched(protocol, fold, task_clips, subjects_present, clip_labels, n_subjects,
                                 draw, max_subjects)
    sp = _make_split(protocol, fold, task_clips, subjects_present, clip_labels, n_subjects)
    if max_subjects:
        k = max_subjects
        sp = Split(sp.protocol, sp.fold, sp.train_subjects[:k], sp.val_subjects[:max(1, k // 2)],
                   sp.test_subjects[:max(1, k // 2)], sp.train_clips, sp.unseen_clips,
                   all_clips=sp.all_clips)
    return sp


def _make_sizematched(protocol, fold, task_clips, subjects_present, clip_labels, n_subjects, draw,
                      max_subjects) -> Split:
    lco_name = protocol.replace("_sizematched", "_lco")
    lco = make_split(lco_name, fold, task_clips, subjects_present, clip_labels, n_subjects,
                     max_subjects=max_subjects)
    lab = np.asarray(clip_labels)
    pool = sorted(lco.train_clips + lco.unseen_clips)
    n_per = {int(c): int((lab[lco.train_clips] == c).sum()) for c in sorted(set(lab[lco.train_clips]))}
    strata = {int(c): int(lab[c]) for c in pool}
    per_subj, attempt = sizematched_subsets(lco.train_subjects, pool, n_per, strata, lco.unseen_clips,
                                            (fold, draw))
    sp = Split(protocol, fold, lco.train_subjects, lco.val_subjects, lco.test_subjects, lco.train_clips,
               unseen_clips=[], train_clips_per_subject=per_subj, heldout_clips=list(lco.unseen_clips),
               draw=draw, all_clips=pool)
    check_sizematched(sp, lco, clip_labels)
    return sp


def check_sizematched(sp: Split, lco: Split, clip_labels=FM.CLIP_LABELS) -> dict:
    """Asserzioni del controllo: stessi soggetti, stessa numerosità e composizione per classe del fold LCO,
    ogni clip di T_k vista da almeno un soggetto di training. Ritorna la copertura di T_k."""
    lab = np.asarray(clip_labels)
    assert sp.train_subjects == lco.train_subjects and sp.val_subjects == lco.val_subjects \
        and sp.test_subjects == lco.test_subjects, "soggetti diversi dal fold LCO"
    assert sp.heldout_clips == lco.unseen_clips
    want = sorted(lab[lco.train_clips].tolist())
    for s in sp.train_subjects:
        cl = sp.train_clips_per_subject[s]
        assert len(set(cl)) == len(cl) == len(lco.train_clips), f"sub{s}: {len(cl)} clip"
        assert sorted(lab[cl].tolist()) == want, f"sub{s}: composizione per classe diversa"
    assert sp.n_train_trials() == lco.n_train_trials(), "numero di trial di training diverso dal fold LCO"
    cov = coverage(sp)
    missing = [c for c, n in cov.items() if n == 0]
    assert not missing, f"clip di T_k mai viste in training: {missing}"
    return cov


def coverage(sp: Split) -> dict:
    """Per ogni clip di T_k: numero di soggetti di training che la vedono."""
    return {c: sum(c in sp.train_clips_per_subject[s] for s in sp.train_subjects) for c in sp.heldout_clips}


def _make_split(protocol: str, fold: int = 0, task_clips: list[int] | None = None,
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
    sp = Split(protocol, fold, tr, va, te, train_clips, unseen, shared_stimuli=True, all_clips=list(task_clips))
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


# ------------------------------------------------------------------ dataset musicali (sessioni future)
def stimulus_folds(stimuli: list, n_heldout: int, seed: int = 0) -> list[list]:
    """Partizione deterministica degli stimoli in fold da `n_heldout` (leave-song-out):
    MUSIN-G 12 brani -> 4 fold da 3 (9/12 in training); NMED-T 10 brani -> 5 fold da 2 (8/10)."""
    st = list(stimuli)
    assert len(st) % n_heldout == 0, f"{len(st)} stimoli non divisibili in fold da {n_heldout}"
    perm = np.random.default_rng(31337 + seed).permutation(len(st))
    return [sorted(st[i] for i in perm[k:k + n_heldout]) for k in range(0, len(st), n_heldout)]


def stimulus_holdout_plan(subjects: list, stimuli: list, heldout: list, draw: int = 0, strata: dict | None = None,
                          fold: int = 0) -> dict:
    """Leave-stimulus-out contro stimoli condivisi sottocampionati, con la stessa logica di
    split_fm_sizematched: ogni soggetto di training vede len(stimuli)-len(heldout) stimoli casuali
    (stessa composizione per strato del leave-stimulus-out), valutazione sugli stessi stimoli tenuti fuori."""
    strata = strata or {c: 0 for c in stimuli}
    train_shared = [c for c in stimuli if c not in heldout]
    n_per: dict = {}
    for c in train_shared:
        n_per[strata[c]] = n_per.get(strata[c], 0) + 1
    per_subj, _ = sizematched_subsets(subjects, list(stimuli), n_per, strata, heldout, (fold, draw))
    assert all(len(v) == len(train_shared) for v in per_subj.values())
    return {"heldout": list(heldout), "train_shared": train_shared, "per_subject": per_subj}
