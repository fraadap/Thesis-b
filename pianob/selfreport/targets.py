"""Target della Sessione 03: label per (soggetto, clip) dai voti, mai dall'EEG.

Task (classificazione; -1 = trial escluso):
- sr_val_bin, sr_aro_bin, sr_lik_bin, sr_fam_bin: split alla mediana entro soggetto (14/14) del voto;
- sr_val_bin_strict: senza i trial in pareggio con la mediana;
- sr_val_bintr: variante di robustezza (§5), z-score e mediana calcolati solo sulle clip di training del fold;
- sr_argmax8: item discreto più votato (8 classi, ordine delle categorie FACED senza neutral);
- sr_val_res, sr_aro_res: segno dello scarto entro clip z(s,i) - mu_train(i), con mu dai soli soggetti di
  training del fold, escluso il soggetto stesso (componente individuale).

Anti-leakage (§3.3): mu_train usa solo soggetti di training e, per loro, solo le clip di training del fold; per le
clip non viste mu serve solo a definire la label di test (voti di training, mai EEG) e lo si dichiara.
Permutazione al training (§3.2): residui a due vie del voto permutati tra i soggetti di training entro ogni clip,
voto ricostruito (media di clip + media di soggetto + residuo permutato), label ricalcolate.
"""
from __future__ import annotations

import numpy as np

from ..data import faced_meta as FM

ARGMAX8_CLASSES = [c for c in range(len(FM.CATEGORIES)) if c != FM.NEUTRAL_CLASS]
SR_TASKS = {
    "sr_val_bin": {"item": "valence", "kind": "bin", "n_classes": 2},
    "sr_aro_bin": {"item": "arousal", "kind": "bin", "n_classes": 2},
    "sr_lik_bin": {"item": "liking", "kind": "bin", "n_classes": 2},
    "sr_fam_bin": {"item": "familiarity", "kind": "bin", "n_classes": 2},
    "sr_val_bin_strict": {"item": "valence", "kind": "bin_strict", "n_classes": 2},
    "sr_val_bintr": {"item": "valence", "kind": "bin_trainclips", "n_classes": 2},
    "sr_val_res": {"item": "valence", "kind": "res", "n_classes": 2},
    "sr_aro_res": {"item": "arousal", "kind": "res", "n_classes": 2},
    "sr_argmax8": {"item": "valence", "kind": "argmax8", "n_classes": 8},
}
CATEGORY_TASKS = {"sr_val_bin": "val2"}      # task di categoria confrontabile (terza famiglia, ipotesi 4)


def is_sr(task: str) -> bool:
    return task in SR_TASKS


def matrix(votes, col: str, n_slots: int) -> np.ndarray:
    """(n_slots, 28) dal formato lungo della tabella; NaN dove manca."""
    M = np.full((n_slots, FM.N_CLIPS), np.nan)
    s = votes["subject"].to_numpy(int)
    c = votes["clip"].to_numpy(int)
    M[s, c] = votes[col].to_numpy(float)
    return M


def _median_split(Zrow: np.ndarray, clips, rng) -> np.ndarray:
    v = Zrow[clips]
    out = np.full(len(Zrow), -1)
    if np.isnan(v).any() or np.nanstd(v) == 0:
        return out
    order = np.argsort(v + rng.uniform(0, 1e-6, len(v)), kind="stable")
    b = np.zeros(len(v), dtype=int)
    b[order[len(v) // 2:]] = 1
    out[np.asarray(clips)] = b
    return out


def mu_train(Z: np.ndarray, split) -> tuple[np.ndarray, np.ndarray]:
    """Media della clip sui soggetti di training (solo clip di training per loro) e somma/conteggio per il
    leave-one-out. Ritorna (somma, conteggio) per clip."""
    tr = np.asarray(split.train_subjects)
    mask = np.zeros_like(Z, dtype=bool)
    for s in tr:
        mask[s, split.subject_train_clips(s)] = True
    if split.is_lco:     # anti-leakage: nessun voto delle clip T_k dei soggetti di training entra nel training
        assert not mask[:, split.unseen_clips].any()
    Zm = np.where(mask & ~np.isnan(Z), Z, 0.0)
    return Zm.sum(0), (mask & ~np.isnan(Z)).sum(0)


def res_labels(Z: np.ndarray, split, eval_mu_full: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Label del segno dello scarto entro clip e scarto continuo. Per i soggetti di training mu esclude il
    soggetto stesso; per val/test mu è la media dei soggetti di training. Sulle clip non viste (solo test) mu
    usa i voti di training di quelle clip: definisce la label di valutazione, non entra nel training."""
    ssum, cnt = mu_train(Z, split)
    S, C = Z.shape
    mu = np.full((S, C), np.nan)
    tr = set(split.train_subjects)
    full_sum = ssum.copy()
    full_cnt = cnt.copy()
    if split.is_lco and eval_mu_full:
        for c in split.unseen_clips:
            v = Z[list(tr), c]
            full_sum[c], full_cnt[c] = np.nansum(v), np.sum(~np.isnan(v))
    for s in range(S):
        if s in tr:      # leave-one-out: il soggetto non entra nella media della propria clip
            inc = (~np.isnan(Z[s]) & np.isin(np.arange(C), split.subject_train_clips(s))).astype(float)
            loo_sum = ssum - np.where(inc > 0, Z[s], 0.0)
            loo_cnt = cnt - inc
            mu[s] = np.where(loo_cnt > 0, loo_sum / np.maximum(loo_cnt, 1), np.nan)
            if split.is_lco:
                mu[s, split.unseen_clips] = np.nan          # nessuna label di training sulle clip T_k
        else:
            mu[s] = full_sum / np.maximum(full_cnt, 1)
    r = Z - mu
    lab = np.where(np.isnan(r) | (r == 0), -1, (r > 0).astype(int))
    return lab, r


def permute_train_residuals(Z: np.ndarray, split, perm: int) -> np.ndarray:
    """Voto dei soggetti di training ricostruito con i residui a due vie permutati entro ogni clip (§3.2).
    Gli altri soggetti restano invariati. Le medie di riga e di colonna (sulle clip di training) si conservano."""
    rng = np.random.default_rng(515_151 + 1000 * perm + split.fold)
    tr = [s for s in split.train_subjects if not np.isnan(Z[s, split.train_clips]).any()]
    clips = list(split.train_clips)
    B = Z[np.ix_(tr, clips)]
    E = B - B.mean(1, keepdims=True) - B.mean(0, keepdims=True) + B.mean()
    Ep = np.empty_like(E)
    for j in range(E.shape[1]):
        Ep[:, j] = E[rng.permutation(E.shape[0]), j]
    Ep -= Ep.mean(1, keepdims=True)      # ricentrato per riga: medie di soggetto e di clip invariate
    Zp = Z.copy()
    Zp[np.ix_(tr, clips)] = B - E + Ep
    return Zp


def label_matrix(task: str, votes, n_slots: int, split=None, perm: int | None = None) -> np.ndarray:
    """(n_slots, 28) int, -1 = escluso. split serve per i target dipendenti dal fold e per le permutazioni."""
    spec = SR_TASKS[task]
    kind, item = spec["kind"], spec["item"]
    rng = np.random.default_rng(97_531 + (perm or 0))
    if kind == "argmax8":
        assert perm is None, "sr_argmax8: permutazione entro clip non definita"
        am = matrix(votes, "argmax8", n_slots)
        lut = {c: k for k, c in enumerate(ARGMAX8_CLASSES)}
        return np.vectorize(lambda v: lut.get(int(v), -1) if not np.isnan(v) else -1)(am).astype(int)
    if perm is None and kind in ("bin", "bin_strict"):
        col = f"{item}_{kind}"
        M = matrix(votes, col, n_slots)
        return np.where(np.isnan(M), -1, M).astype(int)
    Z = matrix(votes, f"{item}_z", n_slots)
    if kind == "bin_trainclips":
        raw = matrix(votes, item, n_slots)
        clips = list(split.train_clips)
        mu, sd = np.nanmean(raw[:, clips], 1, keepdims=True), np.nanstd(raw[:, clips], 1, keepdims=True)
        Z = np.where(sd > 0, (raw - mu) / np.where(sd > 0, sd, 1), np.nan)
    if perm is not None:
        assert split is not None
        Z = permute_train_residuals(Z, split, perm)
    if kind == "res":
        return res_labels(Z, split)[0]
    if kind == "bin_strict":       # dopo una permutazione: pareggi alla mediana del voto ricostruito
        out = np.full(Z.shape, -1)
        for s in range(n_slots):
            v = Z[s]
            if np.isnan(v).all():
                continue
            med = np.nanmedian(v)
            out[s] = np.where(v > med, 1, np.where(v < med, 0, -1))
        return out
    if kind == "bin_trainclips":
        # z e mediana dalle sole clip di training del fold, applicati a tutte le clip del soggetto
        out = np.full(Z.shape, -1)
        clips = list(split.train_clips)
        for s in range(n_slots):
            v = Z[s]
            if np.isnan(v[clips]).all():
                continue
            med = np.nanmedian(v[clips])
            out[s] = np.where(np.isnan(v) | (v == med), -1, (v > med).astype(int))
        return out
    # bin permutato: label vere per val/test, ricalcolate (mediana delle clip di training) per il training
    out = label_matrix(task, votes, n_slots)
    for s in split.train_subjects:
        if not np.isnan(Z[s, split.train_clips]).any():
            out[s] = _median_split(Z[s], list(split.train_clips), rng)
    return out


def continuous_target(task: str, votes, n_slots: int) -> np.ndarray:
    """Voto z (soggetti x clip) usato dalle metriche continue per il task."""
    return matrix(votes, f"{SR_TASKS[task]['item']}_z", n_slots)
