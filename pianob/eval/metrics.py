"""Metriche a livello di finestra e di trial (media delle probabilità per soggetto x clip)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score


def balanced_accuracy(y_true, y_pred) -> float:
    """Media delle recall sulle classi PRESENTI in y_true (= sklearn.balanced_accuracy_score)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    rec = [np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]
    return float(np.mean(rec))


def basic_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    one_class = len(np.unique(np.concatenate([y_true, y_pred]))) < 2
    return {
        "ba": balanced_accuracy(y_true, y_pred),
        "acc": float(np.mean(y_true == y_pred)),
        "kappa": float("nan") if one_class else float(cohen_kappa_score(y_true, y_pred)),
        "f1w": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1m": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def trial_aggregate(probs, subj, clip, label):
    """Media delle probabilità delle finestre per (soggetto, clip)."""
    probs, subj, clip, label = map(np.asarray, (probs, subj, clip, label))
    keys = subj.astype(np.int64) * 1000 + clip.astype(np.int64)
    uk, inv = np.unique(keys, return_inverse=True)
    P = np.zeros((len(uk), probs.shape[1]))
    np.add.at(P, inv, probs)
    P /= np.bincount(inv)[:, None]
    lab = np.zeros(len(uk), dtype=np.int64)
    lab[inv] = label
    return P, uk // 1000, uk % 1000, lab


def evaluate_predictions(probs, subj, clip, label) -> dict:
    """Metriche per soggetto (finestra e trial) e globali (pooled, come il valutatore CBraMod)."""
    probs, subj, clip, label = map(np.asarray, (probs, subj, clip, label))
    pred = probs.argmax(1)
    Pt, st, ct, lt = trial_aggregate(probs, subj, clip, label)
    predt = Pt.argmax(1)
    # Sessione 02: anche la predizione per trial dalla media delle log-probabilità (colonna aggiuntiva)
    Plp, _, _, _ = trial_aggregate(np.log(np.clip(probs.astype(np.float64), 1e-12, None)), subj, clip, label)
    predt_lp = Plp.argmax(1)
    per_subject = {}
    for s in np.unique(subj):
        m, mt = subj == s, st == s
        w = basic_metrics(label[m], pred[m])
        t = basic_metrics(lt[mt], predt[mt])
        per_subject[int(s)] = {"ba_win": w["ba"], "ba_trial": t["ba"],
                               "ba_trial_lp": balanced_accuracy(lt[mt], predt_lp[mt]),
                               "kappa": w["kappa"], "f1w": w["f1w"],
                               "f1m": w["f1m"], "acc_win": w["acc"], "n_windows": int(m.sum()),
                               "n_trials": int(mt.sum())}
    pooled_w = basic_metrics(label, pred)
    pooled_t = basic_metrics(lt, predt)
    pooled_t["ba_lp"] = balanced_accuracy(lt, predt_lp)
    n_cls = probs.shape[1]
    return {"per_subject": per_subject,
            "pooled": {**{f"{k}_win": v for k, v in pooled_w.items()},
                       **{f"{k}_trial": v for k, v in pooled_t.items()}},
            "confusion_win": confusion_matrix(label, pred, labels=list(range(n_cls))).tolist(),
            "confusion_trial": confusion_matrix(lt, predt, labels=list(range(n_cls))).tolist()}


def subject_mean(per_subject: dict, key: str = "ba_win") -> float:
    v = [d[key] for d in per_subject.values() if not np.isnan(d[key])]
    return float(np.mean(v)) if v else float("nan")
