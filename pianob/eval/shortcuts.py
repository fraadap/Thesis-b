"""Indici di scorciatoia: Δstim (seen/unseen + correzione per permutazione), ΔEMG con controllo
appaiato, sonda dell'identità della clip, permutazione delle label di test a livello di clip."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import balanced_accuracy, trial_aggregate


def _subject_means(df: pd.DataFrame, metric: str) -> pd.Series:
    """Media per soggetto su seed e fold."""
    return df.groupby("subject")[metric].mean()


def delta_stim(df: pd.DataFrame, config: str, protocol: str, metric: str = "ba_win",
               base_variant: str = "base", task: str = "emo9") -> dict:
    """Indice di scorciatoia dello stimolo, per soggetto (media su fold e seed).

    raw        = BA_seen - BA_unseen (modello con label vere, stessi soggetti di test)
    perm       = stessa differenza per i modelli con la mappa clip -> classe permutata
    corr       = raw - perm  (definizione del prompt; DISTORTA, vedi sotto)
    corr_chance= raw - (BA_seen^perm - 1/K)   (correzione raccomandata)

    Perché `corr` è distorta: con label permutate (conteggi per classe preservati) il modello predice
    per una clip non vista la label delle clip viste simili; la probabilità che la clip non vista abbia
    la stessa label permutata è (m-1)/(N-1) (= 2/27 = 0.074 con 9 classi da 3 clip), sotto il caso 1/K.
    Quindi BA_unseen^perm < 1/K per costruzione e `perm` sovrastima la memorizzazione.
    `corr_chance` usa solo la parte vista dei modelli permutati: BA_seen^perm - 1/K misura la capacità
    di memorizzare la clip; ciò che resta del divario è il vantaggio specifico dell'emozione.
    """
    d = df[(df.config == config) & (df.protocol == protocol) & (df.task == task)]
    if d.empty:
        return {}
    K = 9 if task == "emo9" else 2

    def per_subject(sub):
        piv = sub.pivot_table(index=["subject", "fold", "seed"], columns="test_condition", values=metric)
        if not {"seen", "unseen"} <= set(piv.columns):
            return None
        return piv[["seen", "unseen"]].groupby("subject").mean()

    tv = per_subject(d[d.variant == base_variant])
    if tv is None:
        return {}
    raw = tv["seen"] - tv["unseen"]
    pv = d[d.variant.str.fullmatch(r"perm\d+")]
    out = {"subjects": raw.index.tolist(), "raw": raw.values.tolist(), "raw_mean": float(raw.mean()),
           "seen_mean": float(tv["seen"].mean()), "unseen_mean": float(tv["unseen"].mean()), "chance": 1 / K,
           "n_perm_runs": int(pv[["variant", "seed"]].drop_duplicates().shape[0])}
    pp = per_subject(pv) if not pv.empty else None
    if pp is not None:
        pp = pp.reindex(raw.index)
        perm = pp["seen"] - pp["unseen"]
        corr = raw - perm
        memo = pp["seen"] - 1 / K
        corr_c = raw - memo
        out.update({"perm": perm.values.tolist(), "perm_mean": float(perm.mean()),
                    "corr": corr.values.tolist(), "corr_mean": float(corr.mean()),
                    "seen_perm_mean": float(pp["seen"].mean()), "unseen_perm_mean": float(pp["unseen"].mean()),
                    "memorization": memo.values.tolist(), "memorization_mean": float(memo.mean()),
                    "emotion_gain_seen_mean": float((tv["seen"] - pp["seen"]).mean()),
                    "corr_chance": corr_c.values.tolist(), "corr_chance_mean": float(corr_c.mean())})
    return out


def delta_emg(df: pd.DataFrame, config: str, protocol: str = "split_fm", metric: str = "ba_win",
              task: str = "emo9") -> dict:
    """ΔEMG = BA_full - BA_(4-30 Hz, senza bordo); controllo: BA_full - BA_(senza 8 canali casuali),
    e confronto diretto 4-30 senza bordo vs 4-30 senza 8 casuali."""
    d = df[(df.config == config) & (df.protocol == protocol) & (df.task == task) & (df.test_condition == "all")]
    if d.empty:
        return {}

    def sm(mask):
        x = d[mask]
        return _subject_means(x, metric) if not x.empty else None

    full = sm(d.variant == "base")
    if full is None:
        return {}
    out = {"subjects": full.index.tolist(), "full": full.values.tolist()}
    for name, mask in {
        "bp4_30": d.variant == "bp4-30",
        "noedge": d.variant == "noedge",
        "bp4_30_noedge": d.variant == "bp4-30_noedge",
        "rand8": d.variant.str.fullmatch(r"rand8d\d+"),
        "bp4_30_rand8": d.variant.str.fullmatch(r"bp4-30_rand8d\d+"),
    }.items():
        s = sm(mask)
        if s is not None:
            s = s.reindex(full.index)
            out[name] = s.values.tolist()
            out[f"delta_{name}"] = float((full - s).mean())
    return out


def clip_probe(train_emb, train_clip, test_emb, test_clip, test_label, clip_labels, test_subj=None, C=1.0) -> dict:
    """Sonda lineare che predice la clip (28 classi) su soggetti di training, valutata sui soggetti di
    test; poi BA della classe ottenuta mappando la clip predetta alla sua categoria."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    clf.fit(train_emb, train_clip)
    pc = clf.predict(test_emb)
    clip_labels = np.asarray(clip_labels)
    pred_cls = clip_labels[pc]
    out = {"clip_acc": float(np.mean(pc == test_clip)), "clip_chance": 1.0 / len(np.unique(train_clip)),
           "ba_from_clip": balanced_accuracy(test_label, pred_cls)}
    if test_subj is not None:
        test_subj = np.asarray(test_subj)
        out["per_subject"] = {int(s): {"clip_acc": float(np.mean(pc[test_subj == s] == test_clip[test_subj == s])),
                                       "ba_from_clip": balanced_accuracy(test_label[test_subj == s], pred_cls[test_subj == s])}
                              for s in np.unique(test_subj)}
    return out


def test_label_permutation(probs, subj, clip, label, n_perm: int = 1000, seed: int = 0, level: str = "win") -> dict:
    """Permuta la mappa clip -> classe sui dati di test (preservando i conteggi per classe) e ricalcola
    la BA media per soggetto. p = (1 + #{null >= oss}) / (1 + n_perm)."""
    probs, subj, clip, label = map(np.asarray, (probs, subj, clip, label))
    if level == "trial":
        probs, subj, clip, label = trial_aggregate(probs, subj, clip, label)
    pred = probs.argmax(1)
    clips = np.unique(clip)
    clip_lab = np.array([label[clip == c][0] for c in clips])
    pos = np.searchsorted(clips, clip)
    subs = np.unique(subj)

    def score(lab_of_clip):
        y = lab_of_clip[pos]
        return np.mean([balanced_accuracy(y[subj == s], pred[subj == s]) for s in subs])

    obs = score(clip_lab)
    rng = np.random.default_rng(seed)
    null = np.array([score(clip_lab[rng.permutation(len(clips))]) for _ in range(n_perm)])
    return {"observed": float(obs), "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "p": float((1 + np.sum(null >= obs)) / (1 + n_perm)), "n_perm": n_perm}
