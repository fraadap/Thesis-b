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


# ------------------------------------------------------------------ quantità di dati vs novità dello stimolo
def _per_subject_k(d: pd.DataFrame, cond_of_k, metric: str) -> pd.DataFrame:
    """Tabella soggetto x k (fold di clip), mediata su seed ed estrazioni."""
    d = d.copy()
    d["k"] = d.apply(cond_of_k, axis=1)
    d = d[d.k.notna()]
    if d.empty:
        return pd.DataFrame()
    return d.pivot_table(index="subject", columns="k", values=metric, aggfunc="mean")


def stim_decomposition(df: pd.DataFrame, config: str, base_protocol: str = "split_fm", metric: str = "ba_win",
                       task: str = "emo9") -> dict:
    """Scomposizione per soggetto (media sui fold di clip k presenti in tutte le condizioni):

        BA_full - BA_unseen = (BA_full - BA_size) + (BA_size - BA_unseen)
                               quantità di dati     Δstim^size

    BA_full   = modello con clip condivise (split_fm / lnso10) valutato su T_k (condizione 'tk{k}')
    BA_size   = modello *_sizematched (stessi soggetti e finestre di training del fold k) su T_k
    BA_unseen = modello *_lco del fold k sulle clip non viste T_k (stessi esempi di BA_size)
    Δstim^size corretto = Δstim^size - (BA_size^perm - 1/K), con BA_size^perm dai modelli *_sizematched
    addestrati con la mappa clip -> classe permutata (stessa correzione 'rispetto al caso' dell'indice
    appaiato). Il confronto con l'indice appaiato corretto (stesso modello, viste vs non viste) misura
    l'effetto della diversità degli stimoli in training: va riportato anche se diverge.
    """
    from .stats import paired_test
    K = 9 if task == "emo9" else 2
    lco, sm = base_protocol + "_lco", base_protocol + "_sizematched"
    d = df[(df.config == config) & (df.task == task)]
    v = d.variant.astype(str)

    def k_from_cond(r):
        c = str(r.test_condition)
        return int(c[2:]) if c.startswith("tk") and c[2:].isdigit() else None

    # 'tkeval' = rivalutazione su T_k di un job con clip condivise completato prima delle condizioni tk
    full = _per_subject_k(d[(d.protocol == base_protocol) & v.isin(["base", "tkeval"])], k_from_cond, metric)
    size = _per_subject_k(d[(d.protocol == sm) & v.str.fullmatch(r"draw\d+")], k_from_cond, metric)
    sperm = _per_subject_k(d[(d.protocol == sm) & v.str.fullmatch(r"draw\d+_perm\d+")], k_from_cond, metric)
    un = d[(d.protocol == lco) & (v == "base") & (d.test_condition == "unseen")]
    if full.empty or size.empty or un.empty:
        return {}
    if (un.fold >= 0).all():
        unseen = _per_subject_k(un, lambda r: int(r.fold) % 3, metric)
        ks = sorted(set(full.columns) & set(size.columns) & set(unseen.columns))
        unseen_m = unseen[ks].mean(axis=1)
    else:  # JSON compatto: 'unseen' già mediato sui fold -> servono tutti e 3 i k
        ks = sorted(set(full.columns) & set(size.columns))
        if len(ks) < 3:
            return {}
        unseen_m = un.groupby("subject")[metric].mean()
    if not ks:
        return {}
    subj = sorted(set(full.index) & set(size.index) & set(unseen_m.index))
    f, s_, u = full.loc[subj, ks].mean(axis=1), size.loc[subj, ks].mean(axis=1), unseen_m.loc[subj]
    comp = {"total": f - u, "data_quantity": f - s_, "dstim_size": s_ - u}
    out = {"subjects": subj, "folds_k": ks, "ba_full": float(f.mean()), "ba_size": float(s_.mean()),
           "ba_unseen": float(u.mean()), "chance": 1 / K}
    if not sperm.empty and set(ks) <= set(sperm.columns):
        sp_ = sperm.reindex(subj)[ks].mean(axis=1)
        comp["dstim_size_corr"] = comp["dstim_size"] - (sp_ - 1 / K)
        out["ba_size_perm"] = float(sp_.mean())
    paired = delta_stim(df, config, lco, metric=metric, task=task)
    if paired.get("corr_chance") is not None:
        pc = pd.Series(paired["corr_chance"], index=paired["subjects"]).reindex(subj)
        comp["paired_corr"] = pc
        if "dstim_size_corr" in comp:
            comp["size_minus_paired"] = comp["dstim_size_corr"] - pc   # effetto della diversità degli stimoli
    for name, x in comp.items():
        x = x.astype(float)
        ok = x.notna()
        out[name] = {"per_subject": np.round(x.values, 4).tolist(),
                     **paired_test(x[ok].values, np.zeros(int(ok.sum())))}
    return out


def decomposition_lines(dec: dict, name: str) -> list[str]:
    if not dec:
        return []
    def f(k):
        if k not in dec:
            return "n.d."
        r = dec[k]
        return f"{r['mean_diff']:+.4f} [IC95 {r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] p {r.get('p', float('nan')):.3g}"
    return [f"- Scomposizione {name} (k = {dec['folds_k']}, {len(dec['subjects'])} soggetti): "
            f"BA full {dec['ba_full']:.3f}, size {dec['ba_size']:.3f}, unseen {dec['ba_unseen']:.3f}"
            + (f", size permutato {dec['ba_size_perm']:.3f}" if "ba_size_perm" in dec else ""),
            f"    totale {f('total')}; quantità di dati {f('data_quantity')}; Δstim^size {f('dstim_size')}; "
            f"Δstim^size corretto {f('dstim_size_corr')}",
            f"    indice appaiato corretto {f('paired_corr')}; differenza (size - appaiato) = diversità degli "
            f"stimoli {f('size_minus_paired')}"]


def tk_rows_from_preds(job_id: str, data) -> list[dict]:
    """Righe 'tk{k}' (BA_full su T_k) dalle predizioni salvate di un job split_fm/lnso10 già completato."""
    from ..data.splits import heldout_sets
    from .metrics import evaluate_predictions
    parts = job_id.split("__")
    if len(parts) != 6:
        return []
    config, protocol, fold, seed, task, variant = parts
    if protocol not in ("split_fm", "lnso10") or variant != "base":
        return []
    probs = np.asarray(data["probs"], dtype=np.float32)
    subj, clip, lab = (np.asarray(data[k]) for k in ("subject", "clip", "label"))
    rows = []
    for k, tk in heldout_sets(task_clips=sorted(set(clip.tolist()))).items():
        m = np.isin(clip, tk)
        if not m.any():
            continue
        ev = evaluate_predictions(probs[m], subj[m], clip[m], lab[m])
        for s, r in ev["per_subject"].items():
            rows.append({"config": config, "protocol": protocol, "fold": int(fold[1:]), "seed": int(seed[1:]),
                         "variant": variant, "test_condition": f"tk{k}", "task": task, "subject": s,
                         **{kk: (None if isinstance(vv, float) and np.isnan(vv) else vv) for kk, vv in r.items()},
                         "source": "preds"})
    return rows
