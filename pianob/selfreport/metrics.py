"""Metriche della Sessione 03 su matrici soggetti x clip (righe = soggetti, colonne = clip).

P = predizione continua per trial del modello, Z = voto z-scorato entro soggetto. Matrici complete (i soggetti
esclusi per un item si tolgono prima). Tutte le funzioni accettano dimensioni iniziali di batch (..., S, C).

- resid_corr: Pearson tra i residui a due vie di P e di Z (interazione soggetto x clip). Primaria. Un modello
  che riconosce solo la clip (P costante entro colonna) dà 0 per costruzione; offset di soggetto non contano.
- within_clip: P e Z centrati entro soggetto, Spearman per clip sui soggetti, media di Fisher pesata (n - 3).
- between_clip: Spearman tra medie per clip di P e di Z (ottenibile memorizzando sulle clip viste).
- within_subject: Spearman per soggetto sulle sue clip, media sui soggetti (mescola le due componenti).
Nulli: permutazione dei residui a due vie del voto tra soggetti entro ogni clip (Winkler et al. 2015),
controllo con la permutazione di Z entro clip seguita dalla ricentratura entro soggetto.
IC: bootstrap incrociato soggetti x clip (Owen 2007) e bootstrap sui soli soggetti.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

EPS = 1e-12


def resid2(M: np.ndarray) -> np.ndarray:
    return M - M.mean(-1, keepdims=True) - M.mean(-2, keepdims=True) + M.mean((-2, -1), keepdims=True)


def _pearson(a, b, axis):
    a = a - a.mean(axis, keepdims=True)
    b = b - b.mean(axis, keepdims=True)
    num = (a * b).sum(axis)
    den = np.sqrt((a * a).sum(axis) * (b * b).sum(axis))
    return np.where(den > EPS, num / np.maximum(den, EPS), 0.0)   # colonna costante -> 0


def resid_corr(P, Z):
    a, b = resid2(P), resid2(Z)
    return _pearson(a.reshape(*a.shape[:-2], -1), b.reshape(*b.shape[:-2], -1), -1)


def _fisher_mean(r, n, axis=-1):
    w = max(n - 3, 1)
    return np.tanh(np.mean(np.arctanh(np.clip(r, -0.999999, 0.999999)) * w, axis) / w)


def _centre(M):
    """Centratura entro soggetto, arrotondata: una colonna costante non deve diventare rumore numerico
    che il rango trasformerebbe in una correlazione spuria (memorizzatore puro = 0 esatto)."""
    return np.round(M - M.mean(-1, keepdims=True), 9)


def within_clip(P, Z, method: str = "spearman"):
    Pc, Zc = _centre(P), _centre(Z)
    if method == "spearman":
        Pc, Zc = rankdata(Pc, axis=-2), rankdata(Zc, axis=-2)
    r = _pearson(Pc, Zc, -2)                              # (..., C): una correlazione per clip
    return _fisher_mean(r, P.shape[-2])


def between_clip(P, Z):
    return _pearson(rankdata(np.round(P.mean(-2), 9), axis=-1), rankdata(np.round(Z.mean(-2), 9), axis=-1), -1)


def within_subject(P, Z):
    return _pearson(rankdata(np.round(P, 9), axis=-1), rankdata(np.round(Z, 9), axis=-1), -1).mean(-1)


METRICS = {"resid_corr": resid_corr, "within_clip": within_clip, "between_clip": between_clip,
           "within_subject": within_subject}


def all_metrics(P, Z) -> dict:
    return {k: float(f(P, Z)) for k, f in METRICS.items()}


def per_clip_within(P, Z, method="spearman") -> np.ndarray:
    Pc, Zc = _centre(P), _centre(Z)
    if method == "spearman":
        Pc, Zc = rankdata(Pc, axis=-2), rankdata(Zc, axis=-2)
    return _pearson(Pc, Zc, -2)


# ------------------------------------------------------------------ intervalli di confidenza
def bootstrap_ci(P, Z, n: int = 2000, seed: int = 0, crossed: bool = True, metrics=None, batch: int = 200) -> dict:
    """IC al 95% (percentili). crossed: soggetti e clip ricampionati indipendentemente ("pigeonhole")."""
    rng = np.random.default_rng(seed)
    S, C = P.shape
    metrics = metrics or list(METRICS)
    vals = {m: [] for m in metrics}
    for b0 in range(0, n, batch):
        nb = min(batch, n - b0)
        ri = rng.integers(0, S, (nb, S))
        ci = rng.integers(0, C, (nb, C)) if crossed else np.broadcast_to(np.arange(C), (nb, C))
        Pb = P[ri[:, :, None], ci[:, None, :]]
        Zb = Z[ri[:, :, None], ci[:, None, :]]
        for m in metrics:
            vals[m].append(np.atleast_1d(METRICS[m](Pb, Zb)))
    return {m: [float(np.nanpercentile(np.concatenate(v), 2.5)), float(np.nanpercentile(np.concatenate(v), 97.5))]
            for m, v in vals.items()}


# ------------------------------------------------------------------ nulli
def permute_resid_within_clip(Z: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """n repliche di Z con i residui a due vie permutati tra soggetti entro ogni clip (e ricentrati per riga):
    medie di riga, di colonna e generale del voto ricostruito identiche a quelle di Z."""
    E = resid2(Z)
    base = Z - E
    idx = np.argsort(rng.random((n, *Z.shape)), axis=1)              # permutazione per clip (colonna)
    Ep = np.take_along_axis(np.broadcast_to(E, (n, *Z.shape)), idx, axis=1)
    return base + Ep - Ep.mean(-1, keepdims=True)     # ricentratura per riga: medie conservate


def permute_recentred(Z: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    idx = np.argsort(rng.random((n, *Z.shape)), axis=1)
    Zp = np.take_along_axis(np.broadcast_to(Z, (n, *Z.shape)), idx, axis=1)
    return Zp - Zp.mean(-1, keepdims=True) + Z.mean(-1, keepdims=True)


def perm_test(P, Z, n: int = 5000, seed: int = 0, mode: str = "residual", metrics=("resid_corr", "within_clip"),
              batch: int = 250) -> dict:
    """p unilaterale (metrica > nullo) a modello fisso; stesse permutazioni per tutte le metriche."""
    rng = np.random.default_rng(seed)
    obs = {m: float(METRICS[m](P, Z)) for m in metrics}
    ge = {m: 0 for m in metrics}
    null = {m: [] for m in metrics}
    gen = permute_resid_within_clip if mode == "residual" else permute_recentred
    for b0 in range(0, n, batch):
        nb = min(batch, n - b0)
        Zp = gen(Z, rng, nb)
        Pb = np.broadcast_to(P, Zp.shape)
        for m in metrics:
            v = METRICS[m](Pb, Zp)
            ge[m] += int((v >= obs[m] - 1e-12).sum())
            null[m].append(v)
    return {m: {"observed": obs[m], "p": (1 + ge[m]) / (1 + n), "null_mean": float(np.mean(np.concatenate(null[m]))),
                "null_p95": float(np.percentile(np.concatenate(null[m]), 95))} for m in metrics}


# ------------------------------------------------------------------ descrittive dei voti
def anova2(Y: np.ndarray) -> dict:
    """Componenti di varianza (effetti casuali incrociati, disegno bilanciato senza repliche)."""
    S, C = Y.shape
    g = Y.mean()
    ms_s = C * ((Y.mean(1) - g) ** 2).sum() / (S - 1)
    ms_c = S * ((Y.mean(0) - g) ** 2).sum() / (C - 1)
    ms_e = (resid2(Y) ** 2).sum() / ((S - 1) * (C - 1))
    v_s, v_c, v_e = max((ms_s - ms_e) / C, 0.0), max((ms_c - ms_e) / S, 0.0), ms_e
    tot = v_s + v_c + v_e
    ss_tot = ((Y - g) ** 2).sum()
    return {"var_subject": v_s, "var_clip": v_c, "var_residual": v_e,
            "share_subject": v_s / tot, "share_clip": v_c / tot, "share_residual": v_e / tot,
            "eta2_clip": S * ((Y.mean(0) - g) ** 2).sum() / ss_tot,
            # clip = bersagli, soggetti = valutatori (Shrout & Fleiss 1979)
            "icc2_1": (ms_c - ms_e) / (ms_c + (S - 1) * ms_e + S * (ms_s - ms_e) / C),
            "icc2_k": (ms_c - ms_e) / (ms_c + (ms_s - ms_e) / C)}


def anova2_ci(Y: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    S, C = Y.shape
    keys = ["share_subject", "share_clip", "share_residual", "icc2_1", "icc2_k"]
    bs = {k: [] for k in keys}
    for _ in range(n):
        Yb = Y[rng.integers(0, S, S)][:, rng.integers(0, C, C)]
        a = anova2(Yb)
        for k in keys:
            bs[k].append(a[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}


def split_half(Y: np.ndarray, n: int = 1000, seed: int = 0) -> float:
    """Affidabilità split-half delle medie per clip (Spearman-Brown), media su n divisioni dei soggetti."""
    rng = np.random.default_rng(seed)
    S = Y.shape[0]
    rs = []
    for _ in range(n):
        p = rng.permutation(S)
        a, b = Y[p[:S // 2]].mean(0), Y[p[S // 2:]].mean(0)
        r = np.corrcoef(a, b)[0, 1]
        rs.append(2 * r / (1 + r))
    return float(np.mean(rs))


def mde_resid_corr(n_subj: int, n_clip: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Correlazione minima rilevabile (unilaterale) per i residui a due vie: gradi di libertà (S-1)(C-1).
    Approssimazione che ignora la dipendenza incrociata (il bootstrap incrociato è più conservativo)."""
    return float(np.tanh((norm.ppf(1 - alpha) + norm.ppf(power)) / np.sqrt((n_subj - 1) * (n_clip - 1) - 1)))


def mde_within_clip(n_subj: int, n_clip: int, alpha: float = 0.05, power: float = 0.8) -> float:
    return float(np.tanh((norm.ppf(1 - alpha) + norm.ppf(power)) / np.sqrt(n_clip * (n_subj - 3))))


def mixed_model(P: np.ndarray, Z: np.ndarray) -> dict | None:
    """(Esplorativo) p ~ media della clip + scarto + (1 | soggetto) + (1 | clip), se statsmodels c'è."""
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    S, C = P.shape
    mu = np.broadcast_to(Z.mean(0), Z.shape)
    df = pd.DataFrame({"p": P.ravel(), "mu": mu.ravel(), "r": (Z - mu).ravel(),
                       "s": np.repeat(np.arange(S), C), "c": np.tile(np.arange(C), S), "g": 1})
    try:
        fit = smf.mixedlm("p ~ mu + r", df, groups="g", vc_formula={"s": "0 + C(s)", "c": "0 + C(c)"}).fit()
    except Exception as e:  # esplorativo: non deve fermare l'analisi
        return {"error": repr(e)[:200]}
    return {"coef_r": float(fit.params["r"]), "p_r": float(fit.pvalues["r"]), "coef_mu": float(fit.params["mu"]),
            "var_components": {k: float(v) for k, v in zip(fit.model.exog_vc.names, fit.vcomp)},
            "var_residual": float(fit.scale)}
