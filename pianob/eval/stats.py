"""Statistica appaiata per soggetto, Holm, Nadeau-Bengio, soglie di caso binomiali."""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def paired_test(a, b, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Confronto appaiato per soggetto di a contro b (stessi soggetti, stesso ordine).
    Wilcoxon signed-rank con n >= 20, permutazione esatta dei segni sotto 20."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    n = len(d)
    out = {"n": n, "mean_diff": float(d.mean()) if n else float("nan"),
           "median_diff": float(np.median(d)) if n else float("nan")}
    if n < 2:
        return out
    sd = d.std(ddof=1)
    out["dz"] = float(d.mean() / sd) if sd > 0 else float("nan")
    out["frac_worse"] = float(np.mean(d < 0))
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(1)
    out["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    if n >= 20:
        if np.all(d == 0):
            out["p"], out["test"] = 1.0, "wilcoxon"
        else:
            out["p"] = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
            out["test"] = "wilcoxon"
    else:
        out["p"] = sign_flip_exact(d)
        out["test"] = "sign-flip esatto"
    return out


def sign_flip_exact(d) -> float:
    """p bilaterale esatto della permutazione dei segni sulla media delle differenze."""
    d = np.asarray(d, float)
    n = len(d)
    obs = abs(d.mean())
    signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    null = np.abs((signs * d).mean(1))
    return float(np.mean(null >= obs - 1e-12))


def holm(pvals: dict) -> dict:
    """Correzione di Holm-Bonferroni. pvals: {nome: p}. Ritorna {nome: p_adj}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def nadeau_bengio(diffs_per_fold, n_train: int, n_test: int) -> dict:
    """t-test corretto (Nadeau & Bengio 2003) sulle differenze per fold/ripetizione."""
    d = np.asarray(diffs_per_fold, float)
    k = len(d)
    if k < 2:
        return {"k": k}
    var = d.var(ddof=1) * (1 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var) if var > 0 else float("nan")
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return {"k": k, "mean": float(d.mean()), "t": float(t), "p": float(p)}


def binomial_chance_threshold(n_trials: int, n_classes: int, alpha: float = 0.05) -> float:
    """Accuratezza minima significativamente sopra il caso (binomiale unilaterale) con n_trials prove.
    Es.: 28 clip, 2 classi -> ~0.68."""
    p0 = 1.0 / n_classes
    for k in range(n_trials + 1):
        if stats.binom.sf(k - 1, n_trials, p0) < alpha:
            return k / n_trials
    return 1.0


def mde_paired(sd_diff: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Effetto minimo rilevabile (media delle differenze appaiate), approssimazione t."""
    if n < 3 or not np.isfinite(sd_diff):
        return float("nan")
    t_a = stats.t.ppf(1 - alpha / 2, n - 1)
    t_b = stats.t.ppf(power, n - 1)
    return float((t_a + t_b) * sd_diff / np.sqrt(n))
