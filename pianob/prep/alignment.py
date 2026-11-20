"""Allineamento per soggetto, label-free (trasduttivo per i soggetti di test: va dichiarato).

EA (Euclidean Alignment, He & Wu 2020):
    R_s = 1/N_s * sum_i LW(X_i X_i^T / L),   X~_i = R_s^{-1/2} X_i
con covarianza Ledoit-Wolf per finestra; riferimento euclideo (default) o Riemanniano.
"""
from __future__ import annotations

import numpy as np


def lw_cov(X: np.ndarray) -> np.ndarray:
    """Covarianza Ledoit-Wolf (assume_centered=True, come sklearn.covariance.ledoit_wolf),
    vettorizzata. X: (n, C, L) oppure (C, L). Ritorna (n, C, C)."""
    X = np.asarray(X, dtype=np.float64)
    single = X.ndim == 2
    if single:
        X = X[None]
    n, C, L = X.shape
    emp = np.einsum("ncl,ndl->ncd", X, X) / L
    tr = np.einsum("ncc->n", emp)
    mu = tr / C
    X2 = X ** 2
    beta_ = np.einsum("ncl,ndl->n", X2, X2)                 # sum_ij sum_k x_ik^2 x_jk^2
    delta_ = np.einsum("ncd,ncd->n", emp, emp)              # ||emp||_F^2
    beta = (beta_ / L - delta_) / (C * L)
    delta = (delta_ - 2 * mu * tr + C * mu ** 2) / C
    beta = np.minimum(beta, delta)
    shrink = np.where(delta > 0, beta / np.where(delta > 0, delta, 1), 0.0)
    eye = np.eye(C)[None]
    out = (1 - shrink)[:, None, None] * emp + (shrink * mu)[:, None, None] * eye
    return out[0] if single else out


def _eig_fn(R: np.ndarray, fn) -> np.ndarray:
    w, V = np.linalg.eigh((R + R.T) / 2)
    w = np.maximum(w, 1e-12 * max(w.max(), 1e-30))
    return (V * fn(w)) @ V.T


def inv_sqrtm(R):
    return _eig_fn(R, lambda w: w ** -0.5)


def sqrtm(R):
    return _eig_fn(R, np.sqrt)


def logm(R):
    return _eig_fn(R, np.log)


def expm_sym(R):
    w, V = np.linalg.eigh((R + R.T) / 2)
    return (V * np.exp(w)) @ V.T


def riemann_mean(covs: np.ndarray, tol: float = 1e-8, max_iter: int = 50) -> np.ndarray:
    """Media geometrica (metrica affine-invariante), algoritmo a punto fisso."""
    M = covs.mean(0)
    for _ in range(max_iter):
        Ms, Mis = sqrtm(M), inv_sqrtm(M)
        T = np.mean([logm(Mis @ c @ Mis) for c in covs], axis=0)
        M = Ms @ expm_sym(T) @ Ms
        if np.linalg.norm(T) < tol:
            break
    return M


def reference_cov(windows: np.ndarray, ref: str = "euclid") -> np.ndarray:
    covs = lw_cov(windows)
    return covs.mean(0) if ref == "euclid" else riemann_mean(covs)


def ea_matrix(windows: np.ndarray, ref: str = "euclid") -> np.ndarray:
    """windows: (n, C, L) finestre NON etichettate di un soggetto."""
    return inv_sqrtm(reference_cov(windows, ref))


class StreamingEA:
    """EA online: la covarianza di riferimento si aggiorna finestra per finestra.

    Hook per un prior di normalizzazione (Proposta 3): R_n = (sum_i C_i + n0 * R_prior) / (n + n0).
    """

    def __init__(self, n_channels: int, prior: np.ndarray | None = None, n0: float = 0.0):
        self.sum = np.zeros((n_channels, n_channels))
        self.n = 0
        self.prior = prior
        self.n0 = n0 if prior is not None else 0.0

    def update(self, window: np.ndarray) -> None:
        self.sum += lw_cov(window)
        self.n += 1

    def reference(self) -> np.ndarray:
        num = self.sum + (self.n0 * self.prior if self.n0 > 0 else 0)
        den = self.n + self.n0
        assert den > 0, "nessuna finestra e nessun prior"
        return num / den

    def matrix(self) -> np.ndarray:
        return inv_sqrtm(self.reference())

    def transform(self, window: np.ndarray, update_first: bool = True) -> np.ndarray:
        if update_first:
            self.update(window)
        return (self.matrix() @ window).astype(np.float32)


def subject_windows(trials: np.ndarray, win_len: int) -> np.ndarray:
    """Finestre non sovrapposte di un soggetto: trials (n_clip, C, T) -> (n, C, win_len)."""
    n_clip, C, T = trials.shape
    k = T // win_len
    w = trials[:, :, :k * win_len].reshape(n_clip, C, k, win_len)
    return w.transpose(0, 2, 1, 3).reshape(-1, C, win_len)


def align_subject(trials: np.ndarray, method: str, win_len: int, ref: str = "euclid",
                  fit_trials: np.ndarray | None = None) -> np.ndarray:
    """Allinea tutti i trial (n_clip, C, T) di un soggetto.

    `fit_trials` = trial su cui stimare le statistiche (default: gli stessi). Per i soggetti di
    training sono solo le clip di training; per i soggetti di test l'intera sessione (trasduttivo).
    """
    fit = trials if fit_trials is None else fit_trials
    if method == "none":
        return trials
    if method == "ea":
        W = ea_matrix(subject_windows(fit, win_len), ref).astype(np.float32)
        return np.einsum("cd,ndt->nct", W, trials).astype(np.float32)
    if method == "zscore":
        mu = fit.mean(axis=(0, 2), keepdims=True)
        sd = fit.std(axis=(0, 2), keepdims=True) + 1e-8
        return ((trials - mu) / sd).astype(np.float32)
    raise ValueError(method)


def ledoit_wolf_reference_check(X: np.ndarray) -> float:
    """Differenza massima con sklearn (usata nei test)."""
    from sklearn.covariance import ledoit_wolf
    ours = lw_cov(X)
    ref = ledoit_wolf(X.T, assume_centered=True)[0]
    return float(np.abs(ours - ref).max())
