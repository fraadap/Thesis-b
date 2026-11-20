import numpy as np
import pytest

from pianob.eval.metrics import balanced_accuracy, evaluate_predictions, trial_aggregate
from pianob.eval.stats import binomial_chance_threshold, holm, paired_test, sign_flip_exact
from pianob.prep.alignment import (StreamingEA, align_subject, ea_matrix, ledoit_wolf_reference_check, lw_cov,
                                   riemann_mean, subject_windows)


def _subject(rng, C=8, n_clip=6, T=2000):
    A = rng.standard_normal((C, C)) + 2 * np.eye(C)
    return np.einsum("cd,ndt->nct", A, rng.standard_normal((n_clip, C, T))).astype(np.float32)


def test_lw_matches_sklearn():
    rng = np.random.default_rng(0)
    for L in (50, 400):
        X = rng.standard_normal((6, L)) * rng.uniform(0.5, 3, (6, 1))
        assert ledoit_wolf_reference_check(X) < 1e-8


def test_ea_mean_covariance_is_identity():
    """Con L >> C lo shrinkage LW è trascurabile: la covarianza media dopo EA è ~ I."""
    rng = np.random.default_rng(1)
    x = _subject(rng, T=6000)
    y = align_subject(x, "ea", win_len=2000)
    w = subject_windows(y, 2000).astype(np.float64)
    emp = np.einsum("ncl,ndl->ncd", w, w).mean(0) / w.shape[-1]
    assert np.abs(emp - np.eye(8)).max() < 0.05
    # proprietà esatta: W R W^T = I con R = media delle covarianze LW
    from pianob.prep.alignment import reference_cov
    R = reference_cov(subject_windows(x, 2000))
    W = ea_matrix(subject_windows(x, 2000))
    assert np.allclose(W @ R @ W.T, np.eye(8), atol=1e-6)


def test_streaming_ea_converges_to_batch():
    rng = np.random.default_rng(2)
    w = subject_windows(_subject(rng), 400)
    s = StreamingEA(8)
    for wi in w:
        s.update(wi)
    assert np.allclose(s.matrix(), ea_matrix(w), atol=1e-8)


def test_riemann_mean_of_identical_is_itself():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((5, 5))
    P = A @ A.T + np.eye(5)
    assert np.allclose(riemann_mean(np.stack([P, P, P])), P, atol=1e-6)


def test_zscore_alignment():
    rng = np.random.default_rng(4)
    y = align_subject(_subject(rng), "zscore", 400)
    assert np.allclose(y.mean(axis=(0, 2)), 0, atol=1e-4) and np.allclose(y.std(axis=(0, 2)), 1, atol=1e-3)


def test_balanced_accuracy_known():
    assert balanced_accuracy([0, 0, 0, 1], [0, 0, 0, 0]) == 0.5
    assert balanced_accuracy([0, 1, 2], [0, 1, 2]) == 1.0
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 9, 300), rng.integers(0, 9, 300)
    assert balanced_accuracy(y, p) == pytest.approx(balanced_accuracy_score(y, p))


def test_trial_aggregation_and_eval():
    # 2 soggetti x 2 clip x 3 finestre; la media delle probabilità decide il trial
    probs = np.array([[.9, .1], [.4, .6], [.4, .6],   # s0 c0 (label 0): finestre 1/3 giuste, trial giusto
                      [.1, .9], [.2, .8], [.3, .7],   # s0 c1 (label 1)
                      [.6, .4], [.6, .4], [.6, .4],   # s1 c0
                      [.6, .4], [.6, .4], [.6, .4]])  # s1 c1
    subj = np.repeat([0, 1], 6)
    clip = np.tile(np.repeat([0, 1], 3), 2)
    lab = clip.copy()
    P, s, c, l = trial_aggregate(probs, subj, clip, lab)
    assert P.shape == (4, 2) and list(l) == [0, 1, 0, 1]
    ev = evaluate_predictions(probs, subj, clip, lab)
    assert ev["per_subject"][0]["ba_win"] == pytest.approx((1 / 3 + 1) / 2)
    assert ev["per_subject"][0]["ba_trial"] == pytest.approx(1.0)
    assert ev["per_subject"][1]["ba_trial"] == pytest.approx(0.5)


def test_stats_helpers():
    assert binomial_chance_threshold(28, 2) == pytest.approx(19 / 28)
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert sign_flip_exact([1, 1, 1, 1, 1]) == pytest.approx(2 / 32)
    r = paired_test(np.arange(25) + 1.0, np.arange(25))
    assert r["test"] == "wilcoxon" and r["p"] < 1e-4 and r["mean_diff"] == 1.0
