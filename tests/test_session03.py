"""Sessione 03 (label da self-report): tabella dei voti, metriche, nulli, anti-leakage, pipeline (§8 del prompt)."""
import json

import numpy as np
import pandas as pd
import pytest

from pianob.data import faced_meta as FM
from pianob.data.splits import make_split
from pianob.selfreport import analysis as SA
from pianob.selfreport import metrics as M
from pianob.selfreport import table as T
from pianob.selfreport import targets as G


@pytest.fixture(scope="module")
def votes():
    df, _, meta = T.build_table(T.synthetic_parsed(), log=lambda *a: None)
    return df, meta


def test_reordering_recovers_canonical_index_and_position():
    parsed = T.synthetic_parsed(n_subjects=30, presentation_rows=True)
    df, _, meta = T.build_table(parsed, log=lambda *a: None)
    assert meta["alignment"]["alignment"] == "vid"
    assert meta["alignment"]["concordance_vid"] - meta["alignment"]["concordance_canonical"] > 0.2
    for s in (0, 7):
        p = parsed[s]
        for k, vid in enumerate(p["vid"]):           # righe in ordine di presentazione: posizione = k + 1
            row = df[(df.subject == s) & (df["clip"] == vid - 1)].iloc[0]
            assert row.trial_position == k + 1
            assert row.valence == p["score"][k, T.ITEMS.index("valence")]
    # righe già in ordine canonico (vid identità): stesso risultato senza ambiguità
    df2, _, meta2 = T.build_table(T.synthetic_parsed(n_subjects=30, presentation_rows=False), log=lambda *a: None)
    assert meta2["alignment"]["vid_identity_everywhere"]


def test_missing_order_fields_stop():
    with pytest.raises(T.SelfReportError):
        T.parse_after_remarks({"After_remarks": {"score": np.zeros((28, 12)), "trial": np.arange(28)}})
    with pytest.raises(T.SelfReportError):
        T.parse_after_remarks({"After_remarks": {"score": np.zeros((28, 11)), "trial": np.arange(28),
                                                 "vid": np.arange(1, 29)}})


def test_struct_array_format_is_parsed():
    rem = [{"score": np.arange(12.0), "trial": k + 1, "vid": 28 - k} for k in range(28)]
    out = T.parse_after_remarks({"After_remarks": rem})
    assert out["score"].shape == (28, 12) and out["vid"][0] == 28


def test_indistinguishable_alignment_stops():
    rng = np.random.default_rng(0)
    parsed = {s: {"score": rng.integers(0, 8, (28, 12)).astype(float), "trial": np.arange(1, 29),
                  "vid": rng.permutation(28) + 1, "extra": {}, "fields": {}} for s in range(20)}
    with pytest.raises(T.SelfReportError, match="indeciso"):
        T.build_table(parsed, log=lambda *a: None)


def test_z_and_bin_per_subject(votes):
    df, meta = votes
    g = df.groupby("subject")
    assert np.allclose(g["valence_z"].mean(), 0, atol=1e-9) and np.allclose(g["valence_z"].std(ddof=0), 1)
    assert (g["valence_bin"].sum() == 14).all() and (g["arousal_bin"].sum() == 14).all()
    assert meta["concordance"]["mean"] > 0.5


def test_hash_deterministic(votes, tmp_path):
    df, _, meta = T.build_table(T.synthetic_parsed(), log=lambda *a: None)
    m1 = T.write_outputs(df, pd.DataFrame(), dict(meta), tmp_path / "a", log=lambda *a: None)
    df2, _, meta2 = T.build_table(T.synthetic_parsed(), log=lambda *a: None)
    m2 = T.write_outputs(df2, pd.DataFrame(), dict(meta2), tmp_path / "b", log=lambda *a: None)
    assert m1["sha256"] == m2["sha256"] == T.canonical_hash(pd.read_parquet(tmp_path / "a" / "selfreport.parquet"))


def test_residual_permutation_preserves_means_and_depends_on_seed(votes):
    df, _ = votes
    Z = G.matrix(df, "valence_z", 123)
    sp = make_split("split_fm_lco", 0, list(range(28)), list(range(123)), clip_labels=FM.CLIP_LABELS)
    tr, c = sp.train_subjects, sp.train_clips
    a, b = G.permute_train_residuals(Z, sp, 0), G.permute_train_residuals(Z, sp, 1)
    B = Z[np.ix_(tr, c)]
    for P in (a, b):
        Pb = P[np.ix_(tr, c)]
        assert np.allclose(Pb.mean(0), B.mean(0)) and np.allclose(Pb.mean(1), B.mean(1))
    assert not np.allclose(a, b)
    Zp = M.permute_resid_within_clip(Z[:40], np.random.default_rng(0), 2)
    assert np.allclose(Zp.mean(-1), Z[:40].mean(-1)) and np.allclose(Zp.mean(-2), Z[:40].mean(-2))


def test_metrics_reference_predictors():
    rng = np.random.default_rng(1)
    S, C = 40, 28
    Z = rng.normal(0, 1, C) + rng.normal(0, 1, (S, 1)) + rng.normal(0, 1, (S, C))
    mem = np.broadcast_to(Z.mean(0), Z.shape) + rng.normal(0, 3, (S, 1))      # costante entro clip + offset
    m = M.all_metrics(mem, Z)
    assert abs(m["resid_corr"]) < 1e-9 and abs(m["within_clip"]) < 1e-9 and m["between_clip"] > 0.99
    assert all(abs(v - 1) < 1e-5 for v in M.all_metrics(Z, Z).values())
    P = 0.3 * Z + rng.normal(0, 1, (S, C))
    off = M.all_metrics(P + rng.normal(0, 10, (S, 1)), Z)
    base = M.all_metrics(P, Z)
    assert abs(off["resid_corr"] - base["resid_corr"]) < 1e-9 and abs(off["within_clip"] - base["within_clip"]) < 1e-9


def test_null_is_calibrated():
    """Voto = clip + soggetto + rumore, predizione = clip + soggetto + rumore indipendente: ~5% di p < 0.05."""
    rng = np.random.default_rng(2)
    S, C, R = 23, 28, 200
    rej = {"resid_corr": 0, "within_clip": 0}
    for r in range(R):
        clip, subj = rng.normal(0, 1, C), rng.normal(0, 1, (S, 1))
        Z = clip + subj + rng.normal(0, 1, (S, C))
        P = clip + rng.normal(0, 1, (S, 1)) + rng.normal(0, 1, (S, C))
        res = M.perm_test(P, Z, n=199, seed=r)
        for k in rej:
            rej[k] += res[k]["p"] < 0.05
    for k, v in rej.items():
        assert 0.01 <= v / R <= 0.10, (k, v / R)


def test_fold_union_unseen_exactly_once():
    recs = [{"fold": f, "subject": np.array([0, 1]), "clip": np.array([f, f]), "cond": np.array(["unseen"] * 2),
             "p": np.array([0.1, 0.2])} for f in range(3)]
    out = SA.merge_folds(recs)
    assert len(out["unseen"]) == 6
    recs.append({"fold": 3, "subject": np.array([0]), "clip": np.array([0]), "cond": np.array(["unseen"]),
                 "p": np.array([0.3])})
    with pytest.raises(AssertionError):
        SA.merge_folds(recs)
    seen = [{"fold": f, "subject": np.array([0]), "clip": np.array([5]), "cond": np.array(["seen"]),
             "p": np.array([float(f)])} for f in range(2)]
    m = SA.merge_folds(seen)
    assert m["seen"].v.iloc[0] == 0.5 and len(m["seen_single"]) == 1


def test_no_leakage_in_residual_targets(votes):
    df, _ = votes
    sp = make_split("split_fm_lco", 1, list(range(28)), list(range(123)), clip_labels=FM.CLIP_LABELS)
    Z = G.matrix(df, "valence_z", 123)
    lab, r = G.res_labels(Z, sp)
    # cambiare i voti dei soggetti di test non cambia le label dei soggetti di training
    Z2 = Z.copy()
    Z2[sp.test_subjects] += 5.0
    lab2, _ = G.res_labels(Z2, sp)
    assert (lab2[sp.train_subjects] == lab[sp.train_subjects]).all()
    # nessun voto delle clip T_k (dei soggetti di training) entra nelle label di training
    Z3 = Z.copy()
    Z3[np.ix_(sp.train_subjects, sp.unseen_clips)] += 5.0
    lab3, _ = G.res_labels(Z3, sp)
    assert (lab3[np.ix_(sp.train_subjects, sp.train_clips)] == lab[np.ix_(sp.train_subjects, sp.train_clips)]).all()
    assert (lab[np.ix_(sp.train_subjects, sp.unseen_clips)] == -1).all()
    # leave-one-out: la media della clip per un soggetto di training non contiene il suo voto
    s, c = sp.train_subjects[0], sp.train_clips[0]
    others = [t for t in sp.train_subjects if t != s]
    assert np.isclose(r[s, c], Z[s, c] - Z[others, c].mean())


def test_selfreport_feature_job_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("PIANOB_WORKING", str(tmp_path))
    from pianob.data.adapters import FacedAdapter
    from pianob.experiments.jobs import Context, run_job
    from pianob.experiments.queue import Job
    ctx = Context(FacedAdapter(synthetic=True).load(), device="cpu", log=lambda m: None)
    job = Job("train", "de_asym_lr_zsub", "split_fm_lco", 0, 0, task="sr_val_bin", n_perm=2,
              overrides={"max_subjects": 2})
    assert run_job(job, ctx) == "completed"
    d = np.load(tmp_path / "pianob_state" / "artifacts" / "srpreds" / f"{job.job_id}.npz")
    assert set(d["cond"]) == {"seen", "unseen"} and d["perm_p"].shape == (len(d["p"]), 2)
    rows = [json.loads(l) for l in (tmp_path / "pianob_state" / "results" / "rows_w0.jsonl").read_text().splitlines()]
    assert {r["task"] for r in rows} == {"sr_val_bin"} and {r["variant"] for r in rows} == {"base", "perm0", "perm1"}


def test_queues_session03():
    from pianob.experiments.queue import QUEUES
    H, I = QUEUES["H"](), QUEUES["I"]()
    assert len({j.job_id for j in H}) == len(H) and len({j.job_id for j in I}) == len(I)
    # priorità: i job della terza famiglia (DE + z-score, lnso10_lco) all'inizio della coda H
    first = H[:60]
    assert all(j.config == "de_asym_lr_zsub" and j.protocol == "lnso10_lco" and j.n_perm == 50 for j in first)
    assert all(j.overrides.get("needs_gate_r") for j in I if j.config == "reve_base")
    assert all(not j.overrides.get("needs_gate_r") for j in I if j.config != "reve_base")


def test_faced_mat_layout_after_remark_singular():
    """Formato reale dei .mat di FACED: variabile `After_remark`, struct 1 x 28, Accuracy vuoto in molti trial."""
    rng = np.random.default_rng(0)
    rem = [{"score": rng.uniform(0, 7, 12), "trial": k + 1, "vid": int(v), "Accuracy": np.array([]) if k % 5 else 0.9,
            "ResponseTime": np.array([]) if k % 5 else rng.uniform(0, 4, 20)} for k, v in enumerate(rng.permutation(28) + 1)]
    out = T.parse_after_remarks({"__header__": b"", "After_remark": rem})
    assert out["score"].shape == (28, 12) and out["trial"].tolist() == list(range(1, 29))
    assert out["extra"]["ResponseTime"].shape == (120,) and "irregolari" in out["fields"]["Accuracy"]
