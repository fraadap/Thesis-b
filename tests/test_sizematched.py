"""Controllo quantità di dati vs novità dello stimolo (split_fm_sizematched / lnso10_sizematched)."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from pianob.data import faced_meta as FM
from pianob.data.adapters import FacedAdapter
from pianob.data.splits import check_sizematched, coverage, make_split
from pianob.experiments.jobs import Context, build_banks, prepare, run_job
from pianob.experiments.queue import Job
from pianob.experiments.registry import get_config


@pytest.mark.parametrize("protocol,folds", [("split_fm_sizematched", range(3)), ("lnso10_sizematched", [0, 7, 29])])
def test_sizematched_assertions(protocol, folds):
    lab = np.array(FM.CLIP_LABELS)
    for k in folds:
        sp = make_split(protocol, k, draw=0)
        lco = make_split(protocol.replace("_sizematched", "_lco"), k)
        assert (sp.train_subjects, sp.val_subjects, sp.test_subjects) == \
               (lco.train_subjects, lco.val_subjects, lco.test_subjects)
        assert sp.heldout_clips == lco.unseen_clips and sp.n_train_trials() == lco.n_train_trials()
        want = sorted(lab[lco.train_clips])
        for s in sp.train_subjects:
            assert sorted(lab[sp.train_clips_per_subject[s]]) == want
        cov = coverage(sp)
        assert min(cov.values()) >= 1
        assert 0.5 < np.mean(list(cov.values())) / len(sp.train_subjects) < 0.8   # ~2/3
        subsets = {tuple(v) for v in sp.train_clips_per_subject.values()}
        assert len(subsets) > len(sp.train_subjects) // 2                       # diversi per soggetto
        assert sp.eval_conditions() == {f"tk{sp.clip_fold}": lco.unseen_clips}


def test_draws_are_deterministic_and_different():
    a = make_split("split_fm_sizematched", 1, draw=0).train_clips_per_subject
    b = make_split("split_fm_sizematched", 1, draw=0).train_clips_per_subject
    c = make_split("split_fm_sizematched", 1, draw=1).train_clips_per_subject
    assert a == b and a != c


def test_broken_composition_is_detected():
    sp = make_split("split_fm_sizematched", 0)
    lco = make_split("split_fm_lco", 0)
    bad = copy.deepcopy(sp)
    s0 = bad.train_subjects[0]
    lab = np.array(FM.CLIP_LABELS)
    cl = bad.train_clips_per_subject[s0]
    other = next(c for c in range(28) if c not in cl and lab[c] != lab[cl[0]])
    bad.train_clips_per_subject[s0] = sorted(cl[1:] + [other])   # stessa numerosità, composizione diversa
    with pytest.raises(AssertionError):
        check_sizematched(bad, lco)
    bad = copy.deepcopy(sp)
    tk0 = sp.heldout_clips[0]
    for s in bad.train_subjects:  # T_k mai vista
        if tk0 in bad.train_clips_per_subject[s]:
            bad.train_clips_per_subject[s] = [c for c in lco.train_clips]
    with pytest.raises(AssertionError):
        check_sizematched(bad, lco)


@pytest.fixture()
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("PIANOB_WORKING", str(tmp_path))
    return Context(FacedAdapter(synthetic=True).load(), device="cpu", log=lambda m: None)


@pytest.mark.parametrize("config", ["eegnet_ea", "eegnet_noalign"])
def test_same_train_windows_and_identical_test_examples(ctx, config):
    cfg = get_config(config)
    win = int(cfg.window_s * 200)
    k = 2
    out = {}
    for proto, draw in (("split_fm_lco", None), ("split_fm_sizematched", 0)):
        job = Job("train", config, proto, k, 0, draw=draw, overrides={"max_subjects": 3})
        jd = prepare(job, ctx, cfg)
        out[proto] = (jd, build_banks(job, ctx, jd, win))
    (jl, bl), (js, bs) = out["split_fm_lco"], out["split_fm_sizematched"]
    assert bl["train"].n == bs["train"].n
    ml = np.isin(bl["test"].ws.clip, jl.split.eval_conditions()["unseen"])
    ms = np.isin(bs["test"].ws.clip, js.split.eval_conditions()[f"tk{k}"])
    for attr in ("subject", "clip", "label"):
        assert np.array_equal(getattr(bl["test"].ws, attr)[ml], getattr(bs["test"].ws, attr)[ms])
    assert np.array_equal(bl["test"].ws.win_start[ml], bs["test"].ws.win_start[ms])
    xl, _ = bl["test"].get(np.where(ml)[0], "cpu")
    xs, _ = bs["test"].get(np.where(ms)[0], "cpu")
    assert torch.equal(xl, xs)   # stessi esempi, stesso allineamento


def test_decomposition_end_to_end(ctx, tmp_path):
    from pianob import report
    from pianob.eval.shortcuts import stim_decomposition
    sm = {"max_subjects": 4, "hparams": {"epochs": 1, "patience": 0}}   # 2 soggetti di test
    jobs = [Job("train", "eegnet_noalign", "split_fm", 0, 0, overrides=sm)]
    jobs += [Job("train", "eegnet_noalign", "split_fm_lco", k, 0, overrides=sm) for k in range(3)]
    jobs += [Job("train", "eegnet_noalign", "split_fm_sizematched", k, 0, draw=0, overrides=sm) for k in range(3)]
    jobs += [Job("train", "eegnet_noalign", "split_fm_sizematched", k, 0, draw=0, perm=0, overrides=sm) for k in range(3)]
    for j in jobs:
        assert run_job(j, ctx) == "completed"
    df = report.load_rows()
    assert {"tk0", "tk1", "tk2"} <= set(df[df.protocol == "split_fm"].test_condition)
    assert set(df[df.protocol == "split_fm_sizematched"].variant) == {"draw0", "draw0_perm0"}
    assert (df[df.protocol == "split_fm_sizematched"].draw == 0).all()
    dec = stim_decomposition(df, "eegnet_noalign", "split_fm")
    assert dec["folds_k"] == [0, 1, 2]
    for key in ("total", "data_quantity", "dstim_size", "dstim_size_corr"):
        assert key in dec and "ci95" in dec[key]
    t = np.array(dec["total"]["per_subject"])
    parts = np.array(dec["data_quantity"]["per_subject"]) + np.array(dec["dstim_size"]["per_subject"])
    assert np.allclose(t, parts, atol=1e-3)   # identità della scomposizione


def test_tk_rows_from_preds_match_job_rows(ctx, tmp_path):
    from pianob.eval.shortcuts import tk_rows_from_preds
    job = Job("train", "eegnet_noalign", "split_fm", 0, 0, overrides={"max_subjects": 2, "hparams": {"epochs": 1}})
    run_job(job, ctx)
    import json
    rows = [json.loads(l) for l in (tmp_path / "pianob_state" / "results" / "rows_w0.jsonl").read_text().splitlines()]
    with np.load(tmp_path / "pianob_state" / "artifacts" / "preds" / f"{job.job_id}.npz") as z:
        rp = tk_rows_from_preds(job.job_id, z)
    a = pd.DataFrame([r for r in rows if r["test_condition"].startswith("tk")]).set_index(["test_condition", "subject"])
    b = pd.DataFrame(rp).set_index(["test_condition", "subject"])
    assert np.allclose(a.loc[b.index, "ba_win"], b.ba_win, atol=2e-3)   # probs salvate in float16


def test_music_hook():
    from pianob.data.adapters import MusinGAdapter, NmedTAdapter
    from pianob.data.splits import stimulus_folds, stimulus_holdout_plan
    for n, h in ((12, MusinGAdapter.N_HELDOUT_STIMULI), (10, NmedTAdapter.N_HELDOUT_STIMULI)):
        folds = stimulus_folds(list(range(n)), h)
        assert sorted(sum(folds, [])) == list(range(n))
        p = stimulus_holdout_plan(list(range(20)), list(range(n)), folds[0])
        assert all(len(v) == n - h for v in p["per_subject"].values())
        assert set(folds[0]) <= set().union(*p["per_subject"].values())
