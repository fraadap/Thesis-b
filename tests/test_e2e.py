"""Mini end-to-end su dati sintetici a 32 canali (CPU): TSception con EA e BN-α, CBraMod a pesi casuali."""
import json

import numpy as np
import pytest
import torch

from pianob.data.adapters import FacedAdapter
from pianob.experiments.jobs import Context, run_job
from pianob.experiments.queue import Job
from pianob.models.cbramod import CBraModClassifier


@pytest.fixture()
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("PIANOB_WORKING", str(tmp_path))
    return Context(FacedAdapter(synthetic=True).load(), device="cpu", log=lambda m: None)


def _rows(tmp_path):
    f = tmp_path / "pianob_state" / "results" / "rows_w0.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines()]


def test_tsception_ea_lco_end_to_end(ctx, tmp_path):
    job = Job("train", "tsception_ea", "split_fm_lco", 1, 0,
              overrides={"max_subjects": 2, "hparams": {"epochs": 1, "patience": 0, "batch_size": 32}})
    assert run_job(job, ctx) == "completed"
    rows = _rows(tmp_path)
    base = [r for r in rows if r["config"] == "tsception_ea"]
    assert {r["test_condition"] for r in base} == {"seen", "unseen"}
    for k in ("ba_win", "ba_trial", "kappa", "f1w", "f1m", "n_windows", "n_trials", "subject", "seed", "fold"):
        assert k in base[0]
    bn = {r["variant"] for r in rows if r["config"] == "tsception_ea_bnalpha"}
    assert "bnalpha_session_n00" in bn and "bnalpha_stream_n032" in bn and any(v.endswith("n0sel") for v in bn)
    # finestre unseen: 9 clip non viste nel fold 1 x (30 s / finestra) finestre (finestra da Optuna se presente)
    from pianob.experiments.registry import resolve_config
    w = resolve_config("tsception_ea")[0].window_s
    unseen = [r for r in base if r["test_condition"] == "unseen"][0]
    assert unseen["n_trials"] == 9 and unseen["n_windows"] == 9 * int(30 // w)


def test_cbramod_random_weights_end_to_end(ctx, tmp_path):
    job = Job("train", "cbramod_scratch_ft", "split_fm", 0, 0,
              overrides={"max_subjects": 2, "hparams": {"epochs": 1, "n_layer": 2, "head": "avgpooling_patch_reps"}})
    assert run_job(job, ctx) == "completed"
    rows = _rows(tmp_path)
    assert rows and rows[0]["n_windows"] == 28 * 3


def test_feature_baseline_end_to_end(ctx, tmp_path):
    job = Job("train", "de_asym_lr_zsub", "split_fm_lco", 0, 0, n_perm=2, overrides={"max_subjects": 2})
    assert run_job(job, ctx) == "completed"
    rows = _rows(tmp_path)
    assert {r["variant"] for r in rows} == {"base", "perm0", "perm1"}


def test_official_head_shape_and_reshape_equivalence():
    m = CBraModClassifier(32, 2000, 9, head="all_patch_reps", n_layer=1)
    assert m.classifier[1].in_features == 32 * 10 * 200
    with torch.no_grad():
        out = m(torch.randn(2, 32, 2000))
    assert out.shape == (2, 9)
    f = torch.randn(2, 32, 10, 200)
    try:
        from einops import rearrange
        assert torch.equal(rearrange(f, "b c s d -> b (c s d)"), f.flatten(1))
    except ImportError:
        pass


def test_synthetic_trials_shape_and_channels():
    ad = FacedAdapter(synthetic=True).load()
    x = ad.trials([0, 5], [0, 27])
    assert x.shape == (2, 2, 32, 6000) and x.dtype == np.float32
    assert ad.channel_indices(["T3", "T4"]) == ad.channel_indices(["T7", "T8"])
