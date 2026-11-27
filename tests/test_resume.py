"""Interruzione per la guardia del tempo -> checkpoint -> ripresa; recupero dello stato dagli input."""
import json

import numpy as np
import torch

from pianob import kaggle_env
from pianob.models.eegnet import EEGNet
from pianob.train.loop import TrialBank, train_model


class StopAfter:
    def __init__(self, n):
        self.n, self.calls = n, 0

    def must_stop(self):
        self.calls += 1
        return self.calls >= self.n


def _banks():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((12, 4, 800)).astype(np.float32)
    lab = np.arange(12) % 2
    data[lab == 1, 0] += 2.0
    s = np.repeat([0, 1, 2], 4)
    c = np.tile(np.arange(4), 3)
    return TrialBank(data, s, c, lab, 200), TrialBank(data, s, c, lab, 200)


def test_interrupt_and_resume(tmp_path):
    hp = {"epochs": 4, "batch_size": 8, "lr": 1e-3, "weight_decay": 0.0, "label_smoothing": 0.0,
          "class_weighted": True, "clip_grad": 1.0, "patience": 0, "select_metric": "ba_subject_mean",
          "scheduler": "cosine_step", "amp": False}
    ck = tmp_path / "job.pt"
    tr, va = _banks()
    torch.manual_seed(0)
    m = EEGNet(4, 200, 2, fs=100)
    r1 = train_model(m, hp, "default", tr, va, 2, "cpu", seed=0, ckpt_path=ck, guard=StopAfter(2), log=lambda x: None)
    assert r1["status"] == "interrupted" and ck.exists() and len(r1["history"]) == 2
    msgs = []
    m2 = EEGNet(4, 200, 2, fs=100)
    r2 = train_model(m2, hp, "default", tr, va, 2, "cpu", seed=0, ckpt_path=ck, guard=None, log=msgs.append)
    assert r2["status"] == "completed"
    assert any("ripresa" in x for x in msgs)
    assert [h["epoch"] for h in r2["history"]] == [0, 1, 2, 3]


def test_restore_state_from_previous_output(tmp_path, monkeypatch):
    prev = tmp_path / "input" / "prev-notebook" / "pianob_state"
    (prev / "results").mkdir(parents=True)
    (prev / "done").mkdir()
    (prev / "optuna").mkdir()
    (prev / "results" / "rows_w0.jsonl").write_text('{"a": 1}\n')
    (prev / "done" / "done_w0.jsonl").write_text(json.dumps({"job_id": "x", "status": "completed"}) + "\n")
    (prev / "optuna" / "s.db").write_bytes(b"db")
    work = tmp_path / "working"
    monkeypatch.setenv("PIANOB_WORKING", str(work))
    monkeypatch.setenv("PIANOB_INPUT", str(tmp_path / "input"))
    st = kaggle_env.state_dir()
    (st / "results" / "rows_w0.jsonl").write_text('{"b": 2}\n')   # righe già presenti: vanno unite
    kaggle_env.restore_state_from_inputs(verbose=False)
    assert (st / "optuna" / "s.db").read_bytes() == b"db"
    assert set((st / "results" / "rows_w0.jsonl").read_text().split()) == {'{"a":', '1}', '{"b":', '2}'}
    from pianob.experiments.queue import done_status
    assert done_status()["x"]["status"] == "completed"


def test_nan_data_fails_loudly(tmp_path):
    import pytest
    hp = {"epochs": 2, "batch_size": 8, "lr": 1e-3, "weight_decay": 0.0, "label_smoothing": 0.0,
          "class_weighted": True, "clip_grad": 1.0, "patience": 0, "select_metric": "ba_subject_mean",
          "scheduler": "cosine_step", "amp": False}
    rng = np.random.default_rng(0)
    data = rng.standard_normal((12, 4, 800)).astype(np.float32)
    data[3, 1, 100] = np.inf
    s, c, lab = np.repeat([0, 1, 2], 4), np.tile(np.arange(4), 3), np.arange(12) % 2
    tr = TrialBank(data, s, c, lab, 200)
    with pytest.raises(FloatingPointError):
        train_model(EEGNet(4, 200, 2, fs=100), hp, "default", tr, tr, 2, "cpu", seed=0, log=lambda x: None)


def test_invalidated_jobs_rerun_and_rows_dropped(tmp_path, monkeypatch):
    import pandas as pd
    from pianob.experiments import queue as Q
    monkeypatch.setenv("PIANOB_WORKING", str(tmp_path))
    jid = "tsception_ea__split_fm__f0__s0__emo9__bp4-30"
    cut = Q.INVALIDATED[jid]["cutoff"]
    d = kaggle_env.state_dir() / "done" / "done_w0.jsonl"
    d.write_text(json.dumps({"job_id": jid, "status": "early_stopped", "t": cut - 100}) + "\n")
    assert jid not in Q.done_status()                       # record vecchio ignorato -> si riesegue
    with open(d, "a") as fh:
        fh.write(json.dumps({"job_id": jid, "status": "completed", "t": cut + 100}) + "\n")
    assert Q.done_status()[jid]["status"] == "completed"    # la nuova esecuzione conta
    base = dict(protocol="split_fm", fold=0, seed=0, task="emo9", subject=100, ba_win=0.1)
    df = pd.DataFrame([{**base, "config": "tsception_ea", "variant": "bp4-30"},
                       {**base, "config": "tsception_ea_bnalpha", "variant": "bp4-30_bnalpha_session_n00"},
                       {**base, "config": "tsception_ea", "variant": "bp4-30", "t": cut + 5},
                       {**base, "config": "tsception_ea", "variant": "bp4-30_noedge"},
                       {**base, "config": "tsception_ea", "variant": "base"}])
    assert Q.invalid_row_mask(df).tolist() == [True, True, False, False, False]


def test_restore_skips_partial_and_large_models(tmp_path, monkeypatch):
    """Notebook G (disco pieno): la ripresa non copia checkpoint parziali né i modelli grandi."""
    from pianob import kaggle_env as K
    src = tmp_path / "inp" / "out" / K.STATE_DIRNAME
    (src / "ckpt").mkdir(parents=True)
    (src / "models").mkdir()
    (src / "ckpt" / "a.pt.tmp").write_bytes(b"x")
    (src / "ckpt" / "b.pt").write_bytes(b"x")
    (src / "models" / "small.pt").write_bytes(b"x")
    big = src / "models" / "big.pt"
    with open(big, "wb") as fh:
        fh.truncate(K.RESTORE_MODEL_MAX_BYTES + 1)
    monkeypatch.setenv("PIANOB_WORKING", str(tmp_path / "work"))
    monkeypatch.setenv("PIANOB_INPUT", str(tmp_path / "inp"))
    K.restore_state_from_inputs(verbose=False)
    dst = K.state_dir()
    assert (dst / "ckpt" / "b.pt").exists() and not (dst / "ckpt" / "a.pt.tmp").exists()
    assert (dst / "models" / "small.pt").exists() and not (dst / "models" / "big.pt").exists()
    assert K.disk_status()["used_gb"] < 1
