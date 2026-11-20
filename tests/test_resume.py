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
