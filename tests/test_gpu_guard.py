import pytest

from pianob import kaggle_env, run


def test_queue_stops_without_gpu_on_kaggle(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PIANOB_WORKING", str(tmp_path))
    monkeypatch.delenv("PIANOB_ALLOW_CPU", raising=False)
    monkeypatch.setattr(kaggle_env, "on_kaggle", lambda: True)
    monkeypatch.setattr(run, "_n_gpus", lambda: 0)
    with pytest.raises(SystemExit) as e:
        run.main(["--queue", "queues/nbB_cbramod_gate.json", "--gpus", "auto", "--skip-bench"])
    assert e.value.code == 2 and "nessuna GPU" in capsys.readouterr().out


def test_smoke_queue_allowed_on_cpu():
    from pianob.experiments.queue import load_queue
    assert not any(run._needs_gpu(j) for j in load_queue("queues/nb0_smoke.json"))
    assert all(run._needs_gpu(j) for j in load_queue("queues/nbB_cbramod_gate.json"))
