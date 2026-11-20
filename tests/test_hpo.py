import json

from pianob.data.adapters import FacedAdapter
from pianob.experiments.jobs import Context
from pianob.experiments.queue import Job
from pianob.experiments.registry import resolve_config
from pianob.hpo.optuna_runner import fail_stale_trials, run_hpo_job


def test_hpo_resumable_and_best_params_used(tmp_path, monkeypatch):
    monkeypatch.setenv("PIANOB_WORKING", str(tmp_path))
    ctx = Context(FacedAdapter(synthetic=True).load(), device="cpu", log=lambda m: None)
    job = Job("hpo", "eegnet_ea", n_trials=2, overrides={"max_subjects": 2, "hparams": {"epochs": 1}})
    assert run_hpo_job(job, ctx) == "completed"
    best = json.loads((tmp_path / "pianob_state" / "hpo_best" / "eegnet_ea.json").read_text())
    assert best["n_complete"] == 2 and "window_s" in best["params"]
    cfg, src = resolve_config("eegnet_ea")
    assert src.startswith("optuna:eegnet_ea") and cfg.window_s == best["params"]["window_s"]
    # ripresa: lo stesso studio con target 3 esegue un solo trial in più
    fail_stale_trials()
    job3 = Job("hpo", "eegnet_ea", n_trials=3, overrides=job.overrides)
    run_hpo_job(job3, ctx)
    best3 = json.loads((tmp_path / "pianob_state" / "hpo_best" / "eegnet_ea.json").read_text())
    assert best3["n_complete"] == 3
