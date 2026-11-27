"""Sessione 02: teste a pooling, standardizzazione per soggetto, ricetta a due fasi, REVE (CPU, piccoli)."""
import json

import numpy as np
import pytest
import torch

from pianob.models.cbramod import CBraModClassifier
from pianob.train.loop import TrialBank, compute_subject_stats
from pianob.train.twophase import make_phase_optimizer, set_phase, train_mode


def _model(head, norm="none", C=4):
    return CBraModClassifier(C, 400, 3, head=head, dropout=0.1, n_layer=1, subj_embed_norm=norm)


@pytest.mark.parametrize("head,dim", [("flat_official", None), ("pool_channel", 4 * 200), ("pool_global", 200)])
def test_head_shapes(head, dim):
    m = _model(head).eval()
    x = torch.randn(5, 4, 400)
    assert m(x).shape == (5, 3)
    if dim:
        assert m.pooled_embedding(x).shape == (5, dim)
        assert m.classifier[0].in_features == dim and m.classifier[0].out_features == 256


def test_lp_ft_trainable_and_groups():
    m = _model("pool_global")
    hp = {"lr_head_lp": 1e-3, "lr_backbone_ft": 2e-5, "lr_head_ft": 4e-4, "lp_epochs": 2, "ft_epochs": 3,
          "warmup_frac": 0.2, "weight_decay": 0.05}
    set_phase(m, "lp")
    train_mode(m, "lp")
    assert all(not p.requires_grad for n, p in m.named_parameters() if n.startswith("backbone."))
    assert all(p.requires_grad for n, p in m.named_parameters() if not n.startswith("backbone."))
    assert not m.backbone.training and m.classifier.training
    set_phase(m, "ft")
    assert all(p.requires_grad for p in m.parameters())
    opt, _ = make_phase_optimizer(m, hp, "ft", steps_per_epoch=10)
    assert len(opt.param_groups) == 2
    assert {id(p) for p in opt.param_groups[0]["params"]} == {id(p) for n, p in m.named_parameters() if n.startswith("backbone.")}


def test_one_cycle_peak_and_end():
    m = _model("pool_global")
    hp = {"lr_head_lp": 1e-3, "lp_epochs": 10, "warmup_frac": 0.2, "weight_decay": 0.05}
    set_phase(m, "lp")
    opt, sched = make_phase_optimizer(m, hp, "lp", steps_per_epoch=10)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert int(np.argmax(lrs)) in (19, 20) and abs(max(lrs) - 1e-3) < 1e-9
    assert lrs[-1] < 1e-3 * 1e-3


def _bank(C=4, T=400, subjects=(5, 9), clips=(0, 1, 2)):
    rng = np.random.default_rng(0)
    n = len(subjects) * len(clips)
    data = rng.normal(size=(n, C, T * 2)).astype(np.float32)
    subj = np.repeat(subjects, len(clips))
    clip = np.tile(clips, len(subjects))
    return TrialBank(data, subj, clip, clip % 3, T, 0.0, "cpu")


def test_session_stats_label_free_and_standardized():
    m = _model("pool_channel", "session").eval()
    bank = _bank()
    lab_before = bank.win_label.clone()
    bank.win_label[:] = 0                      # le etichette non devono servire
    info = compute_subject_stats(m, bank, "cpu", amp=False)
    bank.win_label[:] = lab_before
    assert info["subjects"] == [5, 9]
    with torch.no_grad():
        for s in (5, 9):
            idx = np.where(bank.ws.subject == s)[0]
            x, _ = bank.get(idx, "cpu")
            z = m.norm(m.pooled_embedding(x), torch.full((len(idx),), s))
            assert torch.allclose(z.mean(0), torch.zeros(z.shape[1]), atol=1e-4)
            assert torch.allclose(z.std(0, unbiased=False), torch.ones(z.shape[1]), atol=1e-3)


def test_heldout_mask_excludes_test_fold_windows():
    m = _model("pool_global", "heldout").eval()
    bank = _bank(clips=(0, 1, 2, 3))
    mask = np.isin(bank.ws.clip, [0, 1])       # clip 2 e 3 = fold di test
    info = compute_subject_stats(m, bank, "cpu", mask=mask, amp=False)
    assert sum(info["n_windows"].values()) == int(mask.sum())
    with torch.no_grad():
        idx = np.where(mask & (bank.ws.subject == 5))[0]
        e = m.pooled_embedding(bank.get(idx, "cpu")[0]).double()
    assert torch.allclose(m.norm.mean[5].double(), e.mean(0), atol=1e-4)


def test_stats_recomputed_each_ft_epoch_and_before_validation(monkeypatch):
    import pianob.train.twophase as tp
    calls = []
    orig = tp.compute_subject_stats
    monkeypatch.setattr(tp, "compute_subject_stats", lambda model, bank, *a, **k: (calls.append(bank), orig(model, bank, *a, **k))[1])
    m = _model("pool_global", "session")
    tr, va = _bank(subjects=(1, 2)), _bank(subjects=(3,))
    hp = {"lp_epochs": 1, "ft_epochs": 2, "batch_size": 4, "amp": False, "patience": 0}
    res = tp.train_twophase(m, hp, tr, va, 3, "cpu", seed=0, log=lambda *_: None)
    # LP: train + val una volta; FT: per epoca train (inizio) + val (prima della validation)
    assert [b is tr for b in calls] == [True, False, True, False, True, False]
    assert res["phase_epochs"] == {"lp": 1, "ft": 2}


def test_trial_prediction_uses_log_probabilities():
    from pianob.eval.metrics import evaluate_predictions
    # due finestre dello stesso trial: media delle probabilità -> classe 0, media delle log-prob -> classe 1
    P = np.array([[0.9, 0.05, 0.05], [0.001, 0.5, 0.499]])
    ev = evaluate_predictions(P, [0, 0], [0, 0], [1, 1])
    assert ev["per_subject"][0]["ba_trial"] == 0.0 and ev["per_subject"][0]["ba_trial_lp"] == 1.0


def test_reve_shapes_params_and_positions():
    from pianob.data.faced_meta import CHANNELS
    from pianob.models.reve import REVE_NAME, positions_for
    from pianob.models.reve.model import ReveClassifier
    assert REVE_NAME["A1"] == "A2" and REVE_NAME["A2"] == "A1" and REVE_NAME["T7"] == "T3"
    pos = positions_for(CHANNELS)
    assert pos.shape == (32, 3)
    m = ReveClassifier(pos, 2000, 9).eval()
    assert m.n_tokens == 352 and m.out_shape == 180736
    m.set_phase("lp")
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 1807881
    with torch.no_grad():
        assert m.tokens(torch.randn(1, 32, 2000)).shape == (1, 352, 512)
    keep = [c for c in CHANNELS if c not in ("FP1", "FP2", "F7", "F8")]        # rimozione vera dei canali
    assert positions_for(keep).shape == (28, 3)


def test_reve_warmup_and_stable_adamw():
    from pianob.train.reve_recipe import exponential_warmup
    from pianob.train.stable_adamw import StableAdamW
    f = exponential_warmup(100)
    assert f(0) == 0.0 and abs(f(50) - (10 ** 0.5 - 1) / 9) < 1e-12 and f(100) == 1.0
    p = torch.nn.Parameter(torch.ones(3))
    opt = StableAdamW([p], lr=0.1, betas=(0.92, 0.999), weight_decay=0.0, eps=1e-9)
    p.grad = torch.tensor([1.0, -1.0, 0.5])
    opt.step()
    # primo passo: m = g, v = g^2 -> update = -lr * sign(g) (rapporto RMS = 1)
    assert torch.allclose(p.data, torch.tensor([0.9, 1.1, 0.9]), atol=1e-6)


def test_safetensors_reader_roundtrip(tmp_path):
    import struct
    from pianob.models.reve.weights import read_safetensors
    t = {"a.weight": torch.randn(2, 3), "b": torch.arange(4, dtype=torch.float32)}
    header, blobs, off = {}, b"", 0
    for k, v in t.items():
        b = v.numpy().tobytes()
        header[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        blobs += b
        off += len(b)
    h = json.dumps(header).encode()
    f = tmp_path / "m.safetensors"
    f.write_bytes(struct.pack("<Q", len(h)) + h + blobs)
    import pianob.models.reve.weights as W
    import builtins
    real_import = builtins.__import__

    def no_safetensors(name, *a, **k):
        if name.startswith("safetensors"):
            raise ImportError
        return real_import(name, *a, **k)
    builtins.__import__ = no_safetensors
    try:
        out = W.read_safetensors(f)
    finally:
        builtins.__import__ = real_import
    assert all(torch.equal(out[k], t[k]) for k in t)


def test_queues_session02():
    from pianob.experiments.queue import QUEUES
    F, G = QUEUES["F"](), QUEUES["G"]()
    hpo = [j.job_id for j in F if j.kind == "hpo"]
    sanity = [j for j in F if j.overrides.get("sanity_min_val")]
    sel = [j for j in F if j.kind == "select"]
    cands = [j for j in F if j.overrides.get("hpo_params")]
    assert len(hpo) == 2 and len(sanity) == 1 and len(sel) == 1 and sanity[0].depends_on == [sel[0].job_id]
    # riselezione: candidati dopo Optuna, selezione dopo tutti i candidati, id distinti dai job della Parte A
    assert len(cands) == 12 and all(set(hpo) <= set(j.depends_on) for j in cands)
    assert set(sel[0].depends_on) == {j.job_id for j in cands} and all("__cand" in j.job_id for j in cands)
    partA = [j for j in F if j.config.startswith("cbramod_twophase") and j.kind == "train" and j is not sanity[0]
             and not j.overrides.get("hpo_params")]
    assert all(sanity[0].job_id in j.depends_on for j in partA)
    assert [j.config for j in G[:2]] == ["reve_base_lp", "reve_base"]
    assert sum(1 for j in G if j.config == "reve_base" and j.protocol == "split_fm_lco" and j.perm is None
               and j.task == "emo9") == 9
