import torch
import torch.nn as nn

from pianob.config import TRAIN_DEFAULTS
from pianob.experiments.registry import get_config
from pianob.models.heads import build_model
from pianob.train.loop import build_optimizer


def test_default_init_variant_changes_only_backbone_linears():
    torch.manual_seed(0)
    a = build_model(get_config("cbramod_scratch_ft").with_hparams(n_layer=1, head="avgpooling_patch_reps"), 4, 400, 9)
    torch.manual_seed(0)
    b = build_model(get_config("cbramod_scratch_ft_definit").with_hparams(n_layer=1, head="avgpooling_patch_reps"), 4, 400, 9)
    la = [m.weight.std().item() for m in a.backbone.modules() if isinstance(m, nn.Linear)]
    lb = [m.weight.std().item() for m in b.backbone.modules() if isinstance(m, nn.Linear)]
    assert all(x > y for x, y in zip(la, lb))   # kaiming(fan_out) ha scala più grande del default


def test_lr_warmup_schedule():
    cfg = get_config("cbramod_scratch_ft_warmup")
    hp = {**TRAIN_DEFAULTS, **cfg.hparams, "epochs": 10}
    m = build_model(cfg.with_hparams(n_layer=1, head="avgpooling_patch_reps"), 4, 400, 9)
    opt, sched = build_optimizer(m, hp, cfg.recipe, steps_per_epoch=10)
    lr0 = [g["lr"] for g in opt.param_groups]
    for _ in range(30):
        opt.step()
        sched.step()
    lr30 = [g["lr"] for g in opt.param_groups]
    assert lr0[0] < 0.02 * hp["lr"] and abs(lr30[0] - hp["lr"]) / hp["lr"] < 0.05


def test_eegnet_lawhern_max_norm_and_constant_lr():
    import numpy as np
    from pianob.train.loop import TrialBank, train_model
    cfg = get_config("eegnet_lawhern")
    m = build_model(cfg, 4, 400, 3, fs=100)
    with torch.no_grad():
        m.fc.weight.mul_(100)
        m.block1[2].weight.mul_(100)
    m.apply_constraints()
    assert m.fc.weight.norm(dim=1).max() <= 0.25 + 1e-5
    assert m.block1[2].weight.flatten(1).norm(dim=1).max() <= 1.0 + 1e-5
    rng = np.random.default_rng(0)
    data = rng.standard_normal((12, 4, 400)).astype(np.float32)
    tr = TrialBank(data, np.repeat([0, 1, 2], 4), np.tile(np.arange(4), 3), np.arange(12) % 3, 400)
    hp = {**TRAIN_DEFAULTS, **cfg.hparams, "epochs": 2, "amp": False}
    r = train_model(m, hp, cfg.recipe, tr, tr, 3, "cpu", seed=0, log=lambda x: None)
    assert r["status"] in ("completed", "early_stopped")
    assert m.fc.weight.norm(dim=1).max() <= 0.25 + 1e-5      # vincolo mantenuto durante il training
