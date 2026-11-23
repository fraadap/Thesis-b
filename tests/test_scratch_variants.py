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
