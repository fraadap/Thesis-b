"""Controlli obbligatori prima di lanciare il notebook G (Appendice R §R10). Si fermano al primo errore."""
from __future__ import annotations

import numpy as np
import torch


def preflight(adapter, device="cpu", log=print, require_weights: bool = True) -> dict:
    from ...data.faced_meta import CHANNELS
    from ...data.splits import make_split
    from ...experiments.registry import get_config
    from ...train.loop import TrialBank
    from . import build_reve
    out = {}
    sp = make_split("split_fm", 0, adapter.task_clips("emo9"), adapter.subjects(), clip_labels=adapter.meta["clip_labels"])
    counts = tuple(len(s) * len(sp.train_clips) * 3 for s in (sp.train_subjects, sp.val_subjects, sp.test_subjects))
    assert counts == (6720, 1680, 1932), f"conteggi delle finestre {counts} != (6720, 1680, 1932)"
    assert not (set(sp.train_subjects) & set(sp.test_subjects)) and not (set(sp.val_subjects) & set(sp.test_subjects))
    out["counts"] = counts
    cfg = get_config("reve_base")
    if not require_weights:
        cfg.pretrained = False
    model = build_reve(cfg, len(CHANNELS), 2000, 9, CHANNELS, synthetic=adapter.synthetic, log=log)
    if require_weights:
        assert model.pretrained_info and model.pretrained_info.get("strict"), "pesi REVE non caricati con strict=True"
        out["weights"] = model.pretrained_info
    subs = sp.train_subjects[:3]
    x = adapter.trials(subs, list(range(28)), channels=list(range(32)))     # dati/100 (convenzione del loader)
    x = x.reshape(-1, *x.shape[-2:])
    lab = np.tile(np.asarray(adapter.labels("emo9")), len(subs))
    bank = TrialBank(x, np.repeat(subs, 28), np.tile(np.arange(28), len(subs)), lab, 2000, 0.0, "cpu")
    xb, yb = bank.get(np.arange(64), device)
    assert tuple(xb.shape) == (64, 32, 2000) and tuple(yb.shape) == (64,), (xb.shape, yb.shape)
    assert int(yb.min()) >= 0 and int(yb.max()) <= 8
    out["input_scale_check"] = float((xb * model.input_scale).abs().median())   # ordine di grandezza dopo /1000
    pos = model.positions.unsqueeze(0).expand(64, -1, -1)
    assert tuple(pos.shape) == (64, 32, 3)
    model.to(device).eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=torch.device(device).type == "cuda"):
        tok = model.tokens(xb[:8])
        logits = model(xb[:8])
    assert tuple(tok.shape) == (8, 352, 512), tok.shape
    assert model.out_shape == 180736, model.out_shape
    assert torch.isfinite(logits.float()).all(), "NaN/Inf nell'uscita"
    model.set_phase("lp")
    n_lp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_lp == 1807881, f"parametri allenabili in LP {n_lp} != 1807881"
    model.set_phase("ft")
    out.update(encoder_out=tuple(tok.shape), head_in=model.out_shape, lp_trainable=n_lp,
               ft_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    log(f"[REVE] controlli §R10 superati: {out}")
    return out
