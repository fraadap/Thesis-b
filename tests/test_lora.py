import torch

from pianob.models.cbramod import CBraModClassifier
from pianob.models.lora import apply_lora, count_trainable, lora_parameters, mark_trainable_lora


def _model():
    torch.manual_seed(0)
    return CBraModClassifier(n_channels=4, n_samples=400, n_classes=3, head="pooled_mlp", n_layer=2).eval()


def test_lora_zero_init_identical_output_and_trainable_set():
    m = _model()
    x = torch.randn(2, 4, 400)
    with torch.no_grad():
        y0 = m(x)
    names = apply_lora(m, rank=2)
    mark_trainable_lora(m)
    assert len(names) == 6 * 2   # per layer: in_proj e out_proj (attenzione spaziale e temporale), linear1, linear2
    with torch.no_grad():
        y1 = m(x)
    assert torch.allclose(y0, y1, atol=1e-6)
    for n, p in m.named_parameters():
        is_lora = ".parametrizations." in n and (n.endswith(".A") or n.endswith(".B"))
        assert p.requires_grad == (is_lora or n.startswith("classifier.")), n
    c = count_trainable(m)
    assert c["lora"] > 0 and c["trainable"] < c["total"]


def test_lora_changes_output_when_B_nonzero_and_gets_grad():
    m = _model()
    apply_lora(m, rank=2)
    mark_trainable_lora(m)
    x = torch.randn(2, 4, 400)
    with torch.no_grad():
        y0 = m(x)
        for p in lora_parameters(m):
            if p.dim() == 3 and p.shape[-1] == 2:  # B: (blocks, out, r)
                p.add_(0.05)
    y1 = m(x)
    assert not torch.allclose(y0, y1)
    y1.sum().backward()
    assert all(p.grad is not None for p in lora_parameters(m))
    base = dict(m.named_parameters())
    assert all(base[n].grad is None for n in base if n.endswith(".original"))
