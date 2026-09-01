import torch

from frankenrl.nn.blocks import hard_update, mlp, soft_update


def test_mlp_shape_and_layernorm():
    net = mlp(5, 3, (16, 16), layernorm=True)
    out = net(torch.zeros(4, 5))
    assert out.shape == (4, 3)
    assert any(isinstance(m, torch.nn.LayerNorm) for m in net)


def test_mlp_no_layernorm_by_default():
    net = mlp(5, 3, (16,))
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in net)


def test_output_activation():
    net = mlp(4, 2, (8,), output_activation="tanh")
    out = net(torch.randn(10, 4) * 100)
    assert out.abs().max() <= 1.0


def test_soft_update_is_convex_combination():
    a = mlp(3, 3, (8,))
    b = mlp(3, 3, (8,))
    before = [p.clone() for p in a.parameters()]
    soft_update(a, b, tau=0.25)
    for p0, pa, pb in zip(before, a.parameters(), b.parameters()):
        assert torch.allclose(pa, 0.75 * p0 + 0.25 * pb, atol=1e-6)


def test_hard_update_copies():
    a = mlp(3, 3, (8,))
    b = mlp(3, 3, (8,))
    hard_update(a, b)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb)
