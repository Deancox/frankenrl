"""MLP factory shared by every actor and critic.

Consolidates the per-folder `nn.Sequential(Linear, [LayerNorm,] Mish, ...)` blocks from the
legacy code. `layernorm=True` reproduces the `LayerNormP*` line; see the vault note
[[LayerNorm in RL critics and actors]].
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def _orthogonal_(module: nn.Module, gain: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


def resolve_activation(name: str) -> type[nn.Module]:
    """Look up an activation class by its config key (``"mish"``, ``"relu"``, ...)."""
    if name not in _ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; have {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]


def mlp(
    in_dim: int,
    out_dim: int,
    hidden: Sequence[int] = (256, 256),
    *,
    activation: str = "mish",
    layernorm: bool = False,
    output_activation: str | None = None,
    orthogonal_gain: float | None = np.sqrt(2),
) -> nn.Sequential:
    """Build ``Linear -> [LayerNorm] -> act`` stacks ending in a bare ``Linear``.

    Args:
        in_dim / out_dim: feature sizes.
        hidden: widths of the hidden layers.
        activation: key into the activation table (``mish`` matches the legacy code).
        layernorm: insert ``LayerNorm`` after every hidden ``Linear`` (the ``LayerNormP*`` line).
        output_activation: optional activation on the final layer (e.g. ``tanh``).
        orthogonal_gain: if not None, orthogonal-init every ``Linear`` with this gain and
            zero biases (legacy default). Pass None to keep PyTorch defaults.
    """
    act_cls = resolve_activation(activation)

    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers.append(nn.Linear(prev, width))
        if layernorm:
            layers.append(nn.LayerNorm(width))
        layers.append(act_cls())
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    if output_activation is not None:
        layers.append(resolve_activation(output_activation)())

    net = nn.Sequential(*layers)
    if orthogonal_gain is not None:
        net.apply(lambda m: _orthogonal_(m, orthogonal_gain))
    return net


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak: ``target <- tau * source + (1 - tau) * target`` (in-place, no grad)."""
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Copy ``source`` weights into ``target`` (used once at init)."""
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.copy_(sp)


def reinit_module_(module: nn.Module, *, orthogonal_gain: float | None = np.sqrt(2)) -> None:
    """Re-initialise every ``Linear``/``LayerNorm`` submodule of ``module`` in place.

    BRO's primacy-bias mitigation: wipe all network parameters on a fixed step schedule
    while keeping the replay buffer. See
    [[BRO - scaling off-policy actor-critic RL with regularized critics and optimistic
    exploration]] section 4. ``orthogonal_gain=None`` falls back to each layer's own
    ``reset_parameters`` (PyTorch's default init) instead of orthogonal init.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if orthogonal_gain is not None:
                nn.init.orthogonal_(m.weight, gain=orthogonal_gain)
                nn.init.zeros_(m.bias)
            else:
                m.reset_parameters()
        elif isinstance(m, nn.LayerNorm):
            m.reset_parameters()
