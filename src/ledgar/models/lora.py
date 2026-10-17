"""LoRA by hand (S06 p18-20, TP06 Part C), and the same adapters with `peft` for the training runs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

HEAD = ("pre_classifier", "classifier")  # DistilBERT's new head: trained fully, it did not exist before (S06 p20)


class LoRALinear(nn.Module):
    """h = W x + (alpha / r) B A x, with W frozen, A random and B = 0 (TP06 Ex 14).

    At step 0, B A = 0: the layer is exactly the pretrained one (S06 p19). The big B @ A is never built
    during training: x goes through A (r outputs) then B.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (x @ self.A.T @ self.B.T)


def merge_lora(layer: LoRALinear) -> nn.Linear:
    """W' = W + (alpha / r) B A in a plain nn.Linear: same outputs, no cost at inference (TP06 Ex 15)."""
    merged = nn.Linear(layer.base.in_features, layer.base.out_features, bias=layer.base.bias is not None)
    with torch.no_grad():
        merged.weight.copy_(layer.base.weight + layer.scaling * layer.B @ layer.A)
        if layer.base.bias is not None:
            merged.bias.copy_(layer.base.bias)
    return merged


def inject_lora(
    model: nn.Module, targets: Sequence[str] = ("q_lin", "v_lin"), r: int = 8, alpha: int = 16
) -> nn.Module:
    """Freeze everything, wrap every nn.Linear whose name is in `targets` in a LoRALinear, unfreeze the head.

    TP06 Ex 16. In place; returns the model.
    """
    for p in model.parameters():
        p.requires_grad = False
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha))
    for name, p in model.named_parameters():
        if name.startswith(HEAD):
            p.requires_grad = True
    return model


def lora_param_count(n_layers: int, d_in: int, d_out: int, r: int) -> int:
    """r (d_in + d_out) per adapted projection (S06 p18)."""
    return n_layers * r * (d_in + d_out)


def make_peft_model(model: nn.Module, r: int, alpha: int, targets: Sequence[str], dropout: float = 0.0) -> nn.Module:
    """The same adapters with `peft` (TP06 Ex 23): task SEQ_CLS keeps the head fully trainable."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        task_type="SEQ_CLS", r=r, lora_alpha=alpha, target_modules=list(targets), lora_dropout=dropout
    )
    return get_peft_model(model, config)
