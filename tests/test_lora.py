"""LoRA by hand against TP06 Ex 14-17 and Ex 23 (S06 p16-20)."""

import pytest
import torch
from torch import nn

from ledgar.models.lora import LoRALinear, inject_lora, lora_param_count, make_peft_model, merge_lora
from ledgar.train_bert import count_parameters, training_memory_gb


def test_lora_linear_course_values():
    torch.manual_seed(0)
    base = nn.Linear(768, 768)
    layer = LoRALinear(base, r=8, alpha=16)
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    assert trainable == 12_288 == lora_param_count(1, 768, 768, 8), "r (d_in + d_out) = 8 x 1536"
    x = torch.randn(4, 768)
    assert torch.allclose(layer(x), base(x)), "B = 0: exactly the pretrained layer at step 0"
    layer(x).sum().backward()
    assert layer.B.grad.abs().sum() > 0 and layer.A.grad.abs().sum() == 0, "B moves first, then A"
    assert base.weight.grad is None, "W is frozen"


def test_merge_lora():
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(32, 16), r=4, alpha=8)
    with torch.no_grad():
        layer.B.normal_()
    merged, x = merge_lora(layer), torch.randn(5, 32)
    assert torch.allclose(merged(x), layer(x), atol=1e-5)
    assert torch.linalg.matrix_rank(merged.weight - layer.base.weight) <= 4, "rank(W' - W) <= r"


def test_training_memory_course_values():
    assert training_memory_gb(0, 110_000_000) == pytest.approx(1.76)
    assert training_memory_gb(0, 7_000_000_000) == pytest.approx(112)
    assert training_memory_gb(7_000_000_000, 8_388_608, bytes_frozen=2) == pytest.approx(14.134, abs=1e-3)


@pytest.fixture(scope="module")
def distilbert():
    transformers = pytest.importorskip("transformers")
    transformers.logging.set_verbosity_error()
    try:
        return lambda: transformers.AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=4
        )
    except OSError:
        pytest.skip("distilbert-base-uncased not available")


@pytest.mark.slow
def test_inject_lora_distilbert(distilbert):
    torch.manual_seed(0)
    model = distilbert()
    total, head = sum(p.numel() for p in model.parameters()), 768 * 768 + 768 + 768 * 4 + 4
    assert total == 66_956_548 and head == 593_668, "DistilBERT body 66 362 880 + head (TP06 Ex 7)"
    ids = torch.tensor([[101, 1996, 8642, 4618, 3477, 1996, 9278, 1012, 102]])
    model.eval()
    with torch.no_grad():
        before = model(input_ids=ids).logits
    inject_lora(model, ("q_lin", "v_lin"), r=8, alpha=16)
    assert sum(isinstance(m, LoRALinear) for m in model.modules()) == 12, "6 blocks x (q, v)"
    assert count_parameters(model)[0] == 147_456 + 593_668 == 741_124
    with torch.no_grad():
        assert torch.allclose(model(input_ids=ids).logits, before, atol=1e-5), "same logits at step 0"


@pytest.mark.slow
def test_peft_matches_the_hand_written_version(distilbert):
    torch.manual_seed(0)
    model = make_peft_model(distilbert(), r=8, alpha=16, targets=("q_lin", "v_lin"))
    trainable, _ = count_parameters(model)
    assert trainable - 593_668 == 147_456, "peft trains the same adapters plus the whole head"
