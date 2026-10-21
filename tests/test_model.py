"""Tests for the tiny GPT model."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

import torch
from model import Config, TinyGPT


def make_small_model():
    cfg = Config(d_model=32, n_layers=2, n_heads=2, d_ff=64, context_len=16)
    return cfg, TinyGPT(cfg)


def test_config_param_estimate():
    cfg = Config()
    est = cfg.param_count_estimate()
    assert est > 0
    model = TinyGPT(cfg)
    actual = sum(p.numel() for p in model.parameters())
    ratio = est / actual
    assert 0.8 < ratio < 1.2, f"estimate {est} vs actual {actual}"


def test_forward_shape():
    cfg, model = make_small_model()
    x = torch.randint(0, 256, (2, 8))
    logits, loss = model(x)
    assert logits.shape == (2, 8, 256)
    assert loss is None


def test_forward_with_targets():
    cfg, model = make_small_model()
    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    logits, loss = model(x, y)
    assert loss is not None
    assert loss.item() > 0


def test_generate_length():
    cfg, model = make_small_model()
    model.eval()
    prompt = torch.tensor([[10]], dtype=torch.long)
    out = model.generate(prompt, 20)
    assert out.shape == (1, 21)


def test_generate_stays_in_vocab():
    cfg, model = make_small_model()
    model.eval()
    prompt = torch.tensor([[65]], dtype=torch.long)
    out = model.generate(prompt, 50)
    assert out.min() >= 0
    assert out.max() < 256


def test_loss_decreases_on_overfit():
    """Verify the model can overfit a tiny batch (training signal works)."""
    cfg, model = make_small_model()
    data = torch.randint(0, 256, (4, cfg.context_len + 1))
    x, y = data[:, :-1], data[:, 1:]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first_loss = None
    for i in range(100):
        _, loss = model(x, y)
        if first_loss is None:
            first_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first_loss * 0.5, "loss did not decrease enough"
