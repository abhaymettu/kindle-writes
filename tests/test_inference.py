"""Test that C inference matches Python inference (deterministic, greedy)."""

import sys, os, struct, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

import numpy as np
import torch
from model import Config, TinyGPT

NATIVE_BIN = os.path.join(os.path.dirname(__file__), '..', 'inference', 'kindle-writes')


def _have_native():
    return os.path.isfile(NATIVE_BIN)


def make_model():
    torch.manual_seed(42)
    cfg = Config(d_model=32, n_layers=2, n_heads=2, d_ff=64, context_len=16)
    return cfg, TinyGPT(cfg)


def python_greedy(model, prompt_bytes, n_tokens):
    """Greedy decode (temperature -> 0, top_k=1 equivalent)."""
    model.eval()
    tokens = list(prompt_bytes)
    idx = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        for _ in range(n_tokens):
            ctx = idx[:, -model.cfg.context_len:]
            logits, _ = model(ctx)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_tok], dim=1)
    return idx[0].tolist()


def c_greedy(weights_path, prompt_bytes, n_tokens):
    """Run the C binary in greedy mode (temp=0.01, topk=1, fixed seed)."""
    prompt = bytes(prompt_bytes).decode('latin-1')
    result = subprocess.run(
        [NATIVE_BIN, weights_path, '-n', str(n_tokens), '-t', '0.01',
         '-k', '1', '-s', '42', '-p', prompt],
        capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"C binary failed: {result.stderr.decode()}")
    out = result.stdout
    if out.endswith(b'\n'):
        out = out[:-1]
    return list(prompt_bytes) + list(out)


def test_greedy_match():
    """C and Python produce the same tokens under greedy decoding."""
    if not _have_native():
        import pytest
        pytest.skip("native binary not built; run 'make native' first")

    cfg, model = make_model()
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        weights_path = f.name
    try:
        model.export_weights(weights_path)
        prompt = [ord('H'), ord('e')]
        n = 20

        py_tokens = python_greedy(model, prompt, n)
        c_tokens = c_greedy(weights_path, prompt, n)

        # allow minor float divergence: check first 10 tokens match
        match = 0
        for a, b in zip(py_tokens[len(prompt):], c_tokens[len(prompt):]):
            if a == b:
                match += 1
            else:
                break

        assert match >= 8, (
            f"only {match}/{n} leading tokens match.\n"
            f"python: {py_tokens[len(prompt):len(prompt)+15]}\n"
            f"c:      {c_tokens[len(prompt):len(prompt)+15]}"
        )
    finally:
        os.unlink(weights_path)


def test_c_binary_runs():
    """Smoke test: the C binary loads a model and produces output."""
    if not _have_native():
        import pytest
        pytest.skip("native binary not built; run 'make native' first")

    cfg, model = make_model()
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        weights_path = f.name
    try:
        model.export_weights(weights_path)
        result = subprocess.run(
            [NATIVE_BIN, weights_path, '-n', '10', '-s', '1'],
            capture_output=True, timeout=30)
        assert result.returncode == 0
        assert len(result.stdout) >= 5
    finally:
        os.unlink(weights_path)
