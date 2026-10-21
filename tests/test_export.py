"""Tests for weight export/import roundtrip."""

import sys, os, struct, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

import numpy as np
import torch
from model import Config, TinyGPT


def make_model():
    cfg = Config(d_model=32, n_layers=2, n_heads=2, d_ff=64, context_len=16)
    return cfg, TinyGPT(cfg)


def test_export_header():
    cfg, model = make_model()
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        path = f.name
    try:
        model.export_weights(path)
        with open(path, 'rb') as f:
            hdr = struct.unpack('<6i', f.read(24))
        assert hdr == (cfg.vocab_size, cfg.context_len, cfg.n_layers,
                       cfg.n_heads, cfg.d_model, cfg.d_ff)
    finally:
        os.unlink(path)


def test_export_file_size():
    cfg, model = make_model()
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        path = f.name
    try:
        model.export_weights(path)
        total_params = sum(p.numel() for p in model.parameters())
        expected = 24 + total_params * 4  # header + float32 params
        actual = os.path.getsize(path)
        assert actual == expected, f"file size {actual} != expected {expected}"
    finally:
        os.unlink(path)


def test_export_roundtrip_tok_emb():
    """Verify the first parameter (tok_emb) reads back correctly."""
    cfg, model = make_model()
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        path = f.name
    try:
        model.export_weights(path)
        with open(path, 'rb') as f:
            f.read(24)  # skip header
            tok_emb = np.frombuffer(f.read(cfg.vocab_size * cfg.d_model * 4), dtype=np.float32)
            tok_emb = tok_emb.reshape(cfg.vocab_size, cfg.d_model)
        expected = model.tok_emb.weight.detach().numpy()
        np.testing.assert_allclose(tok_emb, expected, atol=1e-6)
    finally:
        os.unlink(path)


def test_export_roundtrip_all_params():
    """Read back every parameter and compare to the PyTorch model."""
    cfg, model = make_model()
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        path = f.name
    try:
        model.export_weights(path)
        with open(path, 'rb') as f:
            f.read(24)
            rest = np.frombuffer(f.read(), dtype=np.float32)

        offset = 0

        def check(tensor, name):
            nonlocal offset
            expected = tensor.detach().numpy().ravel()
            actual = rest[offset:offset + len(expected)]
            np.testing.assert_allclose(actual, expected, atol=1e-6, err_msg=name)
            offset += len(expected)

        check(model.tok_emb.weight, 'tok_emb')
        check(model.pos_emb.weight, 'pos_emb')

        groups = [
            ('ln1.weight', 'ln1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
             'attn.proj.weight', 'attn.proj.bias', 'ln2.weight', 'ln2.bias',
             'ff.up.weight', 'ff.up.bias', 'ff.down.weight', 'ff.down.bias'),
        ]
        for group_names in groups:
            for gname in group_names:
                for i, block in enumerate(model.blocks):
                    parts = gname.split('.')
                    obj = block
                    for p in parts:
                        obj = getattr(obj, p)
                    check(obj, f'blocks.{i}.{gname}')

        check(model.ln_f.weight, 'ln_f.weight')
        check(model.ln_f.bias, 'ln_f.bias')
        check(model.head.weight, 'head.weight')
        check(model.head.bias, 'head.bias')

        assert offset == len(rest), f"unconsumed floats: {len(rest) - offset}"
    finally:
        os.unlink(path)
