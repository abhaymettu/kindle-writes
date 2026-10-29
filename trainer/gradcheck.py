"""Gradient check: C trainer vs train/model.py on one fixed batch.

Builds TinyGPT from a fixed seed, perturbs every parameter so biases and LayerNorm
paths carry signal, exports it in the tinygpt.c format, runs `nightwriter gradcheck`,
and compares against PyTorch in float64:

  - loss
  - every gradient tensor, split per layer: ||g_c - g_ref|| / ||g_ref|| < --tol (default 1e-4)
  - parameters after one clipped AdamW step, same rule on the update p1 - p0. Torch steps
    from the C gradients so this isolates the optimizer: from its own gradients, Adam's
    first step turns fp32 noise on near-zero entries (the key bias has an exactly zero
    gradient) into a full +-lr step and the comparison would measure that noise.

Exit status 0 only if every tensor passes.
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch

_md = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--model-dir=')),
           os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, _md)
from model import Config, TinyGPT  # noqa: E402

GROUPS = ('ln1.weight', 'ln1.bias', 'attn.qkv.weight', 'attn.qkv.bias', 'attn.proj.weight',
          'attn.proj.bias', 'ln2.weight', 'ln2.bias', 'ff.up.weight', 'ff.up.bias',
          'ff.down.weight', 'ff.down.bias')


def export_order(model):
    """(name, parameter) in the exact order of the flat weights file."""
    out = [('tok_emb', model.tok_emb.weight), ('pos_emb', model.pos_emb.weight)]
    for group in GROUPS:
        for i, block in enumerate(model.blocks):
            obj = block
            for part in group.split('.'):
                obj = getattr(obj, part)
            out.append((f'blocks.{i}.{group}', obj))
    out += [('ln_f.weight', model.ln_f.weight), ('ln_f.bias', model.ln_f.bias),
            ('head.weight', model.head.weight), ('head.bias', model.head.bias)]
    return out


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--bin', default=os.path.join(here, 'nightwriter'))
    p.add_argument('--data', default=os.path.join(here, '..', 'data', 'shakespeare.txt'))
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--ctx', type=int, default=128)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--tol', type=float, default=1e-4)
    p.add_argument('--tokens', help='BPE tokens.bin from prepare_tokens.py; windows and V come from it')
    p.add_argument('--model-dir', help='directory holding train/model.py (read before argparse)')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.tokens:
        hdr = open(args.tokens, 'rb').read(32)
        V, _, n_tok, _ = struct.unpack('<IIQQ', hdr[8:32])
        raw = np.fromfile(args.tokens, dtype='<u2', offset=32, count=n_tok)
    else:
        V, raw = 256, np.frombuffer(open(args.data, 'rb').read(), dtype=np.uint8)
    cfg = Config(vocab_size=V, context_len=args.ctx, n_layers=4, n_heads=4, d_model=128, d_ff=512, dropout=0.0)
    model = TinyGPT(cfg)
    with torch.no_grad():
        for prm in model.parameters():
            prm.add_(0.02 * torch.randn_like(prm))

    rng = np.random.RandomState(args.seed)
    starts = rng.randint(0, len(raw) - args.ctx - 1, size=args.batch)
    x = np.stack([raw[s:s + args.ctx] for s in starts]).astype(np.int32)
    y = np.stack([raw[s + 1:s + args.ctx + 1] for s in starts]).astype(np.int32)

    with tempfile.TemporaryDirectory() as tmp:
        wpath, bpath, gpath = (os.path.join(tmp, n) for n in ('w.bin', 'b.bin', 'g.bin'))
        model.export_weights(wpath)
        with open(bpath, 'wb') as f:
            f.write(struct.pack('<2i', args.batch, args.ctx))
            f.write(x.tobytes())
            f.write(y.tobytes())
        subprocess.run([args.bin, 'gradcheck', '--init', wpath, '--batch', bpath, '--out', gpath,
                        '--lr', repr(args.lr), '--threads', str(args.threads)], check=True, stdout=subprocess.DEVNULL)
        out = np.fromfile(gpath, dtype=np.float32)

    ref = model.double()
    ref.train()
    _, loss = ref(torch.from_numpy(x).long(), torch.from_numpy(y).long())
    loss.backward()

    named = export_order(ref)
    n = sum(prm.numel() for _, prm in named)
    assert out.size == 1 + 2 * n, f'C output has {out.size} floats, expected {1 + 2 * n}'
    c_loss, c_grads, c_after = float(out[0]), out[1:1 + n], out[1 + n:]

    worst, failures = 0.0, []
    loss_err = abs(c_loss - loss.item()) / abs(loss.item())
    print(f'loss  C {c_loss:.7f}  torch(fp64) {loss.item():.7f}  rel {loss_err:.2e}')
    if loss_err >= args.tol:
        failures.append('loss')

    before = [prm.detach().clone() for _, prm in named]
    decay = [prm for name, prm in named if prm.dim() >= 2]
    no_decay = [prm for name, prm in named if prm.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1}, {'params': no_decay, 'weight_decay': 0.0}],
                            lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
    grads_ref = [prm.grad.detach().numpy().ravel().copy() for _, prm in named]
    off = 0
    for _, prm in named:
        prm.grad = torch.from_numpy(c_grads[off:off + prm.numel()].astype(np.float64)).view_as(prm).clone()
        off += prm.numel()
    torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
    opt.step()

    off, rows = 0, []
    for (name, prm), g_ref, p0 in zip(named, grads_ref, before):
        k = prm.numel()
        e_grad = rel(c_grads[off:off + k].astype(np.float64), g_ref)
        d_ref = (prm.detach() - p0).numpy().ravel()
        d_c = c_after[off:off + k].astype(np.float64) - p0.numpy().ravel()
        e_step = rel(d_c, d_ref)
        rows.append((name, k, e_grad, e_step))
        for kind, e in (('grad', e_grad), ('adamw', e_step)):
            if e >= args.tol:
                failures.append(f'{kind} {name}')
        worst = max(worst, e_grad)
        off += k

    print(f'{"tensor":32s} {"numel":>8s} {"grad rel":>10s} {"adamw rel":>10s}')
    for name, k, eg, es in rows:
        print(f'{name:32s} {k:8d} {eg:10.2e} {es:10.2e}')
    print(f'worst gradient rel error {worst:.2e} over {len(rows)} tensors, tol {args.tol:g}')
    if failures:
        print('FAIL: ' + ', '.join(failures))
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
