"""Tiny GPT for Kindle inference. Byte-level, ~2M params."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size: int = 256
    context_len: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.1

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def param_count_estimate(self):
        emb = self.vocab_size * self.d_model + self.context_len * self.d_model
        attn = self.n_layers * (4 * self.d_model * self.d_model + 4 * self.d_model)
        ffn = self.n_layers * (2 * self.d_model * self.d_ff + self.d_model + self.d_ff)
        head = self.d_model * self.vocab_size + self.vocab_size
        return emb + attn + ffn + head


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_ff)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.drop = nn.Dropout(cfg.dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, top_k=40):
        for _ in range(max_new):
            ctx = idx[:, -self.cfg.context_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def export_weights(self, path):
        """Export weights as a flat binary for the C inference engine.

        Layout must match the C loader in inference/tinygpt.c:
        header (6 ints), tok_emb, pos_emb, then per-layer groups
        (all ln1_w, all ln1_b, all qkv_w, ...), then final ln + head.
        """
        import struct
        import numpy as np

        def dump(f, t):
            f.write(t.detach().cpu().float().numpy().tobytes())

        with open(path, 'wb') as f:
            cfg = self.cfg
            f.write(struct.pack('<6i',
                cfg.vocab_size, cfg.context_len, cfg.n_layers,
                cfg.n_heads, cfg.d_model, cfg.d_ff))
            dump(f, self.tok_emb.weight)
            dump(f, self.pos_emb.weight)
            for group in ('ln1.weight', 'ln1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
                          'attn.proj.weight', 'attn.proj.bias', 'ln2.weight', 'ln2.bias',
                          'ff.up.weight', 'ff.up.bias', 'ff.down.weight', 'ff.down.bias'):
                for block in self.blocks:
                    parts = group.split('.')
                    obj = block
                    for p in parts:
                        obj = getattr(obj, p)
                    dump(f, obj)
            dump(f, self.ln_f.weight)
            dump(f, self.ln_f.bias)
            dump(f, self.head.weight)
            dump(f, self.head.bias)
