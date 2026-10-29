#!/usr/bin/env python3
"""Byte-level BPE for nightwriter's token mode. Stdlib only.

  python3 trainer/prepare_tokens.py --data clean.txt --n-train 678725 --vocab-size 768 --out DIR

Merges are learned on bytes [0, n_train) only. The held-out tail never shapes the tokenizer.
Train and held-out are tokenized separately, so the seam sits exactly at n_train.

Pre-split (mirrored by presplit() in nightwriter.c): a chunk is a run of letters, a run of
digits, or a run of other non-space bytes, each optionally led by one space; or a run of
whitespace, minus its final space when that space leads a following chunk. Merges never
cross chunks. Encoding applies merges in rank order (lowest rank first), GPT-2 style.

Writes DIR/tokens.bin, DIR/vocab.bin, DIR/meta.json.
  tokens.bin: "KWTOKS1\\0", u32 V, u32 0, u64 n_tokens, u64 n_train_tokens, u16 ids[n_tokens]
  vocab.bin:  "KWVOCAB1", u32 V, u32 n_merges, V x (u16 len, bytes), n_merges x (u16 a, u16 b)
              where id 256+i is the merge of pair i.
"""
import argparse
import hashlib
import json
import os
import struct
from collections import Counter, defaultdict

SPACE = frozenset(b" \t\n\r\f\v")


def cls(b):
    if b in SPACE:
        return 0
    if 65 <= b <= 90 or 97 <= b <= 122:
        return 1
    if 48 <= b <= 57:
        return 2
    return 3


def presplit(s):
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == 32 and i + 1 < n and cls(s[i + 1]):
            k, j = cls(s[i + 1]), i + 2
            while j < n and cls(s[j]) == k:
                j += 1
        elif cls(c):
            k, j = cls(c), i + 1
            while j < n and cls(s[j]) == k:
                j += 1
        else:
            j = i + 1
            while j < n and s[j] in SPACE:
                j += 1
            if j < n and s[j - 1] == 32 and j - 1 > i:
                j -= 1
        out.append(s[i:j])
        i = j
    return out


def learn(train, n_merges):
    words = Counter(presplit(train))
    seqs = [list(w) for w in words]
    counts = [words[w] for w in words]
    pc, where = defaultdict(int), defaultdict(set)
    for wi, ids in enumerate(seqs):
        for p in zip(ids, ids[1:]):
            pc[p] += counts[wi]
            where[p].add(wi)
    merges = []
    for m in range(n_merges):
        if not pc:
            break
        best = min(pc, key=lambda p: (-pc[p], p))
        new = 256 + m
        merges.append(best)
        for wi in list(where[best]):
            ids, c = seqs[wi], counts[wi]
            for p in zip(ids, ids[1:]):
                pc[p] -= c
                if pc[p] <= 0:
                    del pc[p]
                where[p].discard(wi)
            out, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best:
                    out.append(new)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            seqs[wi] = out
            for p in zip(out, out[1:]):
                pc[p] += c
                where[p].add(wi)
        where.pop(best, None)
    return merges


def encoder(merges):
    rank = {p: i for i, p in enumerate(merges)}
    cache = {}

    def enc_chunk(w):
        if w in cache:
            return cache[w]
        ids = list(w)
        while len(ids) > 1:
            r, at = min((rank.get(p, 1 << 30), i) for i, p in enumerate(zip(ids, ids[1:])))
            if r == 1 << 30:
                break
            pair, out, i = merges[r], [], 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                    out.append(256 + r)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        cache[w] = ids
        return ids

    def enc(s):
        return [t for w in presplit(s) for t in enc_chunk(w)]

    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument("--vocab-size", type=int, default=768)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = open(a.data, "rb").read()
    train, held = raw[: a.n_train], raw[a.n_train:]
    merges = learn(train, a.vocab_size - 256)
    V = 256 + len(merges)
    vocab = [bytes([i]) for i in range(256)]
    for x, y in merges:
        vocab.append(vocab[x] + vocab[y])
    enc = encoder(merges)
    tt, th = enc(train), enc(held)
    assert b"".join(vocab[t] for t in tt) == train and b"".join(vocab[t] for t in th) == held
    os.makedirs(a.out, exist_ok=True)
    tok_path, voc_path = os.path.join(a.out, "tokens.bin"), os.path.join(a.out, "vocab.bin")
    with open(tok_path, "wb") as f:
        f.write(b"KWTOKS1\0" + struct.pack("<IIQQ", V, 0, len(tt) + len(th), len(tt)))
        f.write(struct.pack("<%dH" % (len(tt) + len(th)), *(tt + th)))
    with open(voc_path, "wb") as f:
        f.write(b"KWVOCAB1" + struct.pack("<II", V, len(merges)))
        for v in vocab:
            f.write(struct.pack("<H", len(v)) + v)
        for x, y in merges:
            f.write(struct.pack("<HH", x, y))
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    meta = {
        "data": os.path.abspath(a.data), "data_sha256": hashlib.sha256(raw).hexdigest(),
        "n_bytes": len(raw), "n_train_bytes": len(train), "vocab_size": V, "merges": len(merges),
        "n_train_tokens": len(tt), "n_heldout_tokens": len(th),
        "chars_per_token_train": round(len(train) / len(tt), 4),
        "chars_per_token_heldout": round(len(held) / len(th), 4),
        "tokens_sha256": sha(tok_path), "vocab_sha256": sha(voc_path),
        "presplit": "letters | digits | other non-space, each with optional leading space; whitespace runs",
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
