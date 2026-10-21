"""Train the tiny GPT on a text file. Byte-level, no tokenizer needed."""

import argparse
import os
import time
import torch
from model import Config, TinyGPT


def load_data(path):
    with open(path, 'rb') as f:
        raw = f.read()
    return torch.tensor(list(raw), dtype=torch.long)


def get_batch(data, cfg, batch_size, device):
    ix = torch.randint(len(data) - cfg.context_len - 1, (batch_size,))
    x = torch.stack([data[i:i + cfg.context_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + cfg.context_len + 1] for i in ix]).to(device)
    return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='training text file')
    p.add_argument('--out', default='checkpoints', help='output directory')
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--d-ff', type=int, default=512)
    p.add_argument('--context', type=int, default=256)
    p.add_argument('--device', default='cpu')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--save-every', type=int, default=1000)
    args = p.parse_args()

    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    cfg = Config(
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, context_len=args.context,
    )
    print(f"config: {cfg.__dict__}")
    print(f"estimated params: {cfg.param_count_estimate():,}")

    data = load_data(args.data)
    print(f"data: {len(data):,} bytes")
    if len(data) < cfg.context_len * 2:
        raise ValueError("data file too small for this context length")

    model = TinyGPT(cfg).to(args.device)
    actual = sum(p.numel() for p in model.parameters())
    print(f"actual params: {actual:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr / 10)

    os.makedirs(args.out, exist_ok=True)
    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(data, cfg, args.batch, args.device)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step:5d} | loss {loss.item():.4f} | {elapsed:.1f}s")

        if step % args.save_every == 0:
            ckpt = os.path.join(args.out, f'step_{step:06d}.pt')
            torch.save({'cfg': cfg.__dict__, 'model': model.state_dict(), 'step': step}, ckpt)
            wpath = os.path.join(args.out, f'step_{step:06d}.bin')
            model.export_weights(wpath)
            print(f"  saved {ckpt} and {wpath}")

    final_pt = os.path.join(args.out, 'final.pt')
    final_bin = os.path.join(args.out, 'final.bin')
    torch.save({'cfg': cfg.__dict__, 'model': model.state_dict(), 'step': args.steps}, final_pt)
    model.export_weights(final_bin)
    print(f"done. saved {final_pt} ({os.path.getsize(final_bin):,} bytes)")

    model.eval()
    prompt = torch.tensor([[ord('\n')]], dtype=torch.long, device=args.device)
    out = model.generate(prompt, 200, temperature=0.8)
    text = bytes(out[0].tolist()).decode('utf-8', errors='replace')
    print(f"\nsample:\n{text}")


if __name__ == '__main__':
    main()
