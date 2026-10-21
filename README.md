# kindle-writes

A tiny language model that generates text directly on a jailbroken Kindle's e-ink screen.

Byte-level GPT (~890K params, 128-dim, 4 layers) trained from scratch on Shakespeare, with a pure-C inference engine that runs on the Kindle's ARM processor. No dependencies beyond libc and libm.

## How it works

```
train/model.py      PyTorch model + weight export to flat binary
train/train.py      Trains on any text file (byte-level, no tokenizer)
inference/tinygpt.c Pure C inference, loads exported .bin, generates byte by byte
kindle/write.sh     Runs the model and renders output to e-ink via eips
kindle/deploy.sh    Copies binary + weights to Kindle over SSH
```

The model trains on a Mac, exports weights as a flat float32 binary, and the C engine loads them on the Kindle. The write script feeds generated text character-by-character to `eips`, the Kindle's built-in e-ink text renderer.

## Build

```sh
# native (for testing on Mac)
make native

# ARM for Kindle (needs arm-linux-gnueabihf-gcc or Docker)
make arm
# or with Docker:
docker run --rm -v "$(pwd)/inference:/src" debian:bookworm sh -c \
  "apt-get update -qq && apt-get install -y -qq gcc-arm-linux-gnueabihf >/dev/null 2>&1 && \
   arm-linux-gnueabihf-gcc -O2 -Wall -Wextra -std=c99 -static -mfloat-abi=hard -mfpu=neon \
   -o /src/kindle-writes-arm /src/tinygpt.c -lm"
```

## Train

```sh
python -m venv .venv && source .venv/bin/activate
pip install torch numpy
python train/train.py --data data/shakespeare.txt --steps 3000 --device auto
```

Produces `checkpoints/final.bin` (~3.4 MB) and `checkpoints/final.pt`.

## Run locally

```sh
./inference/kindle-writes checkpoints/final.bin -n 200 -t 0.8 -p "To be or not"
```

## Deploy to Kindle

Requires a jailbroken Kindle with KOReader and SSH access (see [kindledock](https://github.com/abhay/kindledock) for setup).

```sh
./kindle/deploy.sh [kindle-ip] [checkpoints/final.bin]
```

Then SSH in and run:
```sh
/mnt/us/kindle-writes/write.sh
```

## Test

```sh
make test
```

## What's here

- 890K-param byte-level GPT (no tokenizer, vocab is raw bytes 0-255)
- Pure C inference with no dependencies
- Static ARM binary (480 KB) runs on Kindle hardware
- Weight export verified by roundtrip tests (Python export matches C load)
- C and Python outputs match under greedy decoding

## Current state

The model trains, exports, and runs in C on x86 and ARM. Sample output (greedy, Shakespeare-trained):

```
To be or not to be, that is the question.
Hell is empty and all the devils are here.
No legacy is so rich as hon
```

**Hardware blocker:** The Kindle is registered in Tailscale (`kindle`, 100.83.27.105) but offline (last seen 2026-10-20, ~20h). SSH on port 2222 times out. The device needs to be powered on and connected to a network with Tailscale running. Once reachable, deploy with `./kindle/deploy.sh` and run `write.sh` on-device. Everything else is ready.

## License

MIT
