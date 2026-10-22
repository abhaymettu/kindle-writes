# kindle-writes

A tiny language model built to generate text on a jailbroken Kindle's e-ink screen. It is trained on 30 famous Shakespeare lines, and it reproduces them. The on-device run is still waiting on hardware; see Current state.

A byte-level GPT (~890K params, 128-dim, 4 layers) trained from scratch on a small Shakespeare quote file, with a pure-C inference engine built for the Kindle's ARM processor. No dependencies beyond libc and libm. The training file is 30 famous quotes repeated, so the output below is recall, by design.

One real sample at temperature 0.8 (captured for this README; every line appears in the training file):

```
to be, that is the question.
Words without thoughts never to heaven go.
How sharper than a serpent tooth it is to have
```

## How it works

```
train/model.py      PyTorch model + weight export to flat binary
train/train.py      Trains on any text file (byte-level, no tokenizer)
inference/tinygpt.c Pure C inference, loads exported .bin, generates byte by byte
kindle/write.sh     Runs the model and renders output to e-ink via eips
kindle/deploy.sh    Copies binary + weights to Kindle over SSH
```

The model trains on a Mac, exports weights as a flat float32 binary, and the C engine loads them on the Kindle. The write script feeds generated text character by character to `eips`, the Kindle's built-in e-ink text renderer.

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

Requires a jailbroken Kindle with SSH access.

```sh
./kindle/deploy.sh [kindle-ip] [checkpoints/final.bin]
```

Set the Kindle address with `KINDLE_HOST` in your environment or in a gitignored `.env` file at the repo root (for example `KINDLE_HOST=<kindle-ip>`). The first argument overrides it.

Then SSH in and run `/mnt/us/kindle-writes/write.sh`.

## What's here

- 890K-param byte-level GPT (no tokenizer, vocab is raw bytes 0-255)
- Pure C inference with no dependencies
- Static ARM binary (480 KB) built for Kindle hardware
- Weight export verified by roundtrip tests (Python export matches C load)

## Test

```sh
make test
```

## Current state

The model trains, exports, and runs in C on this Mac (arm64). The on-device demo is waiting on hardware: the Kindle is offline, so the on-device run has not happened yet. Once it is powered on and reachable, deploy with `./kindle/deploy.sh` and run `write.sh` on-device.

## License

MIT
