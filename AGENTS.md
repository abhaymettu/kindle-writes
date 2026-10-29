# AGENTS.md

Instructions for a coding agent setting this repo up for someone. Read `README.md` first for what the project is.

## 1. Check the published samples (any machine with a C compiler)

```sh
sh kindle/reproduce.sh check
```

Expect `30 of 30 samples match the Kindle's, byte for byte`. Anything less means the build or the weights differ; stop and report it.

## 2. Write one page locally

```sh
cc -O3 -std=c11 -o nightwriter trainer/gpt.c trainer/nightwriter.c -lm -lpthread
W=docs/klib-2026-10-28/weights
./nightwriter sample --init $W/51.bin --tokens $W/train-mask.bin --vocab $W/vocab.bin \
  --prompt "The phone rang twice." --sample-seed 94411 --temp 0.65 --top-k 8 --page-chars 600 --lexicon 0
```

The prompt must be under 64 characters. The same seed always gives the same page.

## 3. Train on the user's Kindle (only if they ask)

Confirm with the user before starting. The run occupies the Kindle's CPU for about a day.

Preconditions to check, in order:
1. `ssh -i ~/.ssh/kindle_koreader -p 2222 root@$KINDLE_HOST true` succeeds. If not, the Kindle needs a jailbreak and an SSH server (KOReader's works); the user has to do that on the device.
2. `ssh ... 'ls /mnt/us/documents/*.epub'` lists DRM-free epubs. Only use books the user owns.
3. `ssh ... 'lua -v'` prints Lua 5.1.
4. The charger is plugged in. The trainer pauses itself on low battery or heat.

Then:

```sh
KINDLE_HOST=<ip> sh kindle/reproduce.sh kindle
```

Progress: `ssh ... 'tail -f /mnt/base-us/kindle-writes-repro/run/train.log'`. Write to `/mnt/base-us/...`, not `/mnt/us/...` (FUSE view).

## Rules

- Never commit book text, extracted text (`text.txt`), token files (`tokens.bin`) or epubs. The `.gitignore` blocks `*.bin` except the published weights.
- Put `KINDLE_HOST` in `.env` (gitignored), never in a tracked file.
- Don't kill or stop KOReader (`reader.lua`) on the user's Kindle except for a few seconds around a screen capture, and always resume it with `kill -CONT`.
