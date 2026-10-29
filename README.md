# kindle-writes

A language model trained from scratch on a Kindle, using only the books in its own library. The Kindle does every step itself: it pulls the text out of the epubs, learns a tokenizer, trains a 1.07M-parameter transformer on its two ARM cores, and writes pages onto its e-ink screen.

<!-- PHOTO SLOT (hero): replace docs/klib-2026-10-28/photos/PHOTO-SLOT-hero.png with Abhay's photo of the Kindle, then fix this alt text. -->
![The Kindle showing its title card](docs/klib-2026-10-28/photos/PHOTO-SLOT-hero.png)

![A page written by the model on the Kindle, captured from its framebuffer](docs/klib-2026-10-28/framebuffer/04-write-morning.png)

The prompt is in bold. Everything after it was written by the model, on the Kindle, from weights it trained itself in 24 hours.

## What it learned

It writes real short sentences, puts dialogue in quotes, breaks paragraphs where a book would, and gives each book's characters back to the right kind of scene. It does not hold a thought across sentences yet: ask it who someone is and it answers "Yes."

![Same prompt and seed, before training and after](docs/klib-2026-10-28/framebuffer/02-before.png)

**Blind check.**
- **Pack:** 30 samples from this model mixed with 20 from an earlier model of the same size trained on Pride and Prejudice alone. A reader scored them without knowing which was which.

| | this model | earlier model |
|---|---:|---:|
| grammatical sentences per sample | 3.7 | 0.55 |
| samples with no grammatical sentence | 0 of 30 | 12 of 20 |
| samples with a readable paragraph | 3 of 30 (short dialogue, marginal) | 0 of 20 |

- **Copying:** the model is not reciting. The longest run of words any sample shares verbatim with the books is 7.

## The run (Oct 27 to 28, 2026)

| | |
|---|---|
| device | Kindle with a MediaTek MT8110: 2 Cortex-A53 cores (armv7), 1 GB RAM |
| data | 12 DRM-free epubs from the Kindle's library, 6.9 MB of text; the last ~5% of each book held out |
| tokenizer | byte-pair encoding, 1,024 tokens, trained on the Kindle in Lua |
| model | GPT, 4 layers, width 128, 4 heads, context 128, 1,072,896 parameters |
| training | 6,600 steps, 13.52M tokens, 24 h 01 min, 2 threads, about 155 tokens/s |
| held-out loss | 2.9613 to 1.4147 nats per character |
| interruptions | one: the charger came off, the battery guard paused training at 80%, and it resumed from its checkpoint 25 minutes later |

![Held-out loss during training, drawn on the Kindle](docs/klib-2026-10-28/framebuffer/07-curve.png)

**Kindle vs Mac replay.**
- A replay of the same run on a Mac reaches the same held-out loss.
- All 20 fixed samples (5 prompts x 4 seeds) are byte-identical to the Kindle's.
- The weights themselves differ only by float rounding, with a mean absolute difference of 4.6e-7.

## How it runs

Everything below runs on the Kindle (Lua 5.1, busybox, a static C binary, fbink).

| step | file |
|---|---|
| epubs to plain text, in reading order, each book split 95/5 into train and held-out | [`kindle/prep.lua`](kindle/prep.lua) |
| byte-pair tokenizer; output byte-identical to the Python reference | [`kindle/bpe.lua`](kindle/bpe.lua), [`trainer/prepare_tokens.py`](trainer/prepare_tokens.py) |
| training and sampling: forward, backward, AdamW, NEON kernels, battery and temperature guards, checkpoints | [`trainer/nightwriter.c`](trainer/nightwriter.c), [`trainer/gpt.c`](trainer/gpt.c) |
| launcher that refuses to start fresh over a run or resume without a checkpoint | [`kindle/nwrun.sh`](kindle/nwrun.sh) |
| a new page every morning at 7, saved into the Kindle library as a document | [`kindle/morning.sh`](kindle/morning.sh) |
| the cards in this README, drawn with fbink in the Kindle's Bookerly font | [`kindle/show.sh`](kindle/show.sh) |
| framebuffer capture to PNG | [`kindle/grab.sh`](kindle/grab.sh) |

Native writes on this Kindle go to `/mnt/base-us`, not the `/mnt/us` FUSE view. Build the trainer with `trainer/build.sh` (`native` for a Mac, `arm` for the Kindle via Docker). Set `KINDLE_HOST` in your environment or in a gitignored `.env`.

The books and the trained weights are not in this repo. The weights are 4.3 MB; their SHA256 is in [`docs/klib-2026-10-28/samples/WEIGHTS`](docs/klib-2026-10-28/samples/WEIGHTS).

![The Kindle's own terminal: hardware, the end of the training log, the weights hash](docs/klib-2026-10-28/framebuffer/08-proof.png)

## Evidence

In [`docs/klib-2026-10-28/`](docs/klib-2026-10-28/):
- the full training log;
- all 30 samples with SHA256SUMS, plus the sampler's own copy check;
- the blind pack, scores and key;
- framebuffer captures of every card.

## Earlier experiments

- **Shakespeare recall (Oct 2026).** The first version trained a byte-level GPT on a Mac from 30 repeated Shakespeare lines and ran inference on the Kindle in C (`train/`, `inference/`, `kindle/write.sh`). It reproduced its training file, by design.
- **Pride and Prejudice nights (Oct 23 to 27).** Then came on-device training on Pride and Prejudice alone, at 0.5M to 1.1M parameters over several nights. The best held-out loss was 1.14 nats per character, but the text stayed word salad: 0 readable paragraphs in 360 blind-scored samples.
- **What changed:** the data, not the model. One novel read 39 times teaches spelling. Twelve books read about five times teach sentences.

## License

MIT
