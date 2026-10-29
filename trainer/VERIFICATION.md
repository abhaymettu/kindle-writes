# Verifying nightwriter

Run every check on the Mac before the binary goes to the Kindle. Each command prints PASS or FAIL.

## 1. Build

```sh
trainer/build.sh            # trainer/nightwriter (Mac) and trainer/nightwriter-arm (Kindle)
python3 trainer/prepare_book.py data/pride.txt
```

The ARM build needs docker and an image with `arm-linux-gnueabihf-gcc` (`ARM_IMAGE`, default `localhost/armtools`).
To confirm that the matmul kernels compiled to fused NEON multiply-adds, run this command:

```sh
docker run --rm -v "$PWD":/src localhost/armtools arm-linux-gnueabihf-objdump -d /src/trainer/nightwriter-arm | grep -c vfma.f32
```

The result must be greater than 0.

## 2. Gradient check against train/model.py

```sh
.venv/bin/python trainer/gradcheck.py
```

The script builds `TinyGPT` (4 layers, d 128, 4 heads, d_ff 512, ctx 128, byte vocab 256, dropout 0) from seed 1337.
It perturbs every parameter by N(0, 0.02), so that biases and LayerNorm gains carry signal.
It then runs one batch of 4 x 128 windows from `data/shakespeare.txt` through the C trainer and through PyTorch in float64.

**Tolerance rule.** For each tensor, split per layer (54 tensors), the relative error is `||g_C - g_torch|| / ||g_torch||`.
The check passes only if every tensor is below 1e-4 and the loss is within 1e-4 relative.
The same rule applies to the update `p1 - p0` after one clipped AdamW step.
Torch takes that step from the C gradients, so the step check tests the optimizer alone.

Measured on 2026-10-23 with fixed seed 1337:

| build | worst gradient rel error | loss rel error | result |
|---|---|---|---|
| Mac arm64, 2 threads | 5.77e-07 | 3.5e-08 | PASS |
| Mac arm64, 1 thread | 5.77e-07 | 3.5e-08 | PASS |
| armv7 `nightwriter-arm` under qemu-arm, 2 threads | 5.84e-07 | 3.5e-08 | PASS |

The worst AdamW step error is 3.4e-05, on the LayerNorm gains.
That is fp32 rounding of a parameter near 1.0 (ulp 6e-8) measured against a 1e-3 step.

To run the same check against the ARM binary, first build the qemu image once.
Then point `--bin` at the wrapper:

```sh
printf 'FROM debian:bookworm-slim\nRUN apt-get update -qq && apt-get install -y -qq qemu-user python3 make gcc gawk\n' \
  | docker build -t localhost/kw-qemu -
TMPDIR=/tmp .venv/bin/python trainer/gradcheck.py --bin trainer/qemu-arm.sh
```

## 3. Behaviour smoke (minutes)

```sh
trainer/smoke.sh                       # Mac binary, full model shape
```

To run the ARM binary under emulation with a small shape, run this command:

```sh
docker run --rm -v "$PWD":/src -w /src \
  -e SMOKE_SHAPE="--layers 2 --dmodel 64 --heads 4 --dff 256 --ctx 64 --batch 2 --accum 2" localhost/kw-qemu \
  sh -c 'printf "#!/bin/sh\nexec qemu-arm /src/trainer/nightwriter-arm \"\$@\"\n" > /tmp/nw && chmod +x /tmp/nw && trainer/smoke.sh /tmp/nw'
```

The smoke script checks each of the following properties:

1. 1 and 2 threads give bit-identical losses over 20 steps, and an identical final page.
2. Stopping at step 7 and resuming replays the uninterrupted loss log exactly.
3. `kill -9` after step 10 and a resume replay the uninterrupted loss log exactly.
4. A corrupted newest checkpoint slot is rejected by its CRC, and the other slot resumes.
5. A fake SoC temperature of 70 C pauses before step 1. Cooling to 40 C resumes training, and the next page notes the pause.
6. Log rows carry the wifi state.
7. A fake battery at 10% and discharging writes a stop page and exits with status 3.
8. A restart after that resumes from the stop checkpoint and keeps the stop page.
9. `nightwriter sample` reproduces a page's text from that page's `.bin`.
10. `inference/tinygpt.c` and nightwriter decode the same 40 greedy bytes from the random-init page.
    This checks the weights layout. A copy with the ln1 weight and bias blocks swapped fails it.

Measured on 2026-10-23, all 12 checks pass for the Mac binary at full shape, and for `nightwriter-arm` under qemu at the small shape.

## 4. On the device (not done yet)

Run these checks after the binary is copied to `/mnt/us/kindle-writes/`:

```sh
./nightwriter-arm bench --threads 1 --bench-steps 3
./nightwriter-arm bench --threads 2 --bench-steps 3
```

`bench` reports measured tok/s and an estimated GFLOP/s derived from the model's FLOP count.
Use the tok/s value to size the run.

Next, run `smoke.sh` with the device binary: `sh smoke.sh ./nightwriter-arm pride.txt`.
The device has no `python3` or `cc`, so checks 9 and 10 fail there. Run those two on the Mac against a page pulled from the device.
The device busybox may lack `mktemp -d` or `cmp`; confirm both before trusting a device PASS.

Finally, check the sensor paths.
Run without `--sysroot` so the real sensors are read.
The `sensors:` line on stderr must name `zone0:mtktscpu` (or another zone selected by type), and the first `train.log` row must show a real `soc_c`, `batt_pct`, `batt_c` and `cpu_khz`, not `nan` or `-1`.
A missing SoC zone, or a battery without a readable charge or temperature that is not `Charging`, logs `sensor_missing` and pauses instead of training.
`batt_c` reads whole degrees C from `bd71827_bat/temp` on the device (29 to 31, measured 2026-10-23); raw values above 150 are taken as tenths.

## What these checks do not cover

- **Cross-architecture equality.** No check compares bits between architectures. Determinism holds per build: the same binary, seed and thread count give the same run.
- **Thermal thresholds.** `SOC_PAUSE_C 60` and `SOC_RESUME_C 50` are placeholders. The battery 42/38 C values come from the plan. Replace all of them from the M0 thermal trace.
- **Device speed.** Throughput on the Kindle is not measured with this trainer yet. qemu numbers say nothing about device speed.
