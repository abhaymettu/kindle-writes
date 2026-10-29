#!/bin/sh
# Quick behavioural checks of nightwriter on a short corpus run. Minutes, not hours.
#   trainer/smoke.sh [path/to/nightwriter] [corpus]
# SMOKE_SHAPE adds flags to every run, e.g. a smaller model under qemu.
set -u

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/.." && pwd)"
NW="${1:-$DIR/nightwriter}"
DATA="${2:-$REPO/data/pride.txt}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/nightwriter-smoke.XXXXXX")"
COMMON="--data $DATA --steps 20 --eval-every 5 --ckpt-every 3 --time-scale 100000 --page-chars 400 ${SMOKE_SHAPE:-}"
FAILS=0

# A missing SoC sensor pauses training, so a host without thermal zones (the Mac) runs the
# general tests against a cool fake sysroot. On the Kindle they read the real sensors.
if [ ! -d /sys/class/thermal/thermal_zone0 ]; then
    COOL="$WORK/cool"
    mkdir -p "$COOL/sys/class/thermal/thermal_zone0" "$COOL/sys/class/power_supply/bat"
    echo mtktscpu > "$COOL/sys/class/thermal/thermal_zone0/type"
    echo 30000 > "$COOL/sys/class/thermal/thermal_zone0/temp"
    echo 80 > "$COOL/sys/class/power_supply/bat/capacity"
    echo 25 > "$COOL/sys/class/power_supply/bat/temp"
    echo Charging > "$COOL/sys/class/power_supply/bat/status"
    COMMON="$COMMON --sysroot $COOL"
fi

check() {
    if [ "$1" = 0 ]; then echo "PASS  $2"; else echo "FAIL  $2"; FAILS=$((FAILS + 1)); fi
}

# Final row per step of (step, train_loss, val_loss): a resumed run replays steps after its checkpoint.
losses() {
    awk -F'\t' '$3 == "step" { row[$4] = $4 "\t" $6 "\t" $7 } END { for (s in row) print row[s] }' "$1/train.log" | sort -n
}

"$NW" train $COMMON --run "$WORK/t2" --threads 2 >/dev/null 2>&1
"$NW" train $COMMON --run "$WORK/t1" --threads 1 >/dev/null 2>&1
losses "$WORK/t2" > "$WORK/t2.loss"
losses "$WORK/t1" > "$WORK/t1.loss"
[ "$(wc -l < "$WORK/t2.loss")" -eq 20 ] && cmp -s "$WORK/t1.loss" "$WORK/t2.loss"
check $? "1 and 2 threads give bit-identical losses over 20 steps"
# Page text is deterministic except the wall-clock "train time" header field; normalize it,
# then compare normalized page bytes and checkpoint hashes as the real invariants.
sed 's/train time [0-9:]*//' "$WORK/t1/pages/01.txt" > "$WORK/t1.page.norm"
sed 's/train time [0-9:]*//' "$WORK/t2/pages/01.txt" > "$WORK/t2.page.norm"
cmp -s "$WORK/t1.page.norm" "$WORK/t2.page.norm" && [ "$(md5sum "$WORK/t1/pages/01.bin" | awk '{print $1}')" = "$(md5sum "$WORK/t2/pages/01.bin" | awk '{print $1}')" ]
check $? "1 and 2 threads give the identical final page (wall-clock train time normalized; page bytes + checkpoint hash compared)"

"$NW" train $COMMON --run "$WORK/r" --threads 2 --stop-after 7 >/dev/null 2>&1
"$NW" train $COMMON --run "$WORK/r" --threads 2 >/dev/null 2>&1
losses "$WORK/r" > "$WORK/r.loss"
cmp -s "$WORK/r.loss" "$WORK/t2.loss"
check $? "stop at step 7 and resume replays the uninterrupted loss log exactly"

"$NW" train $COMMON --run "$WORK/k" --threads 2 >/dev/null 2>&1 &
PID=$!
TRIES=0
while ! grep -q "	step	1[0-9]	" "$WORK/k/train.log" 2>/dev/null && [ $TRIES -lt 6000 ]; do sleep 0.05; TRIES=$((TRIES + 1)); done
kill -9 $PID; wait $PID 2>/dev/null
"$NW" train $COMMON --run "$WORK/k" --threads 2 >/dev/null 2>&1
losses "$WORK/k" > "$WORK/k.loss"
cmp -s "$WORK/k.loss" "$WORK/t2.loss"
check $? "kill -9 after step 10 and resume replays the uninterrupted loss log exactly"

cp -R "$WORK/r" "$WORK/c"
NEWEST=$(ls -t "$WORK/c/ckpt/"*.ckpt | head -1)
printf 'X' | dd of="$NEWEST" bs=1 seek=5000 conv=notrunc 2>/dev/null
"$NW" train $COMMON --run "$WORK/c" --steps 22 --threads 2 2>"$WORK/c.err" >/dev/null
grep -q "is invalid" "$WORK/c.err" && grep -q "resuming from slot" "$WORK/c.err"
check $? "a corrupted newest slot is rejected by CRC and the other slot resumes"

F="$WORK/sys"
mkdir -p "$F/sys/class/thermal/thermal_zone0" "$F/sys/class/power_supply/bat" "$F/sys/class/net/wlan0"
echo mtktscpu > "$F/sys/class/thermal/thermal_zone0/type"
echo 70000 > "$F/sys/class/thermal/thermal_zone0/temp"
echo 80 > "$F/sys/class/power_supply/bat/capacity"
echo 250 > "$F/sys/class/power_supply/bat/temp"
echo Charging > "$F/sys/class/power_supply/bat/status"
echo down > "$F/sys/class/net/wlan0/operstate"
"$NW" train $COMMON --run "$WORK/hot" --steps 3 --sysroot "$F" >/dev/null 2>&1 &
PID=$!
# Poll for the pause row instead of a fixed sleep: the trainer's first guard read must win the
# race against cooling at any model shape.
T=0
while ! grep -q "	pause	" "$WORK/hot/train.log" 2>/dev/null && [ $T -lt 45 ]; do sleep 1; T=$((T + 1)); done
awk -F'\t' '$3 == "pause" { p = 1 } $3 == "step" { s = 1 } END { exit !(p && !s) }' "$WORK/hot/train.log"
check $? "SoC at 70 C pauses before step 1 (placeholder threshold 60 C)"
echo 40000 > "$F/sys/class/thermal/thermal_zone0/temp"
wait $PID
grep -q "	resume	" "$WORK/hot/train.log" && grep -q "	step	3	" "$WORK/hot/train.log" && grep -q "paused 1 min: warm" "$WORK/hot/pages/01.txt"
check $? "cooling to 40 C resumes training and the next page notes the pause"
awk -F'\t' '$3 == "step" { print $NF }' "$WORK/hot/train.log" | head -1 | grep -q "wlan0=down;wirelessEnable=na"
check $? "log rows carry the wifi state"

echo 10 > "$F/sys/class/power_supply/bat/capacity"
echo Discharging > "$F/sys/class/power_supply/bat/status"
"$NW" train $COMMON --run "$WORK/low" --sysroot "$F" >/dev/null 2>&1
[ $? -eq 3 ] && grep -q "stopped: battery 10% and discharging at step 0" "$WORK/low/pages/01.txt"
check $? "battery 10% and discharging writes a stop page and exits 3"
echo 90 > "$F/sys/class/power_supply/bat/capacity"
echo Charging > "$F/sys/class/power_supply/bat/status"
"$NW" train $COMMON --run "$WORK/low" --sysroot "$F" >/dev/null 2>&1
grep -q "stopped: battery" "$WORK/low/pages/01.txt" && grep -q "	step	20	" "$WORK/low/train.log" && [ -f "$WORK/low/pages/02.txt" ]
check $? "a restart after the battery stop resumes from the stop checkpoint and keeps the stop page"

# Sensor fixtures from the on-device zone list: mtktscpu 29400, mtktspcb2/mtktsAP/mtktspcb3
# pinned at a 125000 placeholder. zone ROOT N TYPE MILLIDEG writes one thermal zone.
zone() { mkdir -p "$1/sys/class/thermal/thermal_zone$2"; echo "$3" > "$1/sys/class/thermal/thermal_zone$2/type"; echo "$4" > "$1/sys/class/thermal/thermal_zone$2/temp"; }
fake_battery() { mkdir -p "$1/sys/class/power_supply/bd71827_bat"; echo 80 > "$1/sys/class/power_supply/bd71827_bat/capacity"; echo 30 > "$1/sys/class/power_supply/bd71827_bat/temp"; echo Charging > "$1/sys/class/power_supply/bd71827_bat/status"; }
# Runs nightwriter on sysroot $1 into run dir $2 until a pause row appears (or 30 s), then kills it.
run_until_pause() {
    "$NW" train $COMMON --run "$2" --steps 3 --sysroot "$1" >/dev/null 2>"$2.err" &
    P=$!
    T=0
    while ! grep -q "	pause	" "$2/train.log" 2>/dev/null && [ $T -lt 600 ]; do sleep 0.05; T=$((T + 1)); done
    sleep 0.5
    kill $P 2>/dev/null; wait $P 2>/dev/null
}

S="$WORK/allbad"
zone "$S" 0 mtktspcb2 125000; zone "$S" 1 mtktsAP 125000; zone "$S" 2 mtktspcb3 125000; fake_battery "$S"
run_until_pause "$S" "$WORK/allbad"
awk -F'\t' '$3 == "sensor_missing" { m = 1 } $3 == "pause" { p = 1 } $3 == "step" { s = 1 } END { exit !(m && p && !s) }' "$WORK/allbad/train.log"
check $? "only 125000 placeholder zones: sensor_missing pause, never steps"

S="$WORK/onegood"
zone "$S" 0 mtktspcb2 125000; zone "$S" 1 mtktscpu 30000; zone "$S" 2 mtktsAP 125000; zone "$S" 3 mtktspcb3 125000
zone "$S" 4 bd71827_bat 2800; fake_battery "$S"
"$NW" train $COMMON --run "$WORK/onegood" --steps 3 --sysroot "$S" >/dev/null 2>&1
awk -F'\t' '$3 == "pause" { p = 1 } $3 == "step" { n++; if ($10 != "30.0") bad = 1 } END { exit !(n == 3 && !p && !bad) }' "$WORK/onegood/train.log"
check $? "mtktscpu 30000 among 125000 placeholders trains 3 steps with soc_c 30.0"

S="$WORK/nozones"
fake_battery "$S"
run_until_pause "$S" "$WORK/nozones"
awk -F'\t' '$3 == "sensor_missing" { m = 1 } $3 == "pause" { p = 1 } $3 == "step" { s = 1 } END { exit !(m && p && !s) }' "$WORK/nozones/train.log" &&
    grep -q "no readable soc sensor" "$WORK/nozones.err"
check $? "no thermal zones: sensor_missing row, pause, never steps"

S="$WORK/errzone"
zone "$S" 0 mtktscpu -274000; zone "$S" 1 mtktspcb2 125000; fake_battery "$S"
run_until_pause "$S" "$WORK/errzone"
awk -F'\t' '$3 == "sensor_missing" { m = 1 } $3 == "step" { s = 1 } END { exit !(m && !s) }' "$WORK/errzone/train.log"
check $? "mtktscpu reading an error code (-274000): sensor_missing pause, never steps"

S="$WORK/nobatt"
zone "$S" 0 mtktscpu 30000
run_until_pause "$S" "$WORK/nobatt"
awk -F'\t' '$3 == "sensor_missing" { m = 1 } $3 == "step" { s = 1 } END { exit !(m && !s) }' "$WORK/nobatt/train.log" &&
    grep -q "no readable battery sensor" "$WORK/nobatt.err"
check $? "no battery and no Charging status: sensor_missing pause, never steps"

S="$WORK/charging"
zone "$S" 0 mtktscpu 30000
mkdir -p "$S/sys/class/power_supply/usb"; echo Charging > "$S/sys/class/power_supply/usb/status"
"$NW" train $COMMON --run "$WORK/charging" --steps 3 --sysroot "$S" >/dev/null 2>&1
awk -F'\t' '$3 == "pause" || $3 == "sensor_missing" { p = 1 } $3 == "step" { n++ } END { exit !(n == 3 && !p) }' "$WORK/charging/train.log"
check $? "no battery gauge but status Charging: trains 3 steps"

cp -R "$WORK/t2" "$WORK/e"
"$NW" train $COMMON --run "$WORK/e" --extend-to 26 >/dev/null 2>&1
awk -F'\t' '$3 == "extend" { x = 1 } $3 == "step" && $4 > 20 { n++; if ($5 != "0.0001") bad = 1 } END { exit !(x && n == 6 && !bad) }' "$WORK/e/train.log" &&
    [ -f "$WORK/e/pages/02.txt" ]
check $? "--extend-to 26 on a finished 20-step run trains steps 21-26 at lr_min and writes a final page"
ROWS=$(grep -c "	step	" "$WORK/e/train.log")
"$NW" train $COMMON --run "$WORK/e" --extend-to 26 >/dev/null 2>&1
[ $? -eq 0 ] && [ "$(grep -c "	step	" "$WORK/e/train.log")" -eq "$ROWS" ]
check $? "repeating --extend-to 26 is a plain resume of the finished run"
"$NW" train $COMMON --run "$WORK/u" --stop-after 3 >/dev/null 2>&1
"$NW" train $COMMON --run "$WORK/u" --extend-to 30 >/dev/null 2>"$WORK/u.err"
[ $? -eq 2 ] && grep -q "not finished" "$WORK/u.err"
check $? "--extend-to refuses an unfinished run"

if command -v python3 >/dev/null 2>&1; then
    "$NW" sample --init "$WORK/t2/pages/01.bin" --data "$DATA" --page-chars 400 2>/dev/null > "$WORK/resample.txt"
    python3 - "$WORK/t2/pages/01.txt" "$WORK/resample.txt" <<'EOF'
import sys
page = open(sys.argv[1], 'rb').read()
body = page.split(b'\n\n', 1)[1].rsplit(b'\n\n---\n', 1)[0]
sys.exit(0 if body == open(sys.argv[2], 'rb').read() else 1)
EOF
    check $? "nightwriter sample reproduces the page text from its .bin"
else
    echo "SKIP_ON_DEVICE  nightwriter sample reproduces the page text from its .bin (provenance: PASS on Mac native and ARM-under-qemu; device has no python3)"
fi

if command -v python3 >/dev/null 2>&1 && command -v "${CC:-cc}" >/dev/null 2>&1; then
    ${CC:-cc} -O2 -o "$WORK/tinygpt" "$REPO/inference/tinygpt.c" -lm
    python3 -c "import sys; sys.stdout.buffer.write(bytes(range(256)) * 40)" > "$WORK/allbytes"
    "$WORK/tinygpt" "$WORK/t2/pages/00.bin" -n 40 -k 1 -p "It is a truth" > "$WORK/tiny.txt" 2>/dev/null
    "$NW" sample --init "$WORK/t2/pages/00.bin" --data "$WORK/allbytes" --top-k 1 --page-chars 40 --prompt "It is a truth" 2>/dev/null > "$WORK/greedy.txt"
    printf 'It is a truth' | cat - "$WORK/tiny.txt" | head -c 53 | cmp -s - "$WORK/greedy.txt" &&
        [ "$(tail -c 40 "$WORK/greedy.txt" | od -An -tu1 | tr -s ' ' '\n' | sort -u | wc -l)" -gt 10 ]
    check $? "inference/tinygpt.c and nightwriter decode the same 40 varied greedy bytes from the random-init page (inside ctx)"
else
    echo "SKIP_ON_DEVICE  inference/tinygpt.c and nightwriter decode the same 40 varied greedy bytes (provenance: PASS on Mac native and ARM-under-qemu; device has no cc/python3)"
fi

echo "work dir: $WORK"
[ "$FAILS" -eq 0 ] && echo "ALL PASS" || { echo "$FAILS FAILED"; exit 1; }
