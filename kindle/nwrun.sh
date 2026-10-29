#!/bin/sh
# nwrun.sh MODE BIN RUNDIR [train flags...]   MODE = fresh | resume
# Fail-closed launcher for the frozen nightwriter BPE trainer (binary unchanged).
#   fresh : refuses if RUNDIR exists; requires --steps (no --budget-hours) so the schedule is fixed up front.
#   resume: refuses unless a checkpoint file AND pages/01.txt exist. With 01.txt present the binary either
#           resumes or dies ("none is valid"); it cannot fall back to a fresh start. Copies ckpt/ aside first,
#           then fails (exit 2, printing the PID to stop) if stderr lacks "resuming from slot" within 60 s.
#           The script never kills; stopping is done by literal PID (harness rule).
# Both modes require every operational guard flag, and append one manifest row per launch to RUNDIR.launches.tsv
# (operational flags are not stored in checkpoints, so this file is their record).
# Env: SYSROOT (sensor root, default empty = real /sys), NICE (default 10).
set -eu
MODE=$1; BIN=$2; RUN=$3; shift 3
SYS=${SYSROOT:-}
NICE=${NICE:-10}
die() { echo "nwrun: $*" >&2; exit 2; }
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
rd() { cat "$1" 2>/dev/null || echo NA; }

need() {   # flag must appear in the argument list
    f=$1; shift
    for a in "$@"; do [ "$a" = "$f" ] && return 0; done
    die "missing required flag $f"
}
val() {    # value following flag
    f=$1; shift
    while [ $# -gt 1 ]; do [ "$1" = "$f" ] && { echo "$2"; return 0; }; shift; done
    echo NA
}
for f in --tokens --vocab --threads --soc-pause --soc-resume --batt-pause --batt-resume --batt-stop; do need $f "$@"; done
for a in "$@"; do [ "$a" = "--run" ] && die "pass RUNDIR positionally, not --run"; done

case $MODE in
fresh)
    [ -e "$RUN" ] && die "fresh refused: $RUN exists"
    need --steps "$@"
    for a in "$@"; do [ "$a" = "--budget-hours" ] && die "fresh refused: use --steps, not --budget-hours"; done
    ;;
resume)
    [ -d "$RUN" ] || die "resume refused: $RUN does not exist"
    { [ -f "$RUN/ckpt/A.ckpt" ] || [ -f "$RUN/ckpt/B.ckpt" ]; } || die "resume refused: no checkpoint file in $RUN/ckpt"
    [ -f "$RUN/pages/01.txt" ] || die "resume refused: $RUN/pages/01.txt absent (binary could start fresh)"
    ;;
*) die "mode must be fresh or resume" ;;
esac

M="$RUN.launches.tsv"
N=1; [ -f "$M" ] && N=$(( $(wc -l < "$M") ))
[ -f "$M" ] || printf 'utc\tmode\tn\tbin_sha256\ttokens_sha256\tvocab_sha256\tnice\tthreads\tsoc_pause\tsoc_resume\tbatt_pause\tbatt_resume\tbatt_stop\tsteps\tac_online\tbatt_status\tbatt_pct\tbatt_c\tsoc_mC\targv\n' > "$M"
if [ "$MODE" = resume ]; then
    B="$RUN/ckpt-before-launch$N"
    [ -e "$B" ] && die "backup dir $B exists"
    mkdir "$B"
    for s in A B; do [ -f "$RUN/ckpt/$s.ckpt" ] && cp -p "$RUN/ckpt/$s.ckpt" "$B/"; done
fi
SOC=NA
for z in "$SYS"/sys/class/thermal/thermal_zone*; do [ "$(rd "$z/type")" = mtktscpu ] && SOC=$(rd "$z/temp"); done
BD="$SYS/sys/class/power_supply/bd71827_bat"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODE" "$N" "$(sha "$BIN")" "$(sha "$(val --tokens "$@")")" "$(sha "$(val --vocab "$@")")" \
    "$NICE" "$(val --threads "$@")" "$(val --soc-pause "$@")" "$(val --soc-resume "$@")" "$(val --batt-pause "$@")" \
    "$(val --batt-resume "$@")" "$(val --batt-stop "$@")" "$(val --steps "$@")" \
    "$(rd "$SYS/sys/class/power_supply/bd71827_ac/online")" "$(rd "$BD/status")" "$(rd "$BD/capacity")" "$(rd "$BD/temp")" "$SOC" \
    "$*" >> "$M"

ERR="$RUN.stderr.$N"
nohup nice -n "$NICE" "$BIN" train --run "$RUN" "$@" > "$ERR" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$RUN.pid"
if [ "$MODE" = resume ]; then
    i=0
    while [ $i -lt 60 ]; do
        grep -q "^resuming from slot" "$ERR" 2>/dev/null && { echo "nwrun: resumed pid $PID, log $ERR"; exit 0; }
        kill -0 "$PID" 2>/dev/null || { echo "nwrun: resume FAILED, pid $PID exited:" >&2; cat "$ERR" >&2; exit 1; }
        sleep 1; i=$((i + 1))
    done
    die "resume FAILED: no 'resuming from slot' line in 60 s; trainer pid $PID may still run: SIGTERM it by that literal PID"
fi
sleep 2
kill -0 "$PID" 2>/dev/null || { grep -q '^nightwriter:' "$ERR" && { echo "nwrun: fresh FAILED:" >&2; cat "$ERR" >&2; exit 1; }; }
grep -q "^resuming from slot" "$ERR" && die "fresh FAILED: binary resumed; SIGTERM pid $PID by that literal PID"
echo "nwrun: fresh pid $PID, log $ERR"
