#!/bin/sh
# kindle/reproduce.sh check            Regenerate my 30 published samples from the published weights and compare
#                                      them byte for byte. Needs a C compiler, no Kindle.
# kindle/reproduce.sh kindle [HOST]    Run the whole pipeline on your own Kindle over SSH: epubs to text, the
#                                      tokenizer, then the same 6,600-step training run, on the Kindle's CPU.
#
# Env for kindle mode: KINDLE_HOST (or .env), EPUBS (glob on the Kindle, default /mnt/us/documents/*.epub),
# DEST (default /mnt/base-us/kindle-writes-repro). SSH is root on port 2222 with ~/.ssh/kindle_koreader,
# the same as deploy.sh.
set -eu
cd "$(dirname "$0")/.."
D=docs/klib-2026-10-28
FLAGS="--threads 2 --soc-pause 40 --soc-resume 36 --batt-pause 36 --batt-resume 33 --batt-stop 80 --steps 6600
 --seed 1337 --data-seed 2027 --wd 0.1 --lr 1e-3 --lr-min 1e-4 --warmup 100 --layers 4 --dmodel 128 --heads 4
 --dff 512 --ctx 128"

check() {
    out=$(mktemp -d)
    ${CC:-cc} -O3 -std=c11 -o "$out/nightwriter" trainer/gpt.c trainer/nightwriter.c -lm -lpthread
    s() {
        "$out/nightwriter" sample --init $D/weights/51.bin --tokens $D/weights/train-mask.bin --vocab $D/weights/vocab.bin \
            --page-chars 600 --threads 1 --prompt "$1" --sample-seed "$2" --temp 0.65 --top-k 8 --lexicon 0 \
            > "$out/klib-final/$3-$2.txt" 2>/dev/null
    }
    mkdir "$out/klib-final"
    i=0
    for p in 'One morning,' 'She looked at him and said, "' 'The little' 'When they got home,' 'He did not know'; do
        i=$((i + 1)); for seed in 1342 2026 71 99; do s "$p" $seed p$i; done
    done
    i=0
    for p in 'The door opened and' 'I remember the summer' 'The phone rang twice.' 'Nobody in the town' 'At the end of the day,'; do
        i=$((i + 1)); for seed in 7 2028; do s "$p" $seed n$i; done
    done
    sums=$(cd "$out" && shasum -a 256 -c "$OLDPWD/$D/samples/SHA256SUMS")
    echo "$sums" | tail -3
    echo "$(echo "$sums" | grep -c ': OK$') of 30 samples match the Kindle's, byte for byte ($out/klib-final)"
}

kindle() {
    [ -z "${KINDLE_HOST:-}" ] && [ -f .env ] && . ./.env
    H=${1:-${KINDLE_HOST:?set KINDLE_HOST or pass the host}}
    DEST=${DEST:-/mnt/base-us/kindle-writes-repro}
    EPUBS=${EPUBS:-/mnt/us/documents/*.epub}
    k() { ssh -i "$HOME/.ssh/kindle_koreader" -p 2222 -o ConnectTimeout=10 "root@$H" "$@"; }
    k "mkdir -p $DEST/data/tok1024" || exit 1
    for f in kindle/prep.lua kindle/bpe.lua kindle/nwrun.sh $D/bin/nightwriter-arm-bpe; do
        k "cat > $DEST/$(basename "$f")" < "$f"
    done
    k "cd $DEST && chmod +x nightwriter-arm-bpe &&
       lua prep.lua data $EPUBS &&
       lua bpe.lua data/text.txt \$(cat data/split.txt) 1024 data/tok1024 &&
       sha256sum data/text.txt data/tok1024/tokens.bin data/tok1024/vocab.bin &&
       sh nwrun.sh fresh ./nightwriter-arm-bpe run --tokens data/tok1024/tokens.bin --vocab data/tok1024/vocab.bin $(echo $FLAGS)"
    echo "mine were text e90f0e54, tokens 873218f5, vocab 616086f1 (different books give different hashes)"
    echo "watch it: ssh in and run   tail -f $DEST/run/train.log"
}

case "${1:-}" in
    check) check ;;
    kindle) shift; kindle "$@" ;;
    *) echo "usage: $0 check | kindle [HOST]" >&2; exit 2 ;;
esac
