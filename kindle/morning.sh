#!/bin/sh
# morning.sh [now]   On the Kindle: once a day at 07:00 (or once immediately with "now"), sample a page from the
# newest klib weights, save it as a document in the Kindle library, and draw it on the screen with eips.
# Everything runs on the device: weights, sampler, text and display.
cd /mnt/base-us/kindle-writes || exit 2
BIN=q1025-bpe-bin/nightwriter-arm-bpe-lex
TOK=klib-data/tok1024
DOC=/mnt/base-us/documents
ST=klib-morning.state
OPENERS="When I woke up that morning,|She looked at him and said, \"|The first thing you need to know is|It was late when the phone rang, and|I have always believed that"

page() {
    day=$(( $(cat $ST 2>/dev/null | wc -l) + 1 ))
    W=$(ls klib/pages/*.bin | tail -1)
    step=$(head -1 "${W%.bin}.txt" | sed 's/.*| step \([0-9]*\) |.*/\1/')
    ho=$(awk -F'\t' -v s="$step" '$3=="val_char" && $4==s' klib/train.log | tail -1 | cut -f7)
    prev=$(tail -1 $ST 2>/dev/null | cut -d' ' -f3)
    n=$(( (day - 1) % 5 + 1 ))
    prompt=$(echo "$OPENERS" | cut -d'|' -f$n)
    out=klib-morning/day$day.txt; mkdir -p klib-morning
    nice -n 10 $BIN sample --init "$W" --tokens $TOK/tokens.bin --vocab $TOK/vocab.bin --prompt "$prompt" \
        --sample-seed $((1000 + day)) --temp 0.65 --top-k 8 --page-chars 900 --threads 1 --lexicon 0 > $out 2>/dev/null
    {
        echo "Kindle Writes, morning $day"
        echo "Trained on this Kindle, from the 12 books in its library. Step $step of 6600. Held-out $ho nats/char${prev:+ (yesterday $prev)}."
        echo
        cat $out
    } > "$DOC/Kindle Writes - morning $day.txt"
    echo "$day $(date +%Y-%m-%dT%H:%M) $ho $step $(sha256sum "$W" | cut -c1-16) $(sha256sum $out | cut -c1-16)" >> $ST
    eips -c
    eips 2 2 "Kindle Writes, morning $day   step $step/6600"
    r=5
    tr '\n' ' ' < $out | fold -s -w 62 | head -48 | while IFS= read -r line; do
        eips 2 $r "$line"; r=$((r + 1))
    done
}

[ "$1" = now ] && { page; exit 0; }
while :; do
    now=$(date +%H%M)
    if [ "$now" = 0700 ]; then page; sleep 70; fi
    [ -f klib/train.log ] && tail -1 klib/train.log | grep -q "	exit	6600" && [ ! -f klib-morning/final.done ] && { page; touch klib-morning/final.done; }
    sleep 30
done
