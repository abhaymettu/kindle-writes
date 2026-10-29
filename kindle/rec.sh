#!/bin/sh
# rec.sh PROMPT   On the Kindle: record the screen while show.sh draws the title card, a live kw page, the loss curve
# and the proof card. Frames are gzip'd raw framebuffer dumps in /mnt/base-us/kwrec with an uptime stamp each
# (times) and a stamp per card (marks). Turn them into a video on your computer with kindle/mkvideo.py.
# KOReader is paused while it records so it doesn't draw over the cards, and resumed on exit.
R=/mnt/base-us/kwrec; rm -rf $R; mkdir -p $R; cd /mnt/base-us/kindle-writes || exit 2
P=$(pidof reader.lua)
trap 'touch $R/stop; for p in $P; do kill -CONT $p; done' EXIT
for p in $P; do kill -STOP $p; done
( n=0; while [ ! -f $R/stop ]; do   # as fast as dd allows, about 6 frames a second
    u=$(cut -d" " -f1 /proc/uptime); dd if=/dev/fb0 bs=1072 count=1448 2>/dev/null | gzip -1 > $R/f$(printf %04d $n).gz
    echo "$n $u" >> $R/times; n=$((n + 1)); done ) &
mark() { echo "$1 $(cut -d" " -f1 /proc/uptime)" >> $R/marks; }
mark title; sh klib-bin/show.sh title; sleep 4
mark kw;    sh klib-bin/show.sh kw "${1:-The phone rang twice.}"; mark page; sleep 5
mark curve; sh klib-bin/show.sh curve; sleep 4
mark proof; sh klib-bin/show.sh proof; sleep 4
mark end; touch $R/stop; wait
