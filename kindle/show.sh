#!/bin/sh
# show.sh CARD [args]   On the Kindle: draw one photo card with fbink in the Kindle's own Bookerly font.
# Every word of model text on these cards is generated on this device by the klib weights.
#   title                      what this is, numbers from klib/train.log
#   write PROMPT SEED [PAGE]   sample a page now and draw it (PAGE defaults to the final weights)
#   before PROMPT SEED         the same prompt and seed from the first weights (step 24) and the last
#   curve                      held-out loss over training, drawn from train.log
#   proof                      raw device facts in the terminal font
#   kw PROMPT                  run kw for real and draw its transcript in the terminal font
cd /mnt/base-us/kindle-writes || exit 2
FB=/mnt/us/libkh/bin/fbink
F=/usr/java/lib/fonts
TT="regular=$F/Bookerly-Regular.ttf,bold=$F/Bookerly-Bold.ttf,italic=$F/Bookerly-Italic.ttf"
BIN=q1025-bpe-bin/nightwriter-arm-bpe-lex
TOK=klib-data/tok1024
LOG=klib/train.log
FINAL=$(ls klib/pages/*.bin | tail -1)
FIRST=klib/pages/00.bin

t() { $FB -q -b -t "$TT,$1" "$2"; }   # t "size=..,top=..,left=..,right=..[,style=..][,format]" TEXT  (no refresh)
gen() {  # gen WEIGHTS PROMPT SEED CHARS -> text on stdout, one paragraph per line
    nice -n 10 $BIN sample --init "$1" --tokens $TOK/tokens.bin --vocab $TOK/vocab.bin --prompt "$2" \
        --sample-seed "$3" --temp 0.65 --top-k 8 --page-chars "$4" --threads 1 --lexicon 0 2>/dev/null
}
stepof() { head -1 "${1%.bin}.txt" | sed 's/.*| step \([0-9]*\) |.*/\1/'; }
ho() { awk -F'\t' -v s="$1" '$3=="val_char" && $4==s' $LOG | tail -1 | cut -f7; }
# Bold the prompt, escape the model's own markdown characters, cut at the last full sentence.
page() { sed 's/[*_]//g' | awk -v p="$1" -v n="$2" '
    { s = s (NR > 1 ? "\n" : "") $0 } END {
      s = substr(s, length(p) + 1); s = substr(s, 1, n)
      c = 0; for (i = length(s); i > 0; i--) { ch = substr(s, i, 1); if (ch ~ /[.!?"]/) { c = i; break } }
      if (c > n / 2) s = substr(s, 1, c)
      sub(/\n+["]$/, "", s)
      printf "**%s**%s", p, s }'; }
start() { $FB -q -k -b; }
show() { $FB -q -s -f -W GC16; }
foot() { t "size=7,top=1340,left=80,right=80,style=ITALIC" "$1"; }

case "$1" in
title)
    st=$(awk -F'\t' '$3=="step"{s=$4} END{print s}' $LOG | sed 's/\([0-9]\)\([0-9]\{3\}\)$/\1,\2/')
    hrs=$(awk -F'\t' '$3=="step"{t=$2} END{printf "%.0f", t/3600}' $LOG)
    nb=$(ls /mnt/us/documents/Library/*.epub | wc -l | tr -d ' ')
    start
    t "size=20,top=230,left=90,right=90,style=BOLD" "my kindle trained its own language model"
    t "size=11,top=560,left=90,right=90" "trained from scratch on the $nb books on it, on its own cpu"
    t "size=10,top=860,left=90,right=90,format" "1.07m params, $st steps, $hrs hours on 2 arm cores"
    show ;;
write)
    W=${4:-$FINAL}
    txt=$(gen "$W" "$2" "$3" 600 | page "$2" 560)
    start
    t "size=11,top=110,bottom=170,left=90,right=90,format" "$txt"
    show ;;
before)
    a=$(gen $FIRST "$2" "$3" 260 | page "$2" 230)
    b=$(gen "$FINAL" "$2" "$3" 380 | page "$2" 340)
    start
    t "size=8,top=90,left=90,right=90,style=BOLD" "step $(stepof $FIRST), before training"
    t "size=10,top=160,bottom=760,left=90,right=90,format" "$a"
    t "size=8,top=700,left=90,right=90,style=BOLD" "step $(stepof "$FINAL"), a day later"
    t "size=10,top=770,bottom=170,left=90,right=90,format" "$b"
    show ;;
curve)
    start
    t "size=8,top=110,left=90,right=90,style=BOLD" "loss on pages it never trained on"
    # plot box: x 140..980, y 420..1180 ; y range 1.3..3.0
    awk -F'\t' '$3=="val_char"{print $4, $7}' $LOG | awk -v last="$(awk -F'\t' '$3=="step"{s=$4} END{print s}' $LOG)" '
      { x = 140 + int(840 * $1 / last); y = 420 + int(760 * (3.0 - $2) / 1.7); if (y > 1180) y = 1180
        h = 1180 - y + 1; if (h < 3) h = 3
        printf "%d %d %d\n", x, y, h }' | while read x y h; do
        $FB -q -b -B BLACK -k top=$y,left=$x,width=6,height=6
    done
    $FB -q -b -B BLACK -k top=1180,left=140,width=846,height=3
    $FB -q -b -B BLACK -k top=420,left=137,width=3,height=763
    a=$(awk -F'\t' '$3=="val_char"{print $7; exit}' $LOG); z=$(awk -F'\t' '$3=="val_char"{v=$7} END{print v}' $LOG)
    t "size=8,top=420,left=175,right=90,style=BOLD" "$a"
    t "size=8,top=1040,left=700,right=90,style=BOLD" "$z"
    t "size=7,top=1200,left=135,right=90" "0"
    t "size=7,top=1200,left=470,right=90" "training steps"
    t "size=7,top=1200,left=905,right=60" "6,600"
    show ;;
proof)
    start
    i=3
    { echo "root@kindle:/mnt/base-us/kindle-writes"
      echo " "
      echo "> grep Hardware /proc/cpuinfo"
      grep -i hardware /proc/cpuinfo
      echo "> nproc; free -m"
      echo "$(grep -c ^processor /proc/cpuinfo) cores, $(free -m | awk '/Mem:/{print $2}') MB RAM"
      echo " "
      echo "> ls -l klib-data/text.txt"
      ls -l klib-data/text.txt | awk '{print $5, "bytes of text from the library"}'
      echo " "
      echo "> tail -4 klib/train.log"
      tail -4 $LOG | cut -f1,3,4,7
      echo " "
      echo "> head -1 $(basename "$FINAL" .bin).txt"
      head -1 "${FINAL%.bin}.txt" | tr '|' '\n' | sed 's/^ *//'
      echo " "
      echo "> sha256sum $FINAL"
      sha256sum "$FINAL" | cut -c1-32
      echo " "
      echo "> uptime"; uptime | cut -d, -f1,2
    } | cut -c1-64 | while IFS= read -r l; do eips 2 $i "$l" >/dev/null; i=$((i + 1)); done
    eips '' >/dev/null 2>&1
    exit 0 ;;
kw)
    shift
    start
    i=3
    # Draw each line as kw prints it, so the screen shows the run as it happens (eips drops text after a %).
    { echo "> kw $*"; echo " "; sh /mnt/us/koreader/scripts/kw "$@"; } 2>&1 | while IFS= read -r l; do
        echo "${l//%/}" | fold -s -w 64 | while IFS= read -r w; do eips 2 $i "${w:- }" >/dev/null; i=$((i + 1)); done
        i=$((i + $(echo "${l//%/}" | fold -s -w 64 | wc -l)))
    done
    eips '' >/dev/null 2>&1
    exit 0 ;;
*) echo "usage: show.sh title|write PROMPT SEED [PAGE]|before PROMPT SEED|curve|proof"; exit 2 ;;
esac
