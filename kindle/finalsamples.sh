#!/bin/sh
# finalsamples.sh   On the Kindle: 30 fixed samples from the final klib weights, written to klib-final/.
# 5 prompts x 4 seeds (same as host/samples.sh) plus 5 new prompts x 2 seeds. Same sampler settings everywhere.
cd /mnt/base-us/kindle-writes || exit 2
BIN=q1025-bpe-bin/nightwriter-arm-bpe-lex
TOK=klib-data/tok1024
W=$(ls klib/pages/*.bin | tail -1)
O=klib-final; mkdir -p $O
s() {
    nice -n 10 $BIN sample --init "$W" --tokens $TOK/tokens.bin --vocab $TOK/vocab.bin --page-chars 600 --threads 1 \
        --prompt "$1" --sample-seed $2 --temp 0.65 --top-k 8 --lexicon 0 > $O/$3-$2.txt 2>> $O/stderr.log
}
i=0
for p in 'One morning,' 'She looked at him and said, "' 'The little' 'When they got home,' 'He did not know'; do
    i=$((i + 1)); for seed in 1342 2026 71 99; do s "$p" $seed p$i; done
done
i=0
for p in 'The door opened and' 'I remember the summer' 'The phone rang twice.' 'Nobody in the town' 'At the end of the day,'; do
    i=$((i + 1)); for seed in 7 2028; do s "$p" $seed n$i; done
done
{ echo "$W"; sha256sum "$W"; } > $O/WEIGHTS
sha256sum $O/*.txt > $O/SHA256SUMS
echo done
