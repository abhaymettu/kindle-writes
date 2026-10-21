#!/bin/sh
# kindle/deploy.sh - copy the binary and weights to the Kindle over SSH.
#
# Usage: ./deploy.sh [kindle-host] [weights.bin]

KINDLE="${1:-100.83.27.105}"
WEIGHTS="${2:-checkpoints/final.bin}"
SSH_KEY="${HOME}/.ssh/kindle_koreader"
PORT=2222
DEST="/mnt/us/kindle-writes"

if [ ! -f inference/kindle-writes-arm ]; then
    echo "error: inference/kindle-writes-arm not found. Run 'make arm' first." >&2
    exit 1
fi
if [ ! -f "$WEIGHTS" ]; then
    echo "error: $WEIGHTS not found. Train a model first." >&2
    exit 1
fi

echo "deploying to ${KINDLE}:${DEST} ..."
ssh -i "$SSH_KEY" -p "$PORT" -o ConnectTimeout=10 "root@${KINDLE}" "mkdir -p ${DEST}" || exit 1
scp -i "$SSH_KEY" -P "$PORT" inference/kindle-writes-arm "root@${KINDLE}:${DEST}/"
scp -i "$SSH_KEY" -P "$PORT" "$WEIGHTS" "root@${KINDLE}:${DEST}/final.bin"
scp -i "$SSH_KEY" -P "$PORT" kindle/write.sh "root@${KINDLE}:${DEST}/"
ssh -i "$SSH_KEY" -p "$PORT" "root@${KINDLE}" "chmod +x ${DEST}/kindle-writes-arm ${DEST}/write.sh"
echo "done. ssh in and run: ${DEST}/write.sh"
