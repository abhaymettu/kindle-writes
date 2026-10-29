#!/bin/sh
# grab.sh OUT.png   Copy the Kindle's visible framebuffer (1072x1448, 8-bit gray) to a PNG (macOS sips).
# Set KINDLE_HOST (and KINDLE_PORT if not 22) in your environment or in .env at the repo root.
[ -z "$KINDLE_HOST" ] && [ -f "$(dirname "$0")/../.env" ] && . "$(dirname "$0")/../.env"
out=${1:?usage: grab.sh OUT.png}; tmp=$(mktemp -t kfb).pgm
{ printf 'P5\n1072 1448\n255\n'
  ssh -p "${KINDLE_PORT:-22}" "root@${KINDLE_HOST:?set KINDLE_HOST}" 'dd if=/dev/fb0 bs=1072 count=1448 2>/dev/null'; } > "$tmp"
sips -s format png "$tmp" --out "$out" >/dev/null && rm -f "$tmp"
