#!/bin/sh
# kindle/write.sh - run the model on the Kindle and write to the e-ink screen.
#
# Expects kindle-writes-arm and a .bin weights file in the same directory.
# Uses eips (Kindle's built-in e-ink text renderer) for output.
#
# Usage: ./write.sh [weights.bin] [prompt]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${SCRIPT_DIR}/kindle-writes-arm"
WEIGHTS="${1:-${SCRIPT_DIR}/final.bin}"
PROMPT="${2:-Once upon a time}"
TOKENS=200
TEMP=0.8

if [ ! -f "$BIN" ]; then
    echo "error: $BIN not found" >&2
    exit 1
fi
if [ ! -f "$WEIGHTS" ]; then
    echo "error: $WEIGHTS not found" >&2
    exit 1
fi

# clear the screen
eips -c 2>/dev/null

# header
eips 0 0 "kindle-writes" 2>/dev/null
eips 0 1 "---" 2>/dev/null

# generate and display line by line
ROW=3
COL=0
MAX_COL=48  # approximate characters per line on a Kindle

"$BIN" "$WEIGHTS" -n "$TOKENS" -t "$TEMP" -p "$PROMPT" --char 2>/dev/null | while IFS= read -r -n 1 ch; do
    if [ -z "$ch" ]; then
        ch=" "
    fi

    # newline handling
    case "$ch" in
        "$(printf '\n')")
            ROW=$((ROW + 1))
            COL=0
            if [ "$ROW" -gt 38 ]; then
                ROW=3
                eips -c 2>/dev/null
                eips 0 0 "kindle-writes" 2>/dev/null
                eips 0 1 "---" 2>/dev/null
            fi
            continue
            ;;
    esac

    # wrap at line end
    if [ "$COL" -ge "$MAX_COL" ]; then
        ROW=$((ROW + 1))
        COL=0
        if [ "$ROW" -gt 38 ]; then
            ROW=3
            eips -c 2>/dev/null
            eips 0 0 "kindle-writes" 2>/dev/null
            eips 0 1 "---" 2>/dev/null
        fi
    fi

    # eips prints a string at (col, row)
    eips "$COL" "$ROW" "$ch" 2>/dev/null
    COL=$((COL + 1))
done

eips 0 $((ROW + 2)) "[done]" 2>/dev/null
