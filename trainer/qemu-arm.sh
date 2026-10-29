#!/bin/sh
# Run trainer/nightwriter-arm under qemu-arm in docker, as a drop-in for the native binary:
#   trainer/gradcheck.py --bin trainer/qemu-arm.sh
#   trainer/smoke.sh trainer/qemu-arm.sh
# The repo and $TMPDIR are mounted at their host paths so file arguments resolve unchanged.
# Needs the image built from the Dockerfile in VERIFICATION.md (QEMU_IMAGE, default localhost/kw-qemu).
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/.." && pwd)"
TMP="$(cd "${TMPDIR:-/tmp}" && pwd -P)"
exec docker run --rm -i -v "$REPO":"$REPO" -v "$TMP":"$TMP" -v /tmp:/tmp -w "$PWD" \
    "${QEMU_IMAGE:-localhost/kw-qemu}" qemu-arm "$REPO/trainer/nightwriter-arm" "$@"
