#!/bin/sh
# Build nightwriter for the Mac (native) and the Kindle (armv7 hard-float, static).
#   trainer/build.sh            both
#   trainer/build.sh native     Mac only
#   trainer/build.sh arm        Kindle only (needs docker and an image with arm-linux-gnueabihf-gcc)
set -eu

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/.." && pwd)"
ARM_IMAGE="${ARM_IMAGE:-localhost/armtools}"
HASH="$(git -C "$REPO" describe --always --dirty 2>/dev/null || echo unknown)"
SRC="trainer/gpt.c trainer/nightwriter.c"
WARN="-std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-format-truncation"

native() {
    (cd "$REPO" && ${CC:-cc} -O3 $WARN -DGIT_HASH="\"$HASH\"" -o trainer/nightwriter $SRC -lm -lpthread)
    echo "built trainer/nightwriter ($HASH)"
}

# -mfpu=neon-fp-armv8 is what makes vfmaq_f32 available; plain -mfpu=neon has no fused multiply-add.
# Static glibc needs the whole libpthread archive or condition variables can crash at runtime.
arm() {
    docker run --rm -v "$REPO":/src -w /src "$ARM_IMAGE" \
        arm-linux-gnueabihf-gcc -O3 -march=armv7-a -mtune=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard \
        $WARN -DGIT_HASH="\"$HASH\"" -static -o trainer/nightwriter-arm $SRC \
        -lm -Wl,--whole-archive -lpthread -Wl,--no-whole-archive
    echo "built trainer/nightwriter-arm ($HASH)"
}

case "${1:-all}" in
    native) native ;;
    arm) arm ;;
    all) native; arm ;;
    *) echo "usage: $0 [native|arm|all]" >&2; exit 2 ;;
esac
