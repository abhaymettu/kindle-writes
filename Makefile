CC       ?= cc
ARM_CC   ?= arm-linux-gnueabihf-gcc
CFLAGS   := -O2 -Wall -Wextra -std=c99
LDFLAGS  := -lm

SRC      := inference/tinygpt.c
NATIVE   := inference/kindle-writes
ARM      := inference/kindle-writes-arm

.PHONY: all native arm clean test

all: native

native: $(NATIVE)

$(NATIVE): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

arm: $(ARM)

$(ARM): $(SRC)
	$(ARM_CC) $(CFLAGS) -static -mfloat-abi=hard -mfpu=neon -o $@ $< $(LDFLAGS)

clean:
	rm -f $(NATIVE) $(ARM)

test: native
	cd tests && python3 -m pytest -v test_model.py test_export.py test_inference.py
