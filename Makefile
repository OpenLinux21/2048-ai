# Build the headless shim shared library around the unmodified games/2048.c
# and run the C self-test plus the Python test suite.
#
#   make          -> build games/libshim2048.so
#   make test     -> C oracle test + pytest
#   make clean    -> remove build artifacts

CC      ?= cc
CFLAGS  ?= -O2 -Wall -shared -fPIC
PY      ?= .venv/bin/python
SHIM    := games/libshim2048.so

.PHONY: all test clean

all: $(SHIM)

$(SHIM): csrc/shim2048.c games/2048.c
	$(CC) $(CFLAGS) -o $@ csrc/shim2048.c

# The stock binary's built-in oracle: 13 slide/merge test vectors.
c-test: games/2048
	./games/2048 test

test: $(SHIM) c-test
	$(PY) -m pytest tests/ -q

clean:
	rm -f $(SHIM)
