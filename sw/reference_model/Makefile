# Makefile for the LonghornSilicon ACU precision-controller reference model.
#
# Targets:
#   all          build the test binary
#   test         build + run the test binary (verifies 143/143 vs RTL)
#   test-py      run the Python self-test suite
#   test-all     run both C++ and Python tests
#   shared       build libprecision_controller_ref.so for FFI from anywhere
#   static       build libprecision_controller_ref.a for static linking
#   clean        rm build artifacts

CXX        ?= g++
CXXFLAGS   ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -fPIC
LDFLAGS    ?=

REF_SRC    := precision_controller_ref.cpp
REF_HDR    := precision_controller_ref.hpp
TEST_SRC   := test_precision_controller_ref.cpp

TEST_BIN   := test_ref
LIB_SHARED := libprecision_controller_ref.so
LIB_STATIC := libprecision_controller_ref.a

# Path to the RTL test vectors. The tests look up the repo root themselves,
# but we also expose a generate-vectors helper for clean builds.
REPO_ROOT  := ../..
TESTVEC    := $(REPO_ROOT)/rtl/tb/testvectors/expected.hex
GEN_SCRIPT := $(REPO_ROOT)/analysis/gen_rtl_testvectors.py

.PHONY: all test test-py test-all shared static clean testvectors

all: $(TEST_BIN)

# Test binary: links the test program against the reference implementation.
$(TEST_BIN): $(TEST_SRC) $(REF_SRC) $(REF_HDR)
	$(CXX) $(CXXFLAGS) $(TEST_SRC) $(REF_SRC) -o $@ $(LDFLAGS)

# Shared library for FFI / dynamic linking from any language.
shared: $(LIB_SHARED)
$(LIB_SHARED): $(REF_SRC) $(REF_HDR)
	$(CXX) $(CXXFLAGS) -shared $(REF_SRC) -o $@ $(LDFLAGS)

# Static library for embedding.
static: $(LIB_STATIC)
$(LIB_STATIC): $(REF_SRC) $(REF_HDR)
	$(CXX) $(CXXFLAGS) -c $(REF_SRC) -o precision_controller_ref.o
	$(AR) rcs $@ precision_controller_ref.o

# Generate replay test vectors (the C++ test reads these from
# rtl/tb/testvectors/). Idempotent — skips if files already exist.
testvectors: $(TESTVEC)
$(TESTVEC):
	python3 $(GEN_SCRIPT)

# Run the C++ tests.
test: $(TEST_BIN) testvectors
	@./$(TEST_BIN)

# Run the Python test suite for parity check.
test-py: testvectors
	@python3 test_precision_controller_ref.py

# Run both — Python and C++ must agree against the same vectors.
test-all: test test-py
	@echo
	@echo "C++ and Python reference models both pass bit-exact verification."

clean:
	rm -f $(TEST_BIN) $(LIB_SHARED) $(LIB_STATIC) *.o
