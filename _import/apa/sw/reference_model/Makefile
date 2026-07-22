# Makefile for the LonghornSilicon ACU reference models.
#
# Covers both blocks:
#   1. precision controller   (PC_*)
#   2. MAC array              (MAC_*)
#
# Targets:
#   all                build all test binaries
#   test               run all C++ tests (pc + mac)
#   test-pc            run only the precision-controller C++ test
#   test-mac           run only the MAC-array C++ test
#   test-py            run all Python tests (pc + mac + integration)
#   test-all           build + run everything (Python + C++)
#   shared             build both shared libraries
#   static             build both static libraries
#   integration        run the integration example
#   clean              rm all build artifacts

CXX        ?= g++
CXXFLAGS   ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -fPIC
LDFLAGS    ?=

# Block 1: precision controller
PC_SRC     := precision_controller_ref.cpp
PC_HDR     := precision_controller_ref.hpp
PC_TEST    := test_precision_controller_ref.cpp
PC_TEST_BIN := test_pc
PC_SHARED  := libprecision_controller_ref.so
PC_STATIC  := libprecision_controller_ref.a

# Block 2: MAC array
MAC_SRC    := mac_array_ref.cpp
MAC_HDR    := mac_array_ref.hpp
MAC_TEST   := test_mac_array_ref.cpp
MAC_TEST_BIN := test_mac
MAC_SHARED := libmac_array_ref.so
MAC_STATIC := libmac_array_ref.a

REPO_ROOT  := ../..
TESTVEC    := $(REPO_ROOT)/rtl/tb/testvectors/expected.hex
GEN_SCRIPT := $(REPO_ROOT)/analysis/gen_rtl_testvectors.py

.PHONY: all test test-pc test-mac test-py test-all shared static \
        integration clean testvectors

all: $(PC_TEST_BIN) $(MAC_TEST_BIN)

# -- precision controller --------------------------------------------------
$(PC_TEST_BIN): $(PC_TEST) $(PC_SRC) $(PC_HDR)
	$(CXX) $(CXXFLAGS) $(PC_TEST) $(PC_SRC) -o $@ $(LDFLAGS)
$(PC_SHARED): $(PC_SRC) $(PC_HDR)
	$(CXX) $(CXXFLAGS) -shared $(PC_SRC) -o $@ $(LDFLAGS)
$(PC_STATIC): $(PC_SRC) $(PC_HDR)
	$(CXX) $(CXXFLAGS) -c $(PC_SRC) -o precision_controller_ref.o
	$(AR) rcs $@ precision_controller_ref.o

# -- MAC array -------------------------------------------------------------
$(MAC_TEST_BIN): $(MAC_TEST) $(MAC_SRC) $(MAC_HDR)
	$(CXX) $(CXXFLAGS) $(MAC_TEST) $(MAC_SRC) -o $@ $(LDFLAGS)
$(MAC_SHARED): $(MAC_SRC) $(MAC_HDR)
	$(CXX) $(CXXFLAGS) -shared $(MAC_SRC) -o $@ $(LDFLAGS)
$(MAC_STATIC): $(MAC_SRC) $(MAC_HDR)
	$(CXX) $(CXXFLAGS) -c $(MAC_SRC) -o mac_array_ref.o
	$(AR) rcs $@ mac_array_ref.o

# Aggregate library targets
shared: $(PC_SHARED) $(MAC_SHARED)
static: $(PC_STATIC) $(MAC_STATIC)

# -- test vectors ----------------------------------------------------------
testvectors: $(TESTVEC)
$(TESTVEC):
	python3 $(GEN_SCRIPT)

# -- run tests -------------------------------------------------------------
test-pc:  $(PC_TEST_BIN) testvectors
	@./$(PC_TEST_BIN)
test-mac: $(MAC_TEST_BIN)
	@./$(MAC_TEST_BIN)
test:     test-pc test-mac

test-py: testvectors
	@python3 test_precision_controller_ref.py
	@python3 test_mac_array_ref.py

integration:
	@python3 integration_example.py

test-all: test test-py
	@echo
	@echo "All C++ and Python reference tests passed (precision controller + MAC array)."

clean:
	rm -f $(PC_TEST_BIN) $(MAC_TEST_BIN) \
	      $(PC_SHARED) $(MAC_SHARED) \
	      $(PC_STATIC) $(MAC_STATIC) *.o
