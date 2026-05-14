CXX      ?= g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fPIC
PYTHON   ?= python3

SRC      := kv_cache_engine_ref.cpp
HDR      := kv_cache_engine_ref.hpp
TEST_SRC := test_kv_cache_engine_ref.cpp
TEST_BIN := test_kv
LIB_SO   := libkv_cache_engine_ref.so
LIB_A    := libkv_cache_engine_ref.a

TV_DIR   := ../../rtl/tb/testvectors

.PHONY: all test test-py test-all shared static clean testvectors

all: $(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SRC) $(SRC) -lm

test: $(TEST_BIN) testvectors
	./$(TEST_BIN)

test-py: testvectors
	$(PYTHON) test_kv_cache_engine_ref.py

test-all: test test-py

shared: $(LIB_SO)

$(LIB_SO): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(SRC) -lm

static: $(LIB_A)

$(LIB_A): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -c -o kv_cache_engine_ref.o $(SRC)
	ar rcs $@ kv_cache_engine_ref.o

testvectors:
	@if [ ! -f $(TV_DIR)/input_vectors.hex ]; then \
		echo "Generating test vectors..."; \
		$(PYTHON) ../../analysis/gen_kv_testvectors.py; \
	fi

clean:
	rm -f $(TEST_BIN) $(LIB_SO) $(LIB_A) *.o
