# Precision Controller Reference Model

Bit-accurate reference implementations of the LonghornSilicon ACU
precision controller, in **three** languages, all verified against the
same 143 RTL replay tiles.

| Implementation | Files | Use case |
|---|---|---|
| **Python** | `precision_controller_ref.py`, `test_precision_controller_ref.py` | Compiler-side simulation, prototyping, FFI from Python tools |
| **C++** (class API) | `precision_controller_ref.hpp`, `precision_controller_ref.cpp` | Native C++ codegen backends (MLIR, TVM C++ runtime, custom compilers) |
| **C** (`extern "C"` shim) | Same headers, but only the `lhsi_pc_*` symbols | Plain-C runtimes, ABI-stable linking, FFI from any language |

All three pass:

- 143 / 143 RTL replay tiles bit-exact (`Tests: 143  Pass: 143  Fail: 0`)
- Canonical boundary cases (spike=10 → INT8, spike=11 → FP16)
- Stateful streaming vs. stateless `decide()` agreement
- C API and C++ class produce identical results
- Accumulator reset between consecutive tiles

## Building and testing

```sh
make test          # build the C++ test binary, run it. ~5 sec.
make test-py       # run the Python test suite.        ~1 sec.
make test-all      # both, with a final parity print.
make shared        # libprecision_controller_ref.so for FFI
make static        # libprecision_controller_ref.a for embedding
make clean         # remove build artifacts
```

The Makefile auto-generates the RTL test vectors on first run via
`analysis/gen_rtl_testvectors.py` if they aren't already on disk.

## C++ class usage

```cpp
#include "precision_controller_ref.hpp"
#include <vector>

int main() {
    lhsi::PrecisionController pc;
    std::vector<int32_t> scores(/* 4096 INT8 values */);
    bool decision = pc.process_tile(scores);   // true = FP16, false = INT8
}
```

## C API usage (for non-C++ runtimes)

```c
#include "precision_controller_ref.hpp"

int main(void) {
    lhsi_pc_handle_t* h = lhsi_pc_create();

    int32_t scores[4096] = { /* ... */ };
    int decision = lhsi_pc_process_tile(h, scores, 4096);

    lhsi_pc_destroy(h);
    return 0;
}
```

Or stateless (no handle needed):

```c
int decision = lhsi_pc_decide(scores, 4096);
```

## What "bit-accurate" means here

The reference uses fixed-width unsigned modular arithmetic that mirrors
the SystemVerilog RTL exactly:

- SCORE_WIDTH-bit two's-complement scores in on the input
- SUM_WIDTH = SCORE_WIDTH + log₂(N) bits for the running sum
- CMP_WIDTH = SUM_WIDTH + 4 bits for the final comparison
- THRESHOLD = 10, implemented as `(sum << 3) + (sum << 1)` (matches the
  shift-and-add hardware path)
- Accumulators reset on the same cycle as `s_last` (last-write-wins,
  matches the SV)

If you find a tile where this model disagrees with the chip, that's a
bug in the spec, not in the chip. Open an issue and reference the tile
index in `expected.hex`.

## Co-design with the compiler team

This directory is the deliverable for the action item from the
2026-05-13 meeting: *"Develop more concrete high-level C and C++
implementations of key chip blocks in the coming weeks to support
close co-design work with the compiler team."*

This is the **first block done** (the precision controller / ACU
decision gate). The other three planned chip blocks each get their
own subdirectory under `sw/` as they land:

- `sw/reference_model/mac_array_ref.{py,hpp,cpp}` (next priority)
- `sw/reference_model/token_importance_ref.{py,hpp,cpp}`
- `sw/reference_model/kv_cache_engine_ref.{py,hpp,cpp}`
- `sw/reference_model/memory_hierarchy_ref.{py,hpp,cpp}`

Each follows this same template: a single Python file as the
executable spec, a matching C++ port for native codegen, the
`extern "C"` shim, and a bit-accurate test suite gated in CI.

For the abstract operations the compiler emits against each model and
how those map to chip ISA, see [`docs/isa/precision_controller_isa.pdf`](../../docs/isa/precision_controller_isa.pdf).
