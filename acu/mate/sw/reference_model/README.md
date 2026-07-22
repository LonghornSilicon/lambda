# Reference Models — API Reference

Bit-accurate reference implementations of the LonghornSilicon ACU
blocks, in **three** languages each, all verified against the same
test vectors.

| Block | C++ class | extern "C" API | Python class | Tests pass? |
|---|---|---|---|---|
| **Precision Controller** | `lhsi::PrecisionController` | `lhsi_pc_*` | `PrecisionController` | ✅ 143/143 vs RTL TB |
| **MAC Array**            | `lhsi::mac::MacArray`        | `lhsi_mac_*` | `MacArray` | ✅ vs numpy reference, all shapes/edges |

Top-level orientation for the compiler team: see [`../README.md`](../README.md).
Design rationale for the MAC array: see [`MAC_ARRAY_DESIGN.md`](MAC_ARRAY_DESIGN.md).

## Building and testing

```sh
make test-all      # build both C++ tests, run them, run all Python tests
make test          # only the C++ tests
make test-py       # only the Python tests
make test-pc       # only the precision controller C++ test
make test-mac      # only the MAC array C++ test
make integration   # end-to-end attention-tile example (printed report)
make shared        # libprecision_controller_ref.so + libmac_array_ref.so
make static        # libprecision_controller_ref.a + libmac_array_ref.a
make clean
```

Requires Python 3.10+, NumPy, and a C++17 compiler (gcc-9+ or
clang-10+). The C++ tests pick up the RTL test vectors from
`rtl/tb/testvectors/*.hex`; the Makefile regenerates them from
`analysis/gen_rtl_testvectors.py` on first build if absent.

---

## Block 1 — Precision Controller

### C++ class

```cpp
#include "precision_controller_ref.hpp"

lhsi::PrecisionController pc;
std::vector<int32_t> scores(/* 4096 INT8 values */);
bool decision = pc.process_tile(scores);   // true = FP16, false = INT8
```

### Plain C (extern "C" API)

```c
#include "precision_controller_ref.hpp"

lhsi_pc_handle_t* h = lhsi_pc_create();
int32_t scores[4096] = { /* ... */ };
int decision = lhsi_pc_process_tile(h, scores, 4096);
lhsi_pc_destroy(h);
```

Stateless one-shot variant (no handle needed):

```c
int decision = lhsi_pc_decide(scores, 4096);
```

### Python

```python
from precision_controller_ref import PrecisionController

pc = PrecisionController()
decision = pc.process_tile(scores)
# Or stateless:
decision = PrecisionController.decide(scores)
```

### Streaming API (mirrors the chip's AXI-Stream interface)

For driver code or cycle-accurate simulators that want to issue one
score per clock:

```cpp
pc.reset();
for (size_t i = 0; i < scores.size(); ++i) {
    pc.tick(/*s_valid=*/true, scores[i], /*s_last=*/(i == scores.size() - 1));
    if (pc.d_valid()) {
        bool decision = pc.d_fp16();
    }
}
```

(Same shape in Python — `pc.tick(s_valid=True, s_data=x, s_last=False)`.)

---

## Block 2 — MAC Array

### C++ class

```cpp
#include "mac_array_ref.hpp"

lhsi::mac::MacArray mac;

// INT8 path
std::vector<int8_t> A(M*K), B(K*N);
std::vector<int32_t> C(M*N);
mac.matmul_int8(A.data(), B.data(), C.data(), M, K, N);

// FP16 path (float storage; fp16 rounding on output)
std::vector<float> Af(M*K), Bf(K*N), Cf(M*N);
mac.matmul_fp16(Af.data(), Bf.data(), Cf.data(), M, K, N);

// Cost estimate for the compiler's scheduler
auto cost = mac.estimate(M, K, N, lhsi::mac::MacArray::DType::Int8);
// cost.cycles, cost.energy_pj
```

### Plain C (extern "C" API)

```c
#include "mac_array_ref.hpp"

int8_t A[M*K], B[K*N];
int32_t C[M*N];
lhsi_mac_matmul_int8(A, B, C, M, K, N);

float Af[M*K], Bf[K*N], Cf[M*N];
lhsi_mac_matmul_fp16(Af, Bf, Cf, M, K, N);

lhsi_mac_cost_t cost;
lhsi_mac_estimate(M, K, N, LHSI_MAC_DTYPE_INT8, &cost);
```

### Python

```python
import numpy as np
from mac_array_ref import MacArray

mac = MacArray()

# INT8
A = np.random.randint(-128, 128, (M, K), dtype=np.int8)
B = np.random.randint(-128, 128, (K, N), dtype=np.int8)
C = mac.matmul_int8(A, B)        # numpy int32 array

# FP16
A = np.random.uniform(-1, 1, (M, K)).astype(np.float16).astype(np.float32)
B = np.random.uniform(-1, 1, (K, N)).astype(np.float16).astype(np.float32)
C = mac.matmul_fp16(A, B)        # float32 storage, fp16-rounded output

# Cost
cost = mac.estimate(M, K, N, dtype="int8")
print(cost.cycles, cost.energy_pj)
```

### FP16 utility helpers

These are also exposed for boundaries where you need explicit
fp16 rounding without going through a matmul:

```cpp
float    rounded = lhsi::mac::round_to_fp16(x);
uint16_t bits    = lhsi::mac::float_to_fp16_bits(x);
float    back    = lhsi::mac::fp16_bits_to_float(bits);
```

```python
from mac_array_ref import round_to_fp16, float_to_fp16_bits, fp16_bits_to_float
```

---

## Numerical semantics (frozen)

### Precision Controller

Pure integer arithmetic in unsigned fixed-width modular form:

- Input `s_data`: SCORE_WIDTH-bit signed two's complement
- `max_acc`: SCORE_WIDTH bits (unsigned abs)
- `sum_acc`: SCORE_WIDTH + log₂(N) bits (unsigned)
- LHS = max << log₂(N) (free shift, wire routing)
- RHS = (sum << 3) + (sum << 1) (= sum × 10)
- Decision: LHS > RHS → FP16

If the model disagrees with the chip on any input, the model is
wrong. Open an issue with the failing tile.

### MAC Array

INT8 path: bit-exact integer matmul, int32 accumulator, no
saturation. The compiler is responsible for any requantization
after the matmul.

FP16 path: float32 internally, IEEE-754 binary16 round-to-nearest-
even on each output element. See
[`MAC_ARRAY_DESIGN.md`](MAC_ARRAY_DESIGN.md) for the v0.1 simplifications
and what may change in v0.2.

---

## End-to-end example

The cleanest pointer for a compiler engineer: read and run
[`integration_example.py`](integration_example.py). It composes
both blocks the way a compiler backend should — process a
QK^T tile, push to the precision controller, dispatch to either
the INT8 or FP16 MAC path, accumulate.

```sh
make integration
```

Output (truncated):

```
Routed:  INT8 = 82, FP16 = 18
Truth :  INT8 = 82, FP16 = 18

Per-tile cost:
  INT8:   1040 cycles, 131.1 nJ
  FP16:   4112 cycles, 655.4 nJ

Mixed-precision policy:  ~159k cycles, ~22 µJ
All-FP16 baseline:       ~411k cycles, ~66 µJ
Speedup:        2.58x
Energy saving:  65.6%
```

The routing matches ground truth perfectly (82 / 18). The cost
model shows the win from mixed-precision dispatch.

---

## Three compiler-use scenarios

[`example_compiler_use.py`](example_compiler_use.py) shows three
levels of sophistication for a compiler backend using just the
precision controller — naive per-tile dispatch, batched index
lists, and offline calibration. The integration example above is
the next level up, composing both blocks.

---

## Verification status

| Test | Block | Status |
|---|---|---|
| 143/143 RTL replay tiles | Precision Controller | ✅ bit-exact in Python and C++ |
| Stateful streaming = batched | Precision Controller | ✅ |
| extern "C" = C++ class | Precision Controller | ✅ |
| Canonical edges (spike=10/11) | Precision Controller | ✅ |
| Accumulator reset | Precision Controller | ✅ |
| INT8 vs naive int64 reference | MAC Array | ✅ exact match |
| FP16 vs naive double reference | MAC Array | ✅ within 5e-3 rel. tol. |
| Edge cases (zero, identity, signed) | MAC Array | ✅ |
| extern "C" = C++ class | MAC Array | ✅ |
| fp16 round-trip helpers | MAC Array | ✅ |
| Cost estimate sanity | MAC Array | ✅ |
| End-to-end integration | both | ✅ routes match ground truth |

---

## Co-design context

This directory is the deliverable for the 2026-05-13 compiler-team
meeting action item: *"Develop more concrete high-level C and C++
implementations of key chip blocks in the coming weeks to support
close co-design work with the compiler team."*

Next blocks for this same C/C++ template:

- `kv_cache_engine_ref.{py,hpp,cpp}` — compression / decompression
- `token_importance_ref.{py,hpp,cpp}` — per-token accumulator
- `memory_hierarchy_ref.{py,hpp,cpp}` — L1/L2/DRAM routing

Each follows the pattern this directory establishes: a single Python
file as the executable spec, a matching C++ port for native codegen,
the `extern "C"` shim, a Makefile target, a bit-accurate test suite
gated in CI.
