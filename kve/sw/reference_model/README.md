# Reference Model — API Reference

> **Codec revamp (TurboQuant+ → ChannelQuant).** The block's codec is being
> replaced (see [`../../README.md`](../../README.md)). The **ChannelQuant** C++
> reference is `channelquant_ref.{hpp,cpp}` — a 1:1 port of the RTL cores, the C++
> leg of 3-way Python↔C++↔SV parity. Run it with **`make test-cq`** (all 9 golden
> vectors bit-exact). Everything else on this page documents the **legacy
> TurboQuant+** model (`kv_cache_engine_ref.*`), retained until the ChannelQuant
> API surface is finalized.

Bit-accurate reference implementation of the LonghornSilicon KV cache
engine, in **three** languages, all verified against the same test vectors.

| Block | C++ class | extern "C" API | Python class | Tests pass? |
|---|---|---|---|---|
| **KV Cache Engine** | `lhsi::KVCacheEngine` | `lhsi_kv_*` | `KVCacheEngine` | 120 Python + 64 C++ |

Top-level orientation for the compiler team: see [`../README.md`](../README.md).
ISA specification: see [`../../docs/isa/kv_cache_engine_isa.pdf`](../../docs/isa/kv_cache_engine_isa.pdf).
Formal API reference PDF: see [`../../docs/reference_model_api.pdf`](../../docs/reference_model_api.pdf).

## Building and testing

```sh
make test-all      # build C++ tests, run them, run all Python tests
make test          # only the C++ tests
make test-py       # only the Python tests
make shared        # libkv_cache_engine_ref.so
make static        # libkv_cache_engine_ref.a
make clean
```

Requires Python 3.10+, NumPy, and a C++17 compiler (gcc-9+ or
clang-10+). The C++ tests pick up the RTL test vectors from
`rtl/tb/testvectors/*.hex`; the Makefile regenerates them from
`analysis/gen_kv_testvectors.py` on first build if absent.

---

## C++ class API

```cpp
#include "kv_cache_engine_ref.hpp"

lhsi::KVCacheEngine engine;   // default config
std::vector<int16_t> key_vec(64, /* ... */);

// Compress a key vector
lhsi::CompressedKey ck = engine.compress_key(key_vec);

// Decompress back
std::vector<int16_t> reconstructed = engine.decompress_key(ck);

// Compress a value vector (no QJL — MSE-optimized path)
lhsi::CompressedValue cv = engine.compress_value(val_vec);
std::vector<int16_t> val_recon = engine.decompress_value(cv);
```

### Sub-operations (mirror the RTL pipeline stages)

```cpp
uint32_t norm = engine.compute_norm(vec, dim);
engine.normalize(vec, norm, normalized, dim);
engine.rotate(normalized, rotated, dim);
engine.quantize_pq(rotated, indices, dim);
engine.dequantize_pq(indices, dequantized, dim);
engine.qjl_compress(residual, signs, &res_norm, dim);
engine.qjl_decompress(signs, res_norm, correction, dim);
engine.inverse_rotate(corrected, output, dim);
```

### Store/load with SRAM

```cpp
engine.store_kv(/*addr=*/0, key_vec.data(), val_vec.data(), 64);
engine.load_k(0, key_out.data(), 64);
engine.load_v(0, val_out.data(), 64);
```

---

## Plain C (extern "C" API)

```c
#include "kv_cache_engine_ref.hpp"

lhsi_kv_handle_t* h = lhsi_kv_create_default();

// Compress key
uint32_t norm, res_norm;
uint8_t indices[64], signs[64];
lhsi_kv_compress_key(h, vec, 64, &norm, indices, &res_norm, signs);

// Decompress key
int16_t output[64];
lhsi_kv_decompress_key(h, norm, indices, res_norm, signs, 64, output);

// Compress/decompress value (no QJL)
lhsi_kv_compress_value(h, vec, 64, &norm, indices);
lhsi_kv_decompress_value(h, norm, indices, 64, output);

// Store/load with SRAM
lhsi_kv_store(h, 0, key_vec, val_vec, 64);
lhsi_kv_load_k(h, 0, key_out, 64);
lhsi_kv_load_v(h, 0, val_out, 64);

lhsi_kv_destroy(h);
```

---

## Python

```python
from kv_cache_engine_ref import KVCacheEngine, KVCacheEngineInfo

engine = KVCacheEngine()

# Compress key
ck = engine.compress_key(key_vector)
reconstructed = engine.decompress_key(ck)

# Compress value
cv = engine.compress_value(val_vector)
val_recon = engine.decompress_value(cv)

# Store/load with SRAM
engine.store_kv(addr=0, key=key_vector, value=val_vector)
k_out = engine.load_k(addr=0)
v_out = engine.load_v(addr=0)

# Static method (no state needed)
ck = KVCacheEngine.decide_compress_key(key_vector)
```

---

## Numerical semantics (frozen)

All arithmetic is **fixed-point integer**, matching the RTL exactly:

- **Coordinates**: `COORD_WIDTH`-bit signed two's complement,
  `COORD_FRAC` fractional bits (default: Q4.12).
- **Norms**: `NORM_WIDTH`-bit unsigned, `NORM_FRAC` fractional bits
  (default: Q8.8).
- **WHT butterfly**: additions widen by 1 bit per stage (log2(D)
  stages), then truncate back to coordinate width.
- **Quantization**: nearest-centroid with 3 comparators per coordinate.
  Centroids are Lloyd-Max optimal for N(0, 1/sqrt(D)).
- **QJL**: random +/-1 matrix from deterministic LCG seed. Inner product
  → sign extraction. Decompression scales by sqrt(pi/2) (fixed-point).
- **Integer sqrt**: non-restoring algorithm, bit-serial from MSB.

If the model disagrees with the chip on any input, the model is
wrong. Open an issue with the failing vector.

---

## Verification status

| Test | Aspect | Status |
|---|---|---|
| 120 Python unit tests | Full round-trip, all paths | Pass |
| 64 C++ unit tests | Full round-trip, all paths | Pass |
| Compress/decompress round-trip | K and V paths | Bit-exact |
| K/V asymmetry verification | QJL on K only | Pass |
| Streaming vs batch agreement | tick() vs compress_*() | Pass |
| extern "C" = C++ class | C API parity | Pass |
| RTL replay vectors | Hex-file bit-exact match | Pass |
| Compression ratio >= 3x | Canonical trace | Pass |
