# LonghornSilicon — Software for Compiler Co-Design

This directory holds the **bit-accurate reference model** of the
KV cache engine block, written for compiler-team integration.
Their compiler targets this model; we keep it bit-aligned with
the RTL as the block evolves.

> **You are a compiler engineer reading this for the first time:**
> jump to [`reference_model/README.md`](reference_model/README.md)
> for the quick-start. This top-level page is the orientation map.

## Layout

```
sw/
├── README.md                     ← you are here
└── reference_model/
    ├── README.md                 ← API reference + build instructions
    ├── Makefile                  ← build everything, run all tests
    │
    ├── kv_cache_engine_ref.py    ← Python golden reference
    ├── kv_cache_engine_ref.hpp   ← C++ header
    ├── kv_cache_engine_ref.cpp   ← C++ implementation
    │
    ├── test_kv_cache_engine_ref.py
    └── test_kv_cache_engine_ref.cpp
```

## Block status

| Block | Status | C++ class | C API | Python |
|---|---|---|---|---|
| KV Cache Engine | Bit-exact vs RTL | `lhsi::KVCacheEngine` | `lhsi_kv_*` | `KVCacheEngine` |

The model follows the same template as block 1
([adaptive-precision-attention](https://github.com/LonghornSilicon/adaptive-precision-attention)):
a header + cpp pair, a C ABI shim, a Python parity model, a test suite
that gates on parity with the RTL TB.

## Build everything at once

```sh
cd reference_model
make test-all     # builds C++ tests, runs them, runs Python tests
make shared       # libkv_cache_engine_ref.so
make static       # libkv_cache_engine_ref.a
make clean
```

Requires Python 3.10+, NumPy, and a C++17 compiler. The C++ tests
verify against the RTL test vectors (`rtl/tb/testvectors/*.hex`)
which the Makefile regenerates from
`analysis/gen_kv_testvectors.py` if they aren't on disk.

## What the compiler team should look at first

1. **Quick orientation** — this page (top to bottom, you're almost done)
2. **Reference-model API** — [`reference_model/README.md`](reference_model/README.md)
3. **ISA spec** — [`../docs/isa/kv_cache_engine_isa.pdf`](../docs/isa/kv_cache_engine_isa.pdf)
   especially §4 (logical operations) and §4.1 (worked lowering example)
4. **Formal API reference** — [`../docs/reference_model_api.pdf`](../docs/reference_model_api.pdf)
5. **Software overview PDF** — [`../docs/sw_overview.pdf`](../docs/sw_overview.pdf)

## Integration plan (recap from the ISA spec)

| Phase | Compiler targets | Status |
|---|---|---|
| 0  | Python + C++ reference models   | This directory |
| 1  | AXI on a ZCU102/104 FPGA        | When the board arrives |
| 2  | Multi-block FPGA project        | After blocks 3/4 land |
| 3  | TSMC 16FFC silicon              | Post-tape-out (2027+) |

The contract (ISA spec) is stable across all four phases. Only the
runtime implementation of the operations changes — Python function
call, AXI register write, or PCIe MMIO.

## Interaction with block 1 (Precision Controller)

The KV cache engine and precision controller work together during
attention inference:

1. KV cache engine decompresses K/V from SRAM
2. Attention unit computes Q·K^T scores
3. Precision controller decides INT8 vs FP16 per tile
4. MAC array runs the matmul at the chosen precision

A compiler backend targeting both blocks emits interleaved
`KV_DECOMPRESS_*` and `PC_PUSH_TILE` operations. The reference
models for both blocks can be composed in the same process.

## Open questions for the compiler team

1. **Compilation granularity**: does your compiler emit individual
   compress/decompress ops, or a fused `kv_attention(Q, K_cache, V_cache)` op?
2. **Batch interface**: should the reference model support batched
   compress/decompress for throughput?
3. **Async model**: should compress operations be non-blocking with
   a separate completion query, matching the hardware pipeline?
4. **Cost model precision**: are placeholder cycle counts enough for
   your scheduler, or do you need post-synthesis numbers?
