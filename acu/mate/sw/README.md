# LonghornSilicon — Software for Compiler Co-Design

This directory holds **bit-accurate reference models** of the
LonghornSilicon chip blocks, written for compiler-team integration.
Their compiler targets these models; we keep them bit-aligned with
the RTL as each block lands.

> 📌 **You are a compiler engineer reading this for the first time:**
> jump to [`reference_model/README.md`](reference_model/README.md)
> for the quick-start. This top-level page is the orientation map.

## Layout

```
sw/
├── README.md                     ← you are here
└── reference_model/
    ├── README.md                 ← API reference + build instructions
    ├── MAC_ARRAY_DESIGN.md       ← MAC array design decisions
    ├── Makefile                  ← build everything, run all tests
    │
    ├── precision_controller_ref.{hpp,cpp,py}
    ├── test_precision_controller_ref.{cpp,py}
    │
    ├── mac_array_ref.{hpp,cpp,py}
    ├── test_mac_array_ref.{cpp,py}
    │
    ├── integration_example.py    ← worked example: full attention tile flow
    └── example_compiler_use.py   ← three usage levels for the precision controller
```

## Which blocks are modeled today

| Block | Status | C++ class | C API | Python |
|---|---|---|---|---|
| ACU — Precision Controller | ✅ bit-exact vs RTL (143/143) | `lhsi::PrecisionController` | `lhsi_pc_*` | `PrecisionController` |
| ACU — MAC Array            | ✅ vs numpy reference          | `lhsi::mac::MacArray`       | `lhsi_mac_*` | `MacArray` |
| KV Cache Engine            | ⏳ not started                | —                            | —             | —          |
| Token Importance Unit      | ⏳ not started                | —                            | —             | —          |
| Memory Hierarchy Controller| ⏳ not started                | —                            | —             | —          |

Each new block follows the same template: a header + cpp pair, a C
ABI shim, a Python parity model, a test suite that gates on parity
with the RTL TB (when RTL exists) or a naive reference (when only
the spec exists). See
[`docs/new_block_blueprint.md`](../docs/new_block_blueprint.md) for
the per-block scaffolding pattern.

## Build everything at once

```sh
cd reference_model
make test-all     # builds both C++ tests, runs them, runs Python tests
make integration  # runs the end-to-end attention-tile worked example
make shared       # libprecision_controller_ref.so + libmac_array_ref.so
make static       # the .a variants
make clean        # remove build artifacts
```

Requires Python 3.10+, NumPy, and a C++17 compiler. The C++ tests
verify against the RTL test vectors (`rtl/tb/testvectors/*.hex`)
which the Makefile regenerates from
`analysis/gen_rtl_testvectors.py` if they aren't on disk.

## What the compiler team should look at first

1. **Quick orientation** — this page (top to bottom, you're almost done)
2. **Reference-model API** — [`reference_model/README.md`](reference_model/README.md)
3. **What a compiler emits, end-to-end** — [`reference_model/integration_example.py`](reference_model/integration_example.py)
4. **ISA spec** — [`../docs/isa/precision_controller_isa.pdf`](../docs/isa/precision_controller_isa.pdf)
   especially §4.1 (worked lowering example) and §4.2 (compiler
   binding patterns: MLIR / TVM / ONNX / custom IR)
5. **Design choices for the MAC array** — [`reference_model/MAC_ARRAY_DESIGN.md`](reference_model/MAC_ARRAY_DESIGN.md)
   with the open questions for v0.2

The integration example is the most concrete pointer — run it and
read it side-by-side. The shape of that script is the shape of code
a compiler backend should produce.

## Integration plan (recap from the ISA spec)

| Phase | Compiler targets | Status |
|---|---|---|
| 0  | Python + C++ reference models   | ✅ this directory (block 1 + 2 done) |
| 1  | AXI on a ZCU102/104 FPGA        | when the board arrives |
| 2  | Multi-block FPGA project        | after blocks 3/4/5 land |
| 3  | TSMC 16FFC silicon              | post-tape-out (2027+) |

The contract (ISA spec) is stable across all four phases. Only the
runtime implementation of the operations changes — Python function
call, AXI register write, or PCIe MMIO.

## Open questions for the compiler team

These are in flight and worth discussing at the next sync. The
answers shape the v0.2 reference models:

1. **Compilation granularity**: does your compiler emit individual
   matmuls + a precision-gate op, or does it expect a fused
   `flash_attention(Q, K, V)` op? Affects whether we add a fused-
   attention reference operation.
2. **FP16 bit-exactness**: do you require bit-exact parity with our
   hardware's eventual FP16 rounding, or is numerically-close
   sufficient? See `MAC_ARRAY_DESIGN.md` §"FP16 path (open)".
3. **Async issue + token model**: do you want non-blocking matmul
   issue today (we'd add it to v0.2 now), or wait until the RTL
   exists?
4. **Cost model precision**: are placeholder cycle counts enough for
   your scheduler, or do you need post-synthesis numbers?
5. **THRESHOLD as a runtime register**: see ISA §9, open question 1.

We will track answers in the design docs and tag each as **frozen**
or **open** so the compiler team knows what they can build against.
