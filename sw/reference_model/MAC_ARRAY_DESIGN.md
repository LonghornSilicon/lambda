# MAC Array — Design Decisions for the v0.1 Reference Model

These are the choices baked into the v0.1 reference. Marked **(open)**
are decisions that the eventual hardware may revise; the reference will
track changes. Marked **(frozen)** are choices we are not planning to
revisit.

## Operations exposed

| Operation | Inputs | Output | Use |
|---|---|---|---|
| `matmul_int8` | int8 [M×K], int8 [K×N] | int32 [M×N] | INT8 attention/MLP path |
| `matmul_fp16` | fp16 [M×K], fp16 [K×N] | fp16 [M×N] | FP16 attention/MLP path |
| `query`       | — | `MacArrayInfo` | Compiler queries supported precisions, accumulator widths, PE grid |
| `estimate`    | M, K, N, dtype | cycles + energy_pJ | Cost model for compiler-side scheduling |

The compiler emits a `matmul_int8` or `matmul_fp16` op for every
attention sub-tile, dispatched on the precision controller's decision
(see the worked lowering example in the ISA spec).

## Data type semantics

### INT8 path (frozen)

- Storage: signed two's-complement 8-bit (`int8_t`, range −128…+127)
- Multiplication: signed 8-bit × signed 8-bit → exact 16-bit product
- Accumulation: 32-bit signed accumulator (`int32_t`)
- For K up to 65,536 inputs the accumulator cannot overflow
  (worst-case `K · 127 · 127 ≈ 1.06 G`, fits in 31 bits)
- No saturation, no requantization at the accumulator boundary —
  the compiler is responsible for any downcast/requantize step after
  the matmul

### FP16 path (open — v0.1 simplification)

- Storage: IEEE-754 binary16 (`half`, 1 sign + 5 exponent + 10 mantissa)
- Multiplication: half × half → round-to-nearest-even → half
- Accumulation: fp32 accumulator (matches typical attention HW)
- Final output: rounded back to fp16 (round-to-nearest-even)
- **v0.1 caveat**: the reference model uses `float` internally and
  rounds back to fp16 between operations. The eventual hardware may
  use a denser accumulator format (e.g., bfloat16) or skip the
  intermediate rounding. The compiler should *not* depend on bit-exact
  parity with v0.1's FP16 path until v0.2 freezes the choice.

### Why both paths (frozen)

The precision controller decides INT8 or FP16 per tile. The MAC
array therefore needs to dispatch to *either* path on a tile-by-tile
basis. A unified mixed-precision path is out of scope for v0.1 because
the cost / energy savings of INT8 only materialize when the entire
matmul stays in INT8.

## Shape support (frozen)

- Arbitrary M × K × N (no power-of-two requirement)
- Recommended tile shape: 64 × 64 × 64 (matches precision controller's
  default tile)
- A typical FlashAttention-style block computes:
  - `Q · K^T` → 64×64×64 = 64-element K · 64-element Q × 64 keys per tile
  - `score · V`→ 64×64×64
- The reference is shape-agnostic — the compiler picks the shape, the
  reference produces the correct output.

## Concurrency / streaming (open)

The v0.1 reference is **synchronous**: `matmul_int8(...)` blocks until
the result is in `C`. The eventual hardware will likely expose:

1. An asynchronous "issue" call that returns a token
2. A "wait" call on the token
3. Double-buffered input/output regions so the compiler can pipeline

For v0.1 the compiler can model these by wrapping the synchronous calls
in its own scheduler. We will revisit when the MAC array RTL lands.

## Cost model (v0.1 placeholders)

`MacArrayInfo` exposes:

- `pe_grid_m × pe_grid_n` — 8 × 8 (64 PEs), matching the canonical chip
  config (`architecture/arch.yml` `matrix_engine`, `STATUS.md`): 128 GOPS
  peak = 64 PE × 2 ops × 1 GHz
- `int8_throughput`: 64 INT8 MACs per cycle (one per PE)
- `fp16_throughput`: 16 FP16 MACs per cycle (FP16 PE is ~4× INT8 area,
  so ¼ the count)

`estimate(M, K, N, dtype)` returns:

- `cycles ≈ ceil((M·N·K) / throughput) + pipeline_depth` (16 cycles)
- `energy_pj ≈ M·N·K · per_op_energy_pj` (per-op = 0.5 pJ INT8,
  2.5 pJ FP16; rough numbers from published 16FFC accelerator data)

These are **placeholders** for the compiler's scheduler. The numbers
will tighten once the MAC array RTL is written and synthesized; today
they are a reasonable order-of-magnitude estimate. The compiler should
use them for relative comparison (INT8 vs FP16, which shape is faster)
rather than absolute timing.

> **Synthesized delta (P·V tile, Sky130A N=4 proxy).** Both the INT8 and
> FP16 P·V tiles now have synthesizable RTL taken to GDSII
> (`rtl/mate_pv.sv`, `rtl/mate_pv_fp16.sv`; see `docs/mate_pv_fp16_rtl.md`).
> Measured: FP16 is **≈ 2.8× the std-cell area, ≈ 3.2× the cell count, and
> ≈ 11.5× the energy per MAC** of INT8, and clocks **≈ 6× slower** (the
> single-cycle fp32 accumulate is the limiter) — the "~4× FP16 PE area" /
> "5× energy" placeholders above are the right order of magnitude, a bit
> low on area and high-ish on relative throughput for the P·V reduction.
> FP16 flip-flop count is ≈ equal to INT8. These replace the earlier
> "FP16 P·V area/power delta — TBD pending re-synthesis" caveat.

## What's NOT in v0.1

- Sparsity acceleration (skip-zero PEs)
- Mixed-precision MAC (e.g., INT4 × INT8)
- Batched matmul as a single op (compiler can loop)
- Hardware fault injection / ECC
- Power-state management

These are all out of scope for the reference model until the chip
architecture commits to them.

## How the compiler binds this

Same template as the precision controller. Three layers:

1. **C++ class** (`lhsi::mac::MacArray`) for MLIR / TVM C++ backends
2. **`extern "C"` API** (`lhsi_mac_*`) for plain-C runtimes and FFI
3. **Python reference** (`mac_array_ref.py`) for compiler-side
   prototyping and unit tests

All three pass identical test vectors. If the C++ and Python disagree,
that's a bug — open an issue.

## Open questions for the compiler team

Bring these to the next sync; the answers shape v0.2:

1. Does your compiler emit individual matmuls, or does it expect a
   fused `attention(Q, K, V)` op? Affects whether we should add a
   fused-attention reference operation.
2. Do you need cycle-level cost estimates, or is operation-count
   estimation sufficient for your scheduler?
3. For FP16: do you expect bit-exact parity with our hardware's
   FP16 rounding, or just numerically-close output? Affects v0.2's
   rounding mode decision.
4. Async issue + token model — do you want this in v0.1 (we'd add it
   now) or v0.3 (after RTL exists)?
