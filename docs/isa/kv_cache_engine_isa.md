# KV Cache Engine — Interface Specification

**Status**: pre-tape-out stub for compiler integration. Stable for the
KV cache engine block only; will be unified with the rest of the
LonghornSilicon ISA when the Token Importance Unit and Memory Hierarchy
Controller blocks land.

**Version**: `kv-isa-0.1` — first public draft, 2026-05-14.

**Scope**: This document describes the externally-visible interface to
the `kv_cache_engine` block as it will appear on the chip, an FPGA
prototype (ZCU102/104), or the bit-accurate software model
(`sw/reference_model/kv_cache_engine_ref.py`). A compiler backend
targeting the KV cache engine writes against this interface; the
hardware implementation must conform to it.

---

## 1. Block overview

The KV cache engine manages on-chip storage of key and value tensors for
transformer attention, with lossy compression to reduce DRAM bandwidth.
It implements TurboQuant+ (`turbo4` configuration):

- **Key path**: PolarQuant (3-bit) + QJL (1-bit) = 4.25 bits per value (bpv)
- **Value path**: PolarQuant (3-bit) only ~ 3.0 bpv

The compression pipeline:
1. **Norm extraction**: L2 norm of the input vector, stored as metadata.
2. **Rotation**: Walsh-Hadamard Transform with random sign flips
   (deterministic from seed). Self-inverse, O(n log n) butterfly structure.
3. **Quantization**: 3-bit optimal Lloyd-Max scalar quantization on the
   rotated coordinates, using 8 centroids symmetric around zero.
4. **QJL projection** (K path only): Quantized Johnson-Lindenstrauss
   transform on the residual. Random +/-1 projection matrix from seed,
   producing 1-bit signs per dimension.
5. **Packing**: Bit-packing of norm, indices, residual norm, and signs
   into a compact SRAM word.

Decompression reverses the pipeline: unpack, centroid lookup, QJL
correction (K only), inverse WHT, norm rescaling.

**Compression ratio:** K = 3.56x, V = 4.92x, combined >= 3x.
**Footprint:** ~6400 FFs estimated (full pipeline); ~93 FFs current top-level.
**Bit-exact reference:** `sw/reference_model/kv_cache_engine_ref.py`
and `sw/reference_model/kv_cache_engine_ref.{hpp,cpp}`.

---

## 2. Address space and memory map

The block exposes a 256-byte AXI-Lite slave register window. All
registers are 32-bit, word-aligned. The base address is set at chip
integration time (e.g., `0x4000_1000` on the ZCU102 PYNQ overlay).

| Offset | Name | Access | Reset | Purpose |
|--------|------|--------|-------|---------|
| `0x00` | `CTRL` | RW | 0x0 | bit[0]: `soft_reset` (write-1 to pulse); bit[1]: `enable` |
| `0x04` | `STATUS` | R | 0x1 | bit[0]: `idle`; bit[3]: `sram_full` |
| `0x08` | `INFO_DIM` | R | (syn) | Synthesis-time `VECTOR_DIM` |
| `0x0C` | `INFO_PQ_BITS` | R | (syn) | PolarQuant quantization bits (3) |
| `0x10` | `INFO_QJL_BITS` | R | (syn) | QJL projection bits (1) |
| `0x14` | `INFO_SRAM_DEPTH` | R | (syn) | Number of SRAM entries |
| `0x18` | `INFO_CR_K` | R | (syn) | Key compression ratio, fixed-point 8.8 |
| `0x1C` | `INFO_CR_V` | R | (syn) | Value compression ratio, fixed-point 8.8 |
| `0x20` | `INFO_VERSION` | R | 0x00010000 | ISA version (major.minor.patch.build) |
| `0x24` | `OCCUPANCY` | R | 0x0 | Number of valid entries in SRAM |
| `0x28` | `WRITE_ADDR` | RW | 0x0 | Target SRAM address for next write |
| `0x2C` | `READ_ADDR` | RW | 0x0 | Target SRAM address for next read |
| `0x30` | `KV_SELECT` | RW | 0x0 | bit[0]: 0 = key path, 1 = value path |
| `0x34` | `IRQ_MASK` | RW | 0x0 | bits[3:0]: interrupt enable mask |
| `0x38` | `IRQ_STATUS` | R/W1C | 0x0 | bits[3:0]: interrupt pending; write-1-to-clear |

**Conventions:**
- `RW` = read/write; `R` = read-only; `W1C` = write-1-to-clear; `(syn)` =
  value is fixed at synthesis time.
- Writes to read-only registers are silently dropped.
- Reading reserved offsets returns `0xDEADBEEF`.

---

## 3. Streaming data interfaces

Two AXI-Stream interfaces carry the actual vector data. AXI-Lite is for
control only.

### 3.1 `s_axis_kv` — Vector Write (slave)

| Signal | Width | Direction | Purpose |
|--------|-------|-----------|---------|
| `tdata` | `COORD_WIDTH` | in | One signed two's-complement coordinate |
| `tlast` | 1 | in | Assert on the final coordinate of the vector |
| `tvalid` | 1 | in | Handshake: data is valid |
| `tready` | 1 | out | Handshake: block is ready to accept |
| `tuser` | 1 | in | 0 = key vector, 1 = value vector |

**Protocol**:
- A complete vector is exactly `VECTOR_DIM` beats. `tlast` asserts on
  the final beat.
- The `tuser` field selects the compression path: key vectors receive
  PolarQuant + QJL compression; value vectors receive PolarQuant only.
- The target SRAM address must be set via `WRITE_ADDR` before
  streaming begins.
- Backpressure: when the block is not ready (e.g., during compression
  or SRAM write), `tready` deasserts.

### 3.2 `m_axis_kv` — Vector Read (master)

| Signal | Width | Direction | Purpose |
|--------|-------|-----------|---------|
| `tdata` | `COORD_WIDTH` | out | One signed two's-complement decompressed coordinate |
| `tlast` | 1 | out | Asserted on the final coordinate |
| `tvalid` | 1 | out | Handshake: data is valid |
| `tready` | 1 | in | Handshake: downstream is ready to accept |

**Protocol**: one beat per coordinate, `VECTOR_DIM` beats per vector.
Set `READ_ADDR` and `KV_SELECT` before initiating a read. The
decompression path is selected by `KV_SELECT`.

### 3.3 Eviction interface

| Signal | Width | Direction | Purpose |
|--------|-------|-----------|---------|
| `evict_needed` | 1 | out | Asserted when SRAM is full |
| `evict_addr` | ceil(log2(SRAM_DEPTH)) | out | Address of entry to evict |

This interface connects to the Memory Hierarchy Controller (block 4).
When `evict_needed` asserts, the controller must either spill the entry
at `evict_addr` to DRAM or invalidate it before the next write.

---

## 4. Logical operations (compiler-facing)

Below are the abstract operations a compiler emits. Each maps to a
short sequence of register writes / stream beats; the exact mapping
is in §5 (C API stub).

| Op | Inputs | Outputs | Description |
|----|--------|---------|-------------|
| `KV_QUERY` | - | INFO struct | Read all `INFO_*` registers |
| `KV_RESET` | - | - | Soft reset: clear SRAM, reset FSM |
| `KV_ENABLE` | - | - | Set bit[1] of `CTRL` |
| `KV_COMPRESS_KEY` | addr, D coords | - | Compress and store a key vector at `addr` |
| `KV_COMPRESS_VALUE` | addr, D coords | - | Compress and store a value vector at `addr` |
| `KV_DECOMPRESS_KEY` | addr | D coords | Read and decompress a key vector from `addr` |
| `KV_DECOMPRESS_VALUE` | addr | D coords | Read and decompress a value vector from `addr` |
| `KV_STATUS` | - | STATUS bits | Read the `STATUS` register |
| `KV_OCCUPANCY` | - | count | Read the `OCCUPANCY` register |

### 4.1 Worked lowering example

Suppose the compiler's IR contains:
```
%out = transformer_block(%q, %k, %v, %kv_cache)
    : tensor<S x D x f16>, ..., kv_cache_handle -> tensor<S x D x f16>
```

**Stage 1: setup.** Query hardware configuration once per compilation:
```
info = KV_QUERY()       // -> dim=64, pq_bits=3, qjl_bits=1, sram_depth=16
KV_RESET()
KV_ENABLE()
```

**Stage 2: store phase.** For each new token, compress and store K/V:
```
for token_idx in new_tokens:
    addr = token_idx % info.sram_depth
    KV_COMPRESS_KEY(addr, k_vectors[token_idx])
    KV_COMPRESS_VALUE(addr, v_vectors[token_idx])
```

**Stage 3: attention phase.** Decompress cached K/V for attention:
```
for cached_idx in range(num_cached):
    k_hat = KV_DECOMPRESS_KEY(cached_idx)
    v_hat = KV_DECOMPRESS_VALUE(cached_idx)
    score = dot(query, k_hat) / sqrt(D)

    // Route through precision controller (block 1)
    decision = PC_PUSH_TILE(score)
    if decision == FP16:
        out += fp16_matmul(softmax(score), v_hat)
    else:
        out += int8_matmul(softmax(score), v_hat)
```

### 4.2 Compiler binding patterns

**MLIR dialect.** Define `lhsi.kv.*` ops mirroring the table above.
The lowering pass rewrites `linalg.attention` or vendor-equivalent ops.

**TVM Relax / TIR.** Register backend target `"lhsi-kv"`,
pattern-match KV cache operations in the attention sub-graph.

**ONNX.** Add `CustomOpDomain` with KV cache ops, graph transform
inserts compression/decompression around attention.

**Custom IR.** Replace KV cache codegen with `KV_*` operation sequences
targeting the C API.

---

## 5. C API stub

```c
/* lhsi_kv_cache_engine.h — LonghornSilicon KV Cache Engine driver API. */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef struct {
    uint32_t vector_dim;
    uint32_t pq_bits;
    uint32_t qjl_bits;
    uint32_t sram_depth;
    uint32_t cr_k_fixed;       /* compression ratio K, fixed-point 8.8 */
    uint32_t cr_v_fixed;       /* compression ratio V, fixed-point 8.8 */
    uint32_t version;
} lhsi_kv_info_t;

typedef struct lhsi_kv_handle lhsi_kv_handle_t;

/* Open / close */
int  lhsi_kv_open (const char *device, lhsi_kv_handle_t **out);
void lhsi_kv_close(lhsi_kv_handle_t *h);

/* Configuration queries */
int  lhsi_kv_query(lhsi_kv_handle_t *h, lhsi_kv_info_t *out);

/* Control */
int  lhsi_kv_reset (lhsi_kv_handle_t *h);
int  lhsi_kv_enable(lhsi_kv_handle_t *h, bool enable);

/* Compress and store a key vector at the given SRAM address. */
int  lhsi_kv_compress_key(lhsi_kv_handle_t *h, uint32_t addr,
                          const int16_t *coords, size_t dim);

/* Compress and store a value vector at the given SRAM address. */
int  lhsi_kv_compress_value(lhsi_kv_handle_t *h, uint32_t addr,
                            const int16_t *coords, size_t dim);

/* Read and decompress a key vector from the given SRAM address. */
int  lhsi_kv_decompress_key(lhsi_kv_handle_t *h, uint32_t addr,
                            int16_t *coords, size_t dim);

/* Read and decompress a value vector from the given SRAM address. */
int  lhsi_kv_decompress_value(lhsi_kv_handle_t *h, uint32_t addr,
                              int16_t *coords, size_t dim);

/* Diagnostics. */
typedef struct {
    bool     idle;
    bool     sram_full;
    uint32_t occupancy;
} lhsi_kv_status_t;
int  lhsi_kv_status(lhsi_kv_handle_t *h, lhsi_kv_status_t *out);
```

Return codes follow the standard Linux convention: `0` on success,
`-errno` on failure.

---

## 6. Compression algorithm: TurboQuant+

TurboQuant+ is a two-stage vector quantization scheme designed for KV
cache compression. The `turbo4` configuration provides 4.25 bpv for
keys and ~3.0 bpv for values.

### PolarQuant (both K and V)

1. **Norm extraction**: Compute ||x||_2 and normalize: x_hat = x / ||x||_2
2. **Rotation**: Apply random sign flips s_i from LCG seed, then WHT:
   y = WHT(x_hat * s). Randomizes coordinates toward i.i.d. Gaussian.
3. **Quantization**: 3-bit optimal Lloyd-Max scalar quantization using
   8 centroids: +/-0.2451*sigma, +/-0.7560*sigma, +/-1.3439*sigma,
   +/-2.1520*sigma where sigma = 1/sqrt(D).

### QJL correction (K path only)

After PolarQuant, the quantization residual r = y - y_hat is projected:
```
z_j = sign(sum_i R_ji * r_i)
```
where R is a random +/-1 matrix from seed + 0xDEADBEEF. The 1-bit signs
preserve inner products in expectation (Johnson-Lindenstrauss property).

### Bit layout

| Path | Fields | Bits (D=64) | Compression |
|------|--------|-------------|-------------|
| Key | norm(16) + indices(3*D) + res_norm(16) + signs(D) | 288 | 3.56x |
| Value | norm(16) + indices(3*D) | 208 | 4.92x |

---

## 7. Synthesis-time configuration

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `VECTOR_DIM` | 64 | 16, 32, 64, 128 | Must be power of two (WHT) |
| `PQ_BITS` | 3 | 2, 3, 4 | Lloyd-Max quantization bits |
| `QJL_BITS` | 1 | 1 | Fixed at 1 in kv-isa-0.1 |
| `SRAM_DEPTH` | 16 | 4-4096 | Behavioral reg for Sky130; real macros at 16FFC |
| `COORD_WIDTH` | 16 | 8, 12, 16 | Fixed-point coordinate width |
| `COORD_FRAC` | 12 | 4-14 | Fractional bits |
| `NORM_WIDTH` | 16 | 8, 12, 16 | Norm storage width |
| `ROTATION_SEED` | 42 | any 32-bit | Deterministic LCG seed |

---

## 8. Bit-accurate reference

The Python model at `sw/reference_model/kv_cache_engine_ref.py` is the
canonical bit-accurate reference. A compiler can verify codegen directly:

```python
from kv_cache_engine_ref import KVCacheEngine

engine = KVCacheEngine()
ck = engine.compress_key(vector)
reconstructed = engine.decompress_key(ck)
```

The model is verified bit-exact against the RTL testbench vectors.
Any divergence between the model and the chip is a bug in this spec.

---

## 9. Integration phases

| Phase | Timeline | Compiler targets | We provide |
|-------|----------|-----------------|------------|
| 0 | now | Python + C++ reference models | Models + ISA + tests |
| 1 | when FPGA arrives | AXI-Lite + AXI-Stream on Zynq | Vivado bitstream + driver |
| 2 | all 4 blocks done | Multi-block FPGA project | Integrated pipeline |
| 3 | post-tape-out (2027+) | PCIe-attached chip | TSMC 16FFC silicon + driver |

Throughout all phases, the interface described in this document is
the stable contract.

---

## 10. Interaction with other blocks

- **Block 1 (Precision Controller)**: Receives decompressed K/V vectors
  from this block for attention score computation. Decides INT8 vs FP16.
- **Block 3 (Token Importance Unit)**: Provides token importance scores
  that inform eviction decisions when SRAM is full.
- **Block 4 (Memory Hierarchy Controller)**: Receives `evict_needed` /
  `evict_addr` signals. Manages spilling to DRAM and fetching back.

---

## 11. Open questions

1. Should the compression path support variable-rate quantization
   (2/3/4-bit PolarQuant selected per-vector based on norm)?
2. Should the QJL projection dimension be configurable (D x D' with
   D' < D for reduced storage)?
3. Should the block support direct SRAM-to-SRAM copy for cache migration?
4. Are the current fixed-point widths sufficient for larger models
   (D = 128, 256)?
5. Should the eviction policy be configurable via registers (LRU,
   FIFO, importance-based)?

---

## 12. Change log

- `kv-isa-0.1` (2026-05-14): First public draft. Stable for the KV cache
  engine block only.
