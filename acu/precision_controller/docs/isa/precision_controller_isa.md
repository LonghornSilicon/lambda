# Precision Controller — Interface Specification (Stub)

**Status**: pre-tape-out stub for compiler integration. Stable for the
precision controller block (ACU) only; will be unified with the rest of
the LonghornSilicon ISA when KV Cache Engine, Token Importance Unit, and
Memory Hierarchy Controller blocks land.

**Version**: `pc-isa-0.1` — first public draft, 2026-05-13.

**Scope**: This document describes the externally-visible interface to
the `precision_controller` block as it will appear on the chip, an FPGA
prototype (ZCU102/104), or the bit-accurate software model
(`sw/reference_model/precision_controller_ref.py`). A compiler backend
targeting the precision controller writes against this interface; the
hardware implementation must conform to it.

---

## 1. Block overview

The precision controller is a streaming block that consumes one signed
attention score per cycle, asserts an end-of-tile signal on the last
score, and produces a single-bit precision decision (INT8 or FP16) one
cycle later. The decision is computed as:

```
   decision = (max(|s|) << log2(N))  >  ((sum(|s|) << 3) + (sum(|s|) << 1))
                  =  max(|s|) * N    >   sum(|s|) * 10
   decision = 1  →  this tile needs FP16
   decision = 0  →  this tile can use INT8
```

`N = BLOCK_M * BLOCK_N` is the synthesis-time tile size; THRESHOLD is
hard-coded at 10 in the synthesized netlist via the multiply-by-10
shift-and-add. Changing these requires a re-synthesis (see §6).

---

## 2. Address space and memory map

The block exposes a 256-byte AXI-Lite slave register window. All
registers are 32-bit, word-aligned. The base address is set at chip
integration time (e.g., `0x4000_0000` on the ZCU102 PYNQ overlay).

| Offset | Name                | Access | Reset | Purpose |
|--------|---------------------|--------|-------|---------|
| `0x00` | `CTRL`              | RW     | 0x0   | bit[0]: `soft_reset` (write-1 to pulse); bit[1]: `enable` |
| `0x04` | `STATUS`            | R      | 0x1   | bit[0]: `idle`; bit[1]: `decision_valid`; bit[2]: `tile_in_progress`; bit[3]: `fifo_full` |
| `0x08` | `INFO_BLOCK_M`      | R      | (syn) | Synthesis-time `BLOCK_M` |
| `0x0C` | `INFO_BLOCK_N`      | R      | (syn) | Synthesis-time `BLOCK_N` |
| `0x10` | `INFO_N`            | R      | (syn) | `BLOCK_M * BLOCK_N` |
| `0x14` | `INFO_SCORE_WIDTH`  | R      | (syn) | Synthesis-time `SCORE_WIDTH` (bits per score) |
| `0x18` | `INFO_THRESHOLD`    | R      | (syn) | Synthesis-time `THRESHOLD` (currently always 10) |
| `0x1C` | `INFO_VERSION`      | R      | 0x100 | ISA version: bits[15:8] = major, bits[7:0] = minor |
| `0x20` | `TILE_COUNT`        | R      | 0x0   | Number of complete tiles processed since the last reset |
| `0x24` | `LAST_DECISION`     | R      | 0x0   | bit[0]: most recent decision (1 = FP16) |
| `0x28` | `DECISION_FIFO`     | R      | -     | Pop one decision from the result FIFO; bit[0] = decision; bit[31] = `valid` |
| `0x2C` | `DECISION_FIFO_LVL` | R      | 0x0   | Number of decisions waiting in the FIFO |
| `0x30` | `IRQ_MASK`          | RW     | 0x0   | bit[0]: enable interrupt on every decision |
| `0x34` | `IRQ_STATUS`        | R/W1C  | 0x0   | bit[0]: decision-available IRQ pending; write 1 to clear |

**Conventions:**
- `RW` = read/write; `R` = read-only; `W1C` = write-1-to-clear; `(syn)` =
  value is fixed at synthesis time.
- Writes to read-only registers are silently dropped.
- Reading reserved offsets returns 0.

---

## 3. Streaming data interfaces

Two AXI-Stream interfaces carry the actual data. AXI-Lite is for control
only.

### 3.1 `s_axis_scores` — score input (slave)

| Signal     | Width        | Direction | Purpose |
|------------|--------------|-----------|---------|
| `tdata`    | `SCORE_WIDTH`| in        | One signed two's-complement attention score |
| `tlast`    | 1            | in        | Assert on the final score of each tile |
| `tvalid`   | 1            | in        | Handshake: data is valid |
| `tready`   | 1            | out       | Handshake: block is ready to accept |

**Protocol**:
- A complete tile is exactly `N` beats. The block does not validate this
  — feeding fewer than N beats and asserting `tlast` mid-tile produces
  an undefined decision; feeding more than N delays the next decision by
  the overflow count.
- `tlast` is asserted on the N-th beat. The decision is computed
  combinationally on that cycle's `tdata` (so the running max/sum sees
  the last score), and the decision is latched on the following rising
  edge.
- Backpressure: when the block is not ready (e.g., decision FIFO full),
  `tready` deasserts. Upstream must hold `tdata`/`tlast` stable until the
  handshake completes.

### 3.2 `m_axis_decisions` — decision output (master)

| Signal     | Width | Direction | Purpose |
|------------|-------|-----------|---------|
| `tdata`    | 1     | out       | bit[0]: decision (1 = FP16, 0 = INT8) |
| `tvalid`   | 1     | out       | Handshake: decision is valid |
| `tready`   | 1     | in        | Handshake: downstream is ready to accept |
| `tlast`    | 1     | out       | Always 1 (each decision is a one-beat packet) |

**Protocol**: one beat per processed tile. If the downstream consumer
is not ready, the decision queues in the internal FIFO (up to 16 deep);
when the FIFO fills, the block stops accepting scores via `tready`.

---

## 4. Logical operations (compiler-facing)

Below are the abstract operations a compiler emits. Each maps to a
short sequence of register writes / stream beats; the exact mapping
is in §5 (C API stub).

| Op              | Inputs              | Outputs        | Description |
|-----------------|---------------------|----------------|-------------|
| `PC_QUERY`      | -                   | INFO struct    | Read all `INFO_*` registers in one transaction |
| `PC_RESET`      | -                   | -              | Soft reset accumulators + clear FIFO |
| `PC_ENABLE`     | -                   | -              | Set bit[1] of `CTRL` |
| `PC_PUSH_TILE`  | `N` scores          | -              | Stream a tile via `s_axis_scores` with `tlast` on the final beat |
| `PC_READ`       | -                   | decision bit   | Pop one decision from the FIFO; blocks if empty |
| `PC_READ_BATCH` | `count`             | `count` bits   | Pop up to `count` decisions; non-blocking |
| `PC_STATUS`     | -                   | STATUS bits    | Read `STATUS` (idle / in-progress / FIFO state) |

The whole block has no other side effects. There is no clock-gating
register, no power state, no DFT scan-chain control at this stub level
(those land on the chip-level ISA).

---

## 5. C API stub

What a runtime built on top of this ISA looks like. Concrete header
sketch:

```c
// lhsi_precision_controller.h — LonghornSilicon precision controller driver API.

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    uint32_t block_m;
    uint32_t block_n;
    uint32_t n;            // = block_m * block_n
    uint32_t score_width;  // bits per score
    uint32_t threshold;
    uint16_t isa_major;
    uint16_t isa_minor;
} lhsi_pc_info_t;

// Handle holds the AXI-Lite base, the AXI-Stream channel, and FD/mapping state.
typedef struct lhsi_pc_handle lhsi_pc_handle_t;

// Open / close
int  lhsi_pc_open (const char *device, lhsi_pc_handle_t **out);
void lhsi_pc_close(lhsi_pc_handle_t *h);

// Configuration queries (read INFO_* registers)
int  lhsi_pc_query(lhsi_pc_handle_t *h, lhsi_pc_info_t *out);

// Control
int  lhsi_pc_reset (lhsi_pc_handle_t *h);
int  lhsi_pc_enable(lhsi_pc_handle_t *h, bool enable);

// Streaming — push a complete tile.
//   scores: pointer to N signed integers, score_width-bits each.
//   stride: bytes between consecutive scores in the source buffer.
// The driver clamps each score to score_width bits and asserts tlast on
// the final beat. Blocks until the entire tile has been accepted by the
// block's tready handshake.
int  lhsi_pc_push_tile(lhsi_pc_handle_t *h,
                       const void *scores, size_t stride);

// Pop a single decision. Blocks if FIFO is empty.
int  lhsi_pc_read_decision(lhsi_pc_handle_t *h, bool *out_fp16);

// Pop up to `count` decisions; returns the number actually popped (0..count).
int  lhsi_pc_read_decisions(lhsi_pc_handle_t *h,
                            bool *out_fp16, size_t count);

// Diagnostics: read STATUS.
typedef struct {
    bool idle;
    bool decision_valid;
    bool tile_in_progress;
    bool fifo_full;
    uint32_t fifo_level;
    uint32_t tile_count;
} lhsi_pc_status_t;
int  lhsi_pc_status(lhsi_pc_handle_t *h, lhsi_pc_status_t *out);
```

Return codes follow the usual `0 = success, -errno = failure` Linux
convention.

---

## 6. Synthesis-time configuration

The following parameters are **baked in at synthesis** and cannot be
changed at runtime. A compiler reads them via the `INFO_*` registers
and tailors its code generation accordingly:

| Parameter      | Default | Range (validated)               | Notes |
|----------------|---------|---------------------------------|-------|
| `BLOCK_M`      | 64      | 16, 32, 64, 128, 256            | Must combine with `BLOCK_N` so `N` is a power of two |
| `BLOCK_N`      | 64      | 16, 32, 64, 128, 256            | (same) |
| `SCORE_WIDTH`  | 8       | 4, 6, 8, 10, 12, 16             | Defines `tdata` width on `s_axis_scores` |
| `THRESHOLD`    | 10      | **10 only** in stub             | The shift-and-add hardware path implements ×10 exactly |
| `FIFO_DEPTH`   | 16      | 4, 8, 16, 32                    | Decision output FIFO depth |

A future revision (`pc-isa-0.2`) will expose `THRESHOLD` as a
runtime-writable register; the RTL change is straightforward but breaks
the current "free shift-and-add" power/area story.

---

## 7. Bit-accurate reference

The Python model at
[`sw/reference_model/precision_controller_ref.py`](../../sw/reference_model/precision_controller_ref.py)
is the canonical bit-accurate reference for this ISA. A compiler can
verify its codegen against the model directly:

```python
from precision_controller_ref import PrecisionController

pc = PrecisionController()
decision = pc.process_tile(scores)   # exactly what the chip will return
```

The model is verified bit-exact against the RTL testbench's 143 replay
tiles (`sw/reference_model/test_precision_controller_ref.py`). Any
divergence between the model and the chip is a bug in this spec.

---

## 8. Integration phases

This stub supports a staged integration before silicon exists. See
the README on this branch for the phase-by-phase plan; in short:

- **Phase 0 (now)**: compiler targets the Python reference model.
- **Phase 1 (ZCU102/104)**: compiler targets the AXI interface via PYNQ
  on an FPGA prototype. Same memory map, same streaming protocol.
- **Phase 2 (multi-block FPGA)**: precision controller is one of four
  AXI-attached blocks; full chip-equivalent on FPGA.
- **Phase 3 (silicon)**: real TSMC 16FFC chip, same software stack
  re-targeted from FPGA to PCIe-attached accelerator.

Throughout all phases, the interface described in this document is
the stable contract.

---

## 9. Open questions for collaborators

- Should `THRESHOLD` be elevated to runtime (cost: one register, one
  parameterized multiplier — kills the free shift-and-add path)?
- Is a single-bit decision stream sufficient, or do we want to also
  emit the raw `max`/`sum` values for compiler-side calibration?
- Should the decision FIFO support coalesced reads (e.g., 32 decisions
  packed into one word) for high-throughput batching?
- Are AXI-Lite + AXI-Stream the right interfaces, or should we expose
  a CHI / TileLink / custom protocol that matches the rest of the chip?

These are the conversation starters for the compiler-integration
meeting.
