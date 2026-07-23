# Token Importance Unit — Interface Specification (Stub)

**Status**: pre-tape-out stub for compiler integration. Stable for the
Token Importance Unit block (TIU) only; will be unified with the rest of the
LonghornSilicon ISA when the Attention Compute Unit, KV Cache Engine, and Memory
Hierarchy Controller blocks land.

**Version**: `tiu-isa-0.1` — first public draft, 2026-07-18.

**Scope**: This document describes the externally-visible interface to the
`token_importance_unit` block as it will appear on the chip, on an FPGA prototype
(ZCU102/104), or in the bit-accurate software model
(`sw/reference_model/tiu_ref.py`). A compiler backend targeting the TIU writes
against this interface; the hardware implementation must conform to it.

---

## 1. Block overview

The Token Importance Unit is the H2O heavy-hitter eviction core for the KV cache.
It maintains `N_SLOTS` KV-cache slots, each holding an accumulated attention-mass
score (the heavy-hitter-oracle statistic). It exposes three operations and one
combinational tier read:

```
   LOAD  (slot)          : score[slot] := 0, valid[slot] := 1     -- install a token
   ACC   (slot, weight)  : score[slot] := sat_add(score[slot], w) -- add attention mass
   EVICT ()      -> slot : serialized argmin over valid slots     -- pick + free victim
   TIER  (thr)   -> keep : tier_keep[k] = valid[k] & (score[k] >= thr)
```

- **ACC** adds the per-step attention mass a cached token just received, one
  weight per cycle, saturating at `SCORE_MAX = 2^SCORE_WIDTH − 1`. Accumulators
  are **distributed** (one saturating adder per slot, no shared broadcast net).
- **LOAD** installs a fresh token: zero its score, mark it valid. LOAD has
  priority over ACC when both target the same slot in the same cycle.
- **EVICT** runs a **serialized argmin** (one comparator, no wide combinational
  min tree) over the valid slots and returns the minimum-mass slot — the token to
  drop — then frees it.
- **TIER** is a combinational read: `N_SLOTS` parallel comparators emit a
  per-slot keep/demote flag against the programmable `TIER_THRESHOLD`.

> **The H2O recent-window / KV-budget bookkeeping is a control-layer wrapper, not
> part of this core.** The silicon core is *accumulate + argmin + tier*; the
> software layer decides *when* to LOAD and EVICT (see
> `sw/reference_model/example_compiler_use.py`). This mirrors how block 1 shipped
> just the ratio gate and left the tile loop to the compiler.

**Latency:** LOAD and ACC complete in 1 cycle from idle; EVICT takes
`N_SLOTS + 1` cycles (the serialized argmin scan + result). TIER is combinational.
**Throughput:** one LOAD/ACC op per cycle; one eviction per `N_SLOTS + 1` cycles.
**Footprint:** 95 flip-flops at `N_SLOTS = 8`; 15 072 µm² die, ~660 µW at
SkyWater Sky130A (`N_SLOTS = 4` physical proxy; see §6 and `docs/sky130_signoff.md`).
**Bit-exact reference:** `sw/reference_model/tiu_ref.py` (40/40 RTL replay
evictions match).

---

## 2. Address space and memory map

The block exposes a 256-byte AXI-Lite slave register window. All registers are
32-bit, word-aligned. The base address is set at chip integration time (e.g.,
`0x4000_2000` on the ZCU102 PYNQ overlay). `SLOT_WIDTH = clog2(N_SLOTS)`.

| Offset | Name | Access | Reset | Purpose |
|--------|------|--------|-------|---------|
| `0x00` | `CTRL`              | RW    | 0x0   | bit[0]: `soft_reset` (write-1 to pulse; clears `valid[]`); bit[1]: `enable` |
| `0x04` | `STATUS`            | R     | 0x1   | bit[0]: `idle` (= `!busy`); bit[1]: `evict_valid`; bit[2]: `scan_in_progress` (= `busy`) |
| `0x08` | `INFO_N_SLOTS`      | R     | (syn) | Synthesis-time `N_SLOTS` |
| `0x0C` | `INFO_SCORE_WIDTH`  | R     | (syn) | Synthesis-time `SCORE_WIDTH` (bits per accumulated-mass score) |
| `0x10` | `INFO_WEIGHT_WIDTH` | R     | (syn) | Synthesis-time `WEIGHT_WIDTH` (bits per ACC increment) |
| `0x14` | `INFO_VERSION`      | R     | 0x0001| ISA version: bits[15:8] = major, bits[7:0] = minor |
| `0x18` | `TIER_THRESHOLD`    | RW    | 0x0   | Accumulated-mass keep/demote boundary (`SCORE_WIDTH` bits) |
| `0x1C` | `OP_LOAD`           | W     | —     | Write a slot index in bits[`SLOT_WIDTH`−1:0] → LOAD that slot |
| `0x20` | `OP_ACC`            | W     | —     | bits[`SLOT_WIDTH`−1:0] = slot; bits[16+`WEIGHT_WIDTH`−1:16] = weight → ACC |
| `0x24` | `OP_EVICT`          | W     | —     | Write-1 to bit[0] → trigger a serialized-argmin eviction scan |
| `0x28` | `EVICT_RESULT`      | R     | 0x0   | bit[31]: `valid` (1 once the scan completed); bits[`SLOT_WIDTH`−1:0]: victim slot |
| `0x2C` | `TIER_KEEP`         | R     | 0x0   | bit[k] = `tier_keep[k]` for k in 0..`N_SLOTS`−1 (1 = keep, 0 = demote; the CQ-8/CQ-4 value-precision reading is **retired** under KVE's flat CQ-3-rot tier — see §5) |
| `0x30` | `SLOT_VALID`        | R     | 0x0   | bit[k] = `valid[k]` (diagnostic occupancy map) |
| `0x34` | `SCORE_SEL`         | RW    | 0x0   | Slot index selecting which score `SCORE_READ` returns (diagnostic) |
| `0x38` | `SCORE_READ`        | R     | —     | Accumulated mass of the slot selected by `SCORE_SEL` (diagnostic) |

**Conventions:**
- `RW` = read/write; `R` = read-only; `W` = write-only trigger; `(syn)` = value
  fixed at synthesis time.
- Writes to read-only registers are silently dropped; reading reserved offsets
  returns 0.
- `OP_*` writes are ignored while `STATUS.scan_in_progress` is asserted (the
  block is mid-eviction). A compiler polls `STATUS.idle` before issuing the next
  op, or uses the streaming interface (§3) which handles backpressure in hardware.

---

## 3. Streaming op interface

For high throughput, the op stream is also exposed as an AXI-Stream channel; the
AXI-Lite window in §2 is then used only for `INFO_*`, `TIER_THRESHOLD`, and
diagnostics. This is the recommended path: one op per cycle with hardware
backpressure, matching the RTL's native op ports.

### 3.1 `s_axis_ops` — operation input (slave)

| Signal   | Width                              | Direction | Purpose |
|----------|------------------------------------|-----------|---------|
| `tdata`  | `2 + SLOT_WIDTH + WEIGHT_WIDTH`     | in        | `{op[1:0], slot[SLOT_WIDTH-1:0], weight[WEIGHT_WIDTH-1:0]}` |
| `tvalid` | 1                                  | in        | Handshake: op is valid |
| `tready` | 1                                  | out       | Handshake: block is ready to accept |

`op` encoding: `0 = ACC`, `1 = LOAD`, `2 = EVICT` (matching the golden trace and
`analysis/gen_tiu_testvectors.py`). For LOAD, `weight` is ignored; for EVICT,
`slot` and `weight` are ignored. `tready` deasserts for `N_SLOTS` cycles after an
EVICT while the argmin scan runs.

### 3.2 `m_axis_evict` — eviction output (master)

| Signal   | Width        | Direction | Purpose |
|----------|--------------|-----------|---------|
| `tdata`  | `SLOT_WIDTH` | out       | The evicted (minimum-mass) victim slot |
| `tvalid` | 1            | out       | Handshake: victim is valid (one beat per EVICT) |
| `tready` | 1            | in        | Handshake: downstream (KVCE) is ready |

One beat per EVICT op, emitted `N_SLOTS + 1` cycles after the EVICT is accepted.
`tier_keep` is not streamed — it is a continuously-valid combinational output the
KV cache engine samples when it (re)compresses a token's value (§5).

---

## 4. Logical operations (compiler-facing)

| Op            | Inputs            | Outputs        | Description |
|---------------|-------------------|----------------|-------------|
| `TIU_QUERY`   | —                 | INFO struct    | Read all `INFO_*` registers in one transaction |
| `TIU_RESET`   | —                 | —              | Soft reset: clear `valid[]` (empty the cache) |
| `TIU_ENABLE`  | —                 | —              | Set bit[1] of `CTRL` |
| `TIU_LOAD`    | slot              | —              | Install a fresh token in `slot` (score := 0, valid := 1) |
| `TIU_ACC`     | slot, weight      | —              | Add `weight` of attention mass to `slot` (saturating) |
| `TIU_EVICT`   | —                 | victim slot    | Run the argmin; return + free the minimum-mass valid slot |
| `TIU_SET_TIER`| threshold         | —              | Program `TIER_THRESHOLD` |
| `TIU_READ_TIER`| —                | keep bitmap    | Read `TIER_KEEP` (per-slot keep/demote) |

The block has no other externally-observable side effects: no clock-gating
register, no power state machine, no DFT scan-chain control at this stub level
(those land on the chip-level ISA).

---

## 5. Tier handshake to the KV Cache Engine

The TIU tells the KV Cache Engine (block 2) how to treat each cached token —
**keep**, **demote**, or **evict** — via two channels:

1. **Evict** (`m_axis_evict` / `EVICT_RESULT`): the minimum-mass victim's K and V
   are **dropped** and the slot freed. This is the per-token lever that applies to
   **both** K and V.
2. **Keep / demote** (`TIER_KEEP` against `TIER_THRESHOLD`): for a surviving
   token in slot `s`, the KVCE reads `tier_keep[s]` when it (re)compresses that
   token's **value**:
   - `tier_keep[s] = 1` (heavy hitter) → store the **value at CQ-8** (per-token INT8).
   - `tier_keep[s] = 0` (demote)       → store the **value at CQ-4** (per-token INT4).

> **RETIRED under CQ-3-rot.** Once KVE moved to a single flat rotated-INT3 value tier
> (CQ-3-rot), there is no per-token value bit-width left to select, so the CQ-8/CQ-4
> value-precision role of `tier_keep` is dropped — the TIU keeps only its **evict-or-keep**
> lever (item 1). This section is kept for provenance; see `../../DECISIONS.md` and
> `docs/tier_handshake.md`.

The tier is a per-token **value**-precision lever only. Keys stay uniform
per-channel (per-token key demotion collapses GQA); the whole key cache shares one
ChannelQuant key tier, set globally. `tier_keep` is emitted as `N_SLOTS` parallel
comparators (no read-mux), so it adds no fanout to the argmin datapath — this is
what kept the Sky130 sign-off at 0 violations after the port was added. The full
protocol, verified end-to-end with the ACU precision controller in the loop, is in
`docs/tier_handshake.md`.

---

## 6. Synthesis-time configuration

The following parameters are baked in at synthesis and cannot be changed at
runtime. A compiler reads them via the `INFO_*` registers and tailors its
code generation accordingly.

| Parameter      | Default | Range (validated)         | Notes |
|----------------|---------|---------------------------|-------|
| `N_SLOTS`      | 8       | 2, 4, 8, 16, 32           | Number of KV-cache slots (per head, or per shared group) |
| `SCORE_WIDTH`  | 8       | 6, 8, 10, 12, 16          | Accumulated-mass width; **8 is loss-free** (h2o-deep-analysis) |
| `WEIGHT_WIDTH` | 8       | 4, 6, 8                   | Per-step attention-mass increment width |

`SCORE_WIDTH = 8` is set from the accumulator-bit-width study
(`docs/findings/h2o-deep-analysis.md`): 8 bits is loss-free for the eviction
ranking on Qwen2 (Δ −0.002 acc_norm); 6 bits breaks (−0.022). Fewer score bits
also shrink the argmin read-mux, which drove the Sky130 max-transition closure.

**Physical proxy: `N_SLOTS = 4`.** The shipped RTL default is `N_SLOTS = 8` (what
the testbench verifies bit-exact). The Sky130 physical run synthesizes an
`N_SLOTS = 4` proxy via `SYNTH_PARAMETERS` — exactly as the KV Cache Engine ships
a `VECTOR_DIM = 8` proxy — to shrink the argmin-mux fanout and clock tree so the
clock-root fanout clears the flow's limit; the datapath is parameter-identical.
FF count: **95 @ N_SLOTS=8** (real), **55 @ N_SLOTS=4** (proxy, the sequential-cell
count in the signoff metrics). Real `N_SLOTS` is set per-instantiation.

The FF count tracks the closed form
`N_SLOTS·(SCORE_WIDTH+1) + SCORE_WIDTH + 3·SLOT_WIDTH + 3` (92 @ N_SLOTS=8;
yosys keeps ~+3 slot-index FFs un-merged → 95 synthesized, the value the CI gate
pins).

---

## 7. Bit-accurate reference

The Python model at [`sw/reference_model/tiu_ref.py`](../../sw/reference_model/tiu_ref.py)
is the canonical bit-accurate reference for this ISA:

```python
from tiu_ref import TokenImportanceUnit

tiu = TokenImportanceUnit()
tiu.load(0); tiu.acc(0, 200)
victim = tiu.evict()             # exactly what the chip will return
keep   = tiu.tier(threshold=40)  # per-slot keep/demote, exactly as tier_keep
```

The model is verified bit-exact against the RTL by replaying the committed golden
trace `rtl/tb/testvectors/tiu_trace.hex` (48 real Qwen2 tokens, 40 evictions)
through both the model and the SystemVerilog replay testbench: all 40 eviction
victims agree (`sw/reference_model/test_tiu_ref.py`). Any divergence between the
model and the chip is a bug in this specification, not in the chip.

---

## 8. Integration phases

- **Phase 0 (now)**: compiler targets the Python reference model.
- **Phase 1 (ZCU102/104)**: compiler targets the AXI interface via PYNQ on an
  FPGA prototype. Same memory map, same streaming protocol.
- **Phase 2 (multi-block FPGA)**: the TIU is one of four AXI-attached blocks;
  the tier handshake to the KV Cache Engine is exercised on hardware.
- **Phase 3 (silicon)**: TSMC 16FFC chip, same software stack re-targeted from
  FPGA to PCIe-attached accelerator.

Throughout all phases, the interface in this document is the stable contract.

---

## 9. Change log

- `tiu-isa-0.1` (2026-07-18): First public draft. Stable for the Token Importance
  Unit block only. Documents LOAD/ACC/EVICT + the tier read, the AXI-Lite
  register/streaming map, `N_SLOTS = 4` physical proxy, and the KVCE tier
  handshake.
