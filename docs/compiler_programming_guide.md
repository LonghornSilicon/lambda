# Lambda — Compiler Programming Guide

**Version** `lh-isa-0.1` — first unified draft, 2026-07-18.
**Audience** compiler / runtime engineers writing a backend that emits a Lambda
schedule (the pre-compiled instruction program the host loads at boot). This is the
map above the per-block interface specs; it says what the compiler *emits*, what it
*decides*, and where the ground-truth models live.

> **Ground truth.** Every programmable surface here has a bit-exact software reference
> model (`src/golden/` and each block repo's `sw/reference_model/`). Develop and test a
> backend against those first; the RTL, the FPGA prototype, and the taped-out chip all
> conform. Authoritative numbers live in [`arch.yml`](../arch.yml); the token-level
> pipeline is narrated in [`dataflow_walkthrough.md`](../dataflow_walkthrough.md).

---

## 1. The programming model in one paragraph

Lambda runs a transformer **decode** as a **static schedule**: a pre-compiled sequence
of **LSU instructions** (the chip-level ISA) that the host DMAs into the LSU's 4 KB
instruction RAM at boot. There are no branches in the hot loop — decode is structurally
identical layer to layer, so the compiler unrolls one layer's schedule and the runtime
walks it once per layer per token. Instructions **dispatch** work to the compute blocks
(MatE, VecU, KVE, TIU) and to the memory subsystem (MSC/DMA); the blocks run
asynchronously and the LSU issues ahead speculatively. Per-workload behaviour (precision
policy, KV codec tier, eviction mode) is selected through **CSRs** the host writes over
PCIe BAR0 before launching a schedule. So a Lambda backend produces two artifacts: **(a)
a CSR configuration** and **(b) an LSU instruction schedule** (plus any VecU microcode it
overrides).

---

## 2. Blocks the compiler targets

| Block | id | What the compiler does with it |
|---|---|---|
| **LSU** — Layer Sequencer | `layer_sequencer` | *Is* the program. Emits the instruction stream (§4). |
| **MatE** — Matrix Engine | `matrix_engine` | 8×8 INT8×INT4 systolic. `ISSUE_MAT_E` for QKV/FFN projections (weight-stationary) and Q·Kᵀ / P·V (output-stationary via a dataflow-mode CSR). |
| **VecU** — Vector Unit | `vector_unit` | 8-lane FP16/BF16 SIMD microcode. `ISSUE_VEC_U <op>` for online softmax, RoPE, RMSNorm, SiLU, residual, sampling. |
| **KVE** — KV Cache Engine | `kv_compression_engine` | ChannelQuant compress/decompress. `ISSUE_KCE_COMP` / `ISSUE_KCE_DECOMP`; codec tier via CSR. Owns the KV data format (§5). |
| **TIU** — Token Importance Unit | `token_importance_unit` | Per-KV-block importance accumulator. Fed from softmax weights; drives H2O eviction (MSC) and per-block ChannelQuant tier (KVE). Mode via CSR. |
| **MSC** — Memory Subsystem Ctrl | `memory_subsystem_controller` | DMA + block-table TLB + SRAM crossbar + LPDDR5X. `LOAD_WEIGHTS` / DMA descriptors; paged-attention block mapping. |
| **HIF** — Host Interface | `host_interface` | PCIe Gen3 x1 / BAR0 CSR window + 16 doorbell queues. Boot: load schedule, DMA weights. |

Correspondence to the standalone block repos (which are the RTL of record for three of
these): **KVE** ⇢ `kv-cache-engine`; **TIU** ⇢ `token-importance-unit`; the
**precision-controller + MAC-array** work in `attention-compute-unit` is the ACU
research line — see the precision reconciliation in §6.

---

## 3. Boot & control surface (CSRs)

At power-on the host driver, over HIF/PCIe BAR0: (1) writes the LSU schedule into
instruction RAM, (2) DMAs W4 weights into LPDDR5X, (3) writes the CSR configuration,
(4) rings a doorbell. CSRs are the *static* knobs; the schedule is the *dynamic* program.
The compiler emits the CSR block; see [`src/isa/csr_map.h`](../src/isa/csr_map.h) for the
authoritative field list. The compiler-relevant CSRs:

| CSR group | Selects | Compiler guidance |
|---|---|---|
| `LAYER_CFG[l]` | per-layer dims, head count, GQA group, RoPE base | From the model config. |
| `MATE_MODE` | dataflow (weight-stationary / output-stationary), precision mode (§6) | Set output-stationary around Q·Kᵀ / P·V; weight-stationary for projections. |
| `KVE_TIER` | ChannelQuant tier CQ-8 / CQ-4 / CQ-4+; group `G`; outlier `k` | CQ-4+ (`G`=128, `k`=2) is the near-lossless default (~4.2 b/val). Per-layer granularity today. |
| `TIU_MODE` | `off` / `h2o` / `streaming_llm` / `adaptive_precision`; budget; threshold | `h2o` at a 25%-of-context budget is near-lossless; `adaptive_precision` also drives the per-block value tier. |
| `SAMPLING` | temperature / top-k / greedy | From the request. |

---

## 4. The LSU instruction schedule

The chip-level ISA is a set of **32 fixed-width 32-bit instructions**, three-lane issue
(scalar + vector-dispatch + DMA). The authoritative encoding is
[`src/isa/lsu.h`](../src/isa/lsu.h). The compiler emits, per layer, roughly this body
(from `dataflow_walkthrough.md`, one decode token):

```
LOAD_WEIGHTS   layer=l, slice=qkv_proj        ; DMA descriptor -> MSC (async)
ISSUE_MAT_E    qkv_proj, in=act_buf, out=qkv  ; weight-stationary GEMM (INT8×INT4)
ISSUE_VEC_U    rope,     q, k                  ; RoPE on Q,K (microcode)
ISSUE_KCE_COMP k, v -> kv_scratchpad           ; ChannelQuant compress (grouped keys)
SET_MODE       MatE = output_stationary
ISSUE_MAT_E    qk_dot, q, k_cache -> scores    ; Q·Kᵀ; KVE dequantizes K per-channel
ISSUE_VEC_U    softmax_online, scores -> P     ; FA-3 online softmax
                                               ;   (side-channel: per-KV-block weight -> TIU)
ISSUE_MAT_E    pv, P, v_cache -> o_head         ; P·V (output-stationary)
ISSUE_VEC_U    rmsnorm/residual/…, o -> act_buf ; FFN + norms (microcode ops)
…                                              ; FFN GEMMs via ISSUE_MAT_E, SiLU via VecU
```

Rules for the backend:
- **No branches in the hot loop.** Unroll the layer body; the runtime repeats it. Control
  instructions (loop/sync/doorbell/halt) bracket the schedule, not the inner tiles.
- **Issue ahead.** Dispatch instructions do not block; the LSU queues them behind the
  memory fetch. Order for correctness (RAW through the named SRAM buffers), not for
  stalls — the blocks self-synchronize on buffer readiness.
- **Tile to the fixed sizes.** MatE is 8×8; Q·Kᵀ / P·V stream in tiles the VecU softmax
  consumes (FA-3, ~32 scores/tile). Emit exactly the synthesized tile shape.

---

## 5. KV data format (the compiler's serialization contract)

The KVE owns the on-SRAM KV format; a backend that stages or inspects KV must match it.
A head is `[T, D]` FP16 (`D`=head dim). Full C++ structs:
`kv-cache-engine/sw/reference_model/channelquant_ref.hpp`.

- **Values / CQ-8 keys — per-token** `ValueBlob{T, D, bits, scales[T] (fp16),
  codes[T·D] (signed), payload}`. `payload` = INT4 nibble-packed (elt 2i→low nibble,
  2i+1→high; odd tail zero-padded) or INT8 bytes.
- **CQ-4/CQ-4+ keys — per-channel, grouped by `G`** `KeyBlob{T, D, bits, G, groups,
  keep[], outlier[], scales (per group, nk fp16), payload (per group INT4-packed),
  sidecar[T·k] (fp16 outlier columns, t-major)}`. Keys are **dequantized per-channel**
  (`INT4·FP16` + FP16 replay of the `k` outlier channels) before Q·Kᵀ — there is **no
  compressed-domain scoring path**.
- **Quant math** (bit-exact): `s = max(amax/qmax, EPS=2⁻¹⁴)` as fp16, `qmax`=7 (INT4)/127
  (INT8); `code = clamp(round_half_even(x/s), qmin, qmax)`; dequant `code·s` as fp32.
- **Effective bits/value**: value `4 + 16/D`; key CQ-4 `4 + 16/G`; CQ-4+ `+ (k/D)(16−4)`.
  CQ-4+ ≈ 4.2 b/val ≈ 3.8× vs FP16, near-lossless (HellaSwag within ~0.4–0.8 pt of FP16).

---

## 6. Precision model

**MatE datapath** (`arch.yml` `matrix_engine`). MatE is a **heterogeneous INT8 + FP16**
MAC array:

- **Weight / FFN GEMMs** — **INT8 activation × INT4 weight → INT24 accumulator** (W4A8).
  Always INT8×INT4; no per-tile choice.
- **Attention scores** `Q·Kᵀ` — **INT8 Q × per-channel-dequantized FP16 K → INT24**.
- **Attention `P·V`** — **INT8 OR FP16**, chosen **per tile** by the **ACU precision
  controller** (`max(|s|)·N > 10·sum(|s|)` → FP16, else INT8). The FP16 path is
  `FP16×FP16 → FP32-accumulate` in the same array.

**This is committed** (decided 2026-07-18; resolves the former `STATUS.md` §7 "no FP16
path" reconciliation — MatE gains the FP16 escape rather than dropping the precision
gate). The controller + MAC-array RTL live in `attention-compute-unit` (both
Sky130-signed-off); `precision_controller_ref.py` is 143/143 bit-exact vs its RTL and is
the calibration tool.

The compiler selects the policy through `MATE_MODE.precision`:

- **`adaptive`** (default): the precision controller drives the per-tile INT8/FP16
  decision at runtime from the scores — the compiler just enables it. On Qwen the gate
  routes ~99.99% INT8 (INT8 `P·V` is near-lossless there); the FP16 escape catches the
  rare peaked tile.
- **`static_w4a8`**: force INT8 `P·V` everywhere (skip the gate) — smallest energy,
  for workloads a calibration pass shows are fully INT8-safe.

FP16 mode area/power delta is TBD pending re-synthesis; functionally, the full
three-block stack with the gate active holds Δ−0.031 vs FP16 on Qwen2-0.5B (§7).

---

## 7. Token importance & the tier handshake

The TIU tracks importance **per KV block** (a block = a fixed run of tokens; `arch.yml`
uses 16-token blocks, 128 blocks, a 16-bit importance register each). It is fed a
side-channel from the VecU softmax: for each KV block that contributed to an attention
pass, the cumulative softmax weight is added to that block's importance register (~1 extra
µop per softmax tile — essentially free). Two consumers, both selected by `TIU_MODE`:

- **Eviction (→ MSC).** When the scratchpad fills, the MSC asks the TIU for the
  lowest-importance block and evicts it (H2O heavy-hitter retention). Near-lossless down
  to a **25%-of-context** budget; the `h2o` mode with a 50/50 recent-vs-heavy split is the
  measured sweet spot.
- **Per-block tier (→ KVE).** On re-compress, the KVE queries the TIU for the block's
  tier: high importance → **keep the block's VALUE at CQ-8**; low importance → **demote to
  CQ-4** (`tier_keep` in the TIU RTL: `valid && importance ≥ threshold`).

**Compiler rule — the tier is a VALUE lever; keys stay uniform per-channel.** ChannelQuant
keys are compressed per-channel (a scale shared across the token group) to protect GQA's
high-magnitude key channels; per-token/per-block *key* bit-width demotion degenerates to
per-token key scaling and collapses GQA (measured −0.17 vs −0.03 keeping keys uniform). So
the TIU's per-block lever for **keys is evict-or-keep only**; graded precision is a
**value-path** decision; the whole key cache shares one ChannelQuant key tier. Full
protocol: `token-importance-unit/docs/tier_handshake.md`. The three-block stack (TIU tier +
KVE + the ACU precision line) holds Δ−0.031 vs FP16 on Qwen2-0.5B with the precision gate
routing ~99.99% INT8 — the precision controller is score-only, so the codec and tier are
transparent to it.

---

## 8. What the compiler decides (summary)

| Decision | Surface | Guidance (measured) |
|---|---|---|
| Layer schedule / tiling | LSU instructions | Unroll one layer; tile to the synthesized MatE/VecU shapes. |
| KV codec tier | `KVE_TIER` CSR | CQ-4+ default; per-layer today (per-tile is an open CSR-granularity question, STATUS §7). |
| KV budget + eviction | `TIU_MODE` + control | `h2o`, budget ≈ 25% of context, 50/50 recent/heavy. |
| Per-block value tier | TIU threshold (runtime) + `TIU_MODE=adaptive_precision` | keep→CQ-8 / demote→CQ-4; calibrate the threshold to a mass percentile. |
| Precision mode | `MATE_MODE.precision` | `static_w4a8` for silicon-of-record; `adaptive` reserved pending the §6 reconciliation. |
| Sampling | `SAMPLING` CSR | From the request. |

Calibration: run the reference models over a representative workload offline to fix codec
tier, KV budget, tier threshold, and (if adaptive) the precision policy; bake into the CSR
block + schedule. `example_compiler_use.py` in each block's `sw/reference_model/` shows the
per-block calibration entry points.

---

## 9. Reference models & test vectors

| Block | Model | Parity |
|---|---|---|
| KVE ChannelQuant | `kv-cache-engine/sw/reference_model/channelquant_ref.{cpp,hpp,py}` | Python↔C++↔SV, 9/9 vectors |
| TIU | `token-importance-unit/sw/reference_model/tiu_ref.py` | Python↔RTL on the real-Qwen2 golden trace |
| ACU precision controller | `attention-compute-unit/sw/reference_model/precision_controller_ref.{py,cpp}` | 143/143 vs RTL |
| ACU MAC array | `…/mac_array_ref.{py,cpp}` | self-test clean |
| LSU assembler | `src/golden/lsu_asm.py` | *to draft* (see `src/isa/README.md`) |

End-to-end accuracy is measured with `token-importance-unit/analysis/full_stack_integration.py`
(TIU + KVE + precision gate on Qwen2 via HellaSwag).

---

## 10. Status & versioning

`lh-isa-0.1` (2026-07-18): first unified guide. Blocks KVE, TIU, and the ACU
precision-controller/MAC are RTL-complete and Sky130-signed-off as standalone units; MatE,
VecU, LSU, MSC, HIF are specified in `arch.yml` with HLS under `src/blocks/`. **Open ISA
items tracked in `STATUS.md` §7** and reserved in the headers: the §6 precision
reconciliation, KVE tier CSR granularity (per-layer vs per-tile), and the ISSUE_MAT_E +
ISSUE_VEC_U FA-3 fusion. Per-block ISA versions: `pc-isa-0.1`, `kv-isa-0.2`, `tiu-isa-0.1`.
The reference models are the ground truth and are versioned with the RTL.
