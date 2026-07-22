# Architecture — Lambda ACU decode-attention datapath

Text descriptions of the block diagram, dataflow, and pad map for the Chipathon
2026 GF180 submission. (Canonical chip/ISA/compiler docs live in the
LonghornSilicon `architecture` repo; this file is the submission-local summary.)

## 1. What it is

The **Lambda ACU** (Attention Compute Unit) is the decode-time attention +
KV-cache datapath of the LonghornSilicon "Lambda" inference chip. It targets the
memory-bound, single-query **decode** step of autoregressive LLM inference: for
each new token, attend over the cached K/V of all prior tokens and produce one
output row. It combines (a) a **KV-cache codec** that compresses stored K/V with
ChannelQuant, (b) **adaptive precision** (per-tile INT8 vs FP16), and (c) an
**H2O heavy-hitter** policy that scores/evicts low-importance cached tokens.

## 2. Dataflow (one decode step)

```
 Q (INT8, D=128) ─┐
 K (INT8, D=128) ─┼─► kve ──► (dequant K,V from cache) ─┐
 V (INT8, D=128) ─┘                                     │
                                                        ▼
                          mate_qkt:  scores[t] = Q · K[t]ᵀ     (INT24 accum)   [Phase 1]
                                                        │
                          precision_controller:  d_fp16 = ( max|S|·N > THR·Σ|S| )
                                                        │  (selects INT8 vs FP16 tile)
                                                        ▼
                          vecu_softmax:  P[t] = softmax(scores)[t]             [Phase 2]
                                                        │
                          token_importance_unit:  score[t] += P[t]; argmin → evict
                                                        │
                          mate_pv (INT8) / mate_pv_fp16 (FP16):
                                    o[n] = Σ_t P[t] · V[t][n]   (INT32 / fp32 accum)
                                                        ▼
                          o (attention output row, D-wide) ─► host
```

- **Accumulator widths.** The K-axis (hidden-dim) GEMM `mate_qkt` reduces over D
  → INT24. The token-reduction `mate_pv` reduces over context length → INT32
  (covers ~133k tokens before overflow). The FP16 path accumulates in internal
  fp32 and rounds to fp16 once.
- **Precision gate is division-free:** `max(|S|)·N > THRESHOLD·Σ|S|` — a free
  left-shift on the LHS and a constant-multiply (shift+add tree) on the RHS, so
  no divider on the critical path.

## 3. Serialization (why the SPI loader)

Head dim D=128 with INT8 Q/K/V means each vector is 128 bytes; the workshop slot
exposes only ~20 bidir pads. So the ACU cannot present its tensors in parallel on
the pads. Instead a narrow **SPI slave** (`rtl/spi_loader.sv`) streams Q/K/V in,
the datapath computes, and the D-wide output row streams back out. The loader is
byte-oriented (CMD / ADDR / DATA frames, auto-incrementing address) with a small
CSR window (CTRL, STATUS, SEQ_LEN, HEAD_DIM) and tensor streaming regions. This
is the standard "wide accelerator behind a serial test harness" pattern for a
pad-limited shuttle.

## 4. Pad map (workshop slot)

Workshop slot = **1 input pad + 20 bidir pads + 60 analog pads**, plus `clk`,
`rst_n`, 4/4 DVDD/DVSS; die 2935×2935 µm.

| pad            | dir | use                                            |
|----------------|-----|------------------------------------------------|
| `bidir[0]`     | in  | `spi_sclk`                                     |
| `bidir[1]`     | in  | `spi_cs_n`                                      |
| `bidir[2]`     | in  | `spi_mosi`                                      |
| `bidir[3]`     | out | `spi_miso`                                      |
| `bidir[19:4]`  | out | observation: 8-bit heartbeat + `busy`/`done`   |
| `input[0]`     | in  | spare / reserved external strobe               |
| `analog[59:0]` | —   | pass-through, unconnected at core level        |

`bidir_oe` is `0` for pads `[2:0]` (host-driven inputs) and `1` for `[19:3]`
(chip-driven outputs); `bidir_ie = ~bidir_oe`; pulls disabled; CMOS input
buffers, fast slew — the standard workshop-slot pad defaults.

## 5. Macro partitioning

Every block is a **standalone GF180 LibreLane macro** (own GDS/LEF/lib/netlist),
composed at the chip top. This keeps each block independently hardenable and
verifiable and matches the chipathon multi-macro reference flow. The composition
site is `rtl/lambda_acu.sv`; the padring interface is `rtl/chip_core.sv`. See
`README.md` for the block→macro table and RTL status, and `docs/build.md` for the
per-macro + integration build steps.
