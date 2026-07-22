# Token Importance Unit

This is the **Token Importance Unit (TIU)** block of the LonghornSilicon LLM
inference accelerator — **block 3 of four** targeting TSMC 16nm FinFET (N16FFC)
tape-out. It decides, per cached token, whether to **keep, demote, or evict** its
KV entry — so the KV cache stays within a fixed on-die budget as context grows.

> **Status: built and signed off.** The retention algorithm (H2O accumulated-mass)
> is validated on real Qwen2 traces (below); the RTL is verified (29/29 directed,
> 40/40 real-data replay) and signs off on Sky130 with **0 violations**; a
> bit-accurate Python reference model is at parity (40/40 evictions), and the
> compiler-facing ISA spec, reference model, and paper section are in `docs/isa/`,
> `sw/reference_model/`, and `paper/`. Follows the pattern of
> [`attention-compute-unit`](https://github.com/LonghornSilicon/attention-compute-unit)
> (block 1) and [`kv-cache-engine`](https://github.com/LonghornSilicon/kv-cache-engine) (block 2).

---

## TL;DR

| | |
|---|---|
| **What** | Per-token importance scorer + eviction/demotion controller for the KV cache |
| **Why** | KV cache grows linearly with context; a fixed on-die budget needs a policy for *which* tokens to drop first |
| **How** | **H2O** — accumulate each token's post-softmax attention mass; keep a recent local window + the top "heavy-hitter" tokens by accumulated mass; evict the rest |
| **Signal** | Post-softmax attention mass (the ACU sparsity study proved pre-softmax proxies fail at r≈0, post-softmax works at r≈0.99) |
| **Integration** | Emits the **tier signal** that the KV Cache Engine already consumes (keep → CQ-8, demote → CQ-4, evict → drop) — mixed-precision retention |
| **Verified (algorithm)** | HellaSwag acc_norm within **−0.006** of full cache down to **25% KV budget** on Qwen2-0.5B (n=500, per-layer/head H2O, recent-ratio 0.5; source `analysis/h2o_qwen05b_n500.json`) |
| **Status** | Built & signed off — RTL verified (29/29 directed `rtl/tb/tb_token_importance_unit.sv`, 40/40 real-data replay `rtl/tb/tb_realdata.sv`), **0-violation Sky130** (`openlane/token_importance_unit/results/signoff_metrics.json`), bit-exact Python reference at parity, ISA spec + paper shipped |

---

## How H2O works

The **Heavy-Hitter Oracle** (Zhang et al., 2023) observation: attention mass is
highly concentrated — a small, stable set of tokens receives most of the attention
across the whole sequence. Track them and you can throw the rest away.

Per (layer, head), for each cached key token *j*:

1. **Accumulate** its received attention mass: `acc[i,j] = Σ_{q≤i} A[q,j]`
   (a running sum, one add per token per step — cheap and streaming).
2. Maintain a fixed cache budget of **C** tokens. Once the sequence exceeds C, keep
   - a **recent local window** of L tokens (recency matters for coherence), plus
   - the **top (C − L) heavy hitters** by accumulated mass,

   and **evict** everything else. Evicted tokens' K/V are never attended to again.

The scorer is an accumulator + a running top-k — a natural streaming datapath, the
same shape as the precision controller (block 1). This is prior art as an
*algorithm*; **the contribution of this block is the streaming silicon
implementation** and its integration with the ChannelQuant tier signal.

---

## Algorithm result — verified on Qwen2

HellaSwag `acc_norm`, n=500, H2O eviction applied to every layer/head of
Qwen2-0.5B (recent-window share = 50% of budget). "KV budget" is the cache size C
as a fraction of the sequence length. Source: `analysis/h2o_qwen05b_n500.json` (measured by `analysis/h2o_analysis.py`):

| KV budget | acc_norm | Δ vs full cache |
|---|---|---|
| 100% (full) | 0.498 | — |
| 75% | 0.490 | −0.008 |
| 50% | 0.496 | −0.002 |
| 35% | 0.496 | −0.002 |
| **25%** | **0.492** | **−0.006** |
| 15% | 0.454 | −0.044 |
| 10% | 0.376 | −0.122 |

**H2O holds accuracy to within −0.006 of the full cache down to a 25% KV budget,
then falls off sharply below ~15%.** This sizes the block: a cache of ~25–30% of
context is near-lossless on this workload. (The classic H2O ~20% result,
reproduced on Qwen2.)

Reproduce:

```sh
python analysis/h2o_analysis.py --model Qwen/Qwen2-0.5B --n 500
```

Gold config: **recent-window ratio 0.5, KV budget 25%** (an even recency/heavy-hitter
split wins at every budget; see `docs/findings/h2o-analysis.md`).

---

## All three blocks together

The TIU is the last of the three live blocks. Composed in chip order — KVCE
decompress → scores → **TIU** keep/evict → APA route — on Qwen2 (HellaSwag n=1000,
gold config: frac 0.25, recent-ratio 0.5; sources `analysis/full_stack_qwen05b_n1000.json` + `full_stack_qwen15b_n1000.json` via `analysis/full_stack_integration.py`), Δ vs the FP16 full-cache baseline:

| config | Qwen2-0.5B | Qwen2-1.5B |
|---|---|---|
| TIU evict (25% budget) | −0.016 | −0.034 |
| KVCE (cq4+) | −0.015 | −0.003 |
| APA | +0.001 | −0.002 |
| **ALL 3 stacked** | **−0.033** | **−0.030** |
| ALL 3 + graded value demotion | −0.023 | −0.029 |

Stacking **75% cache eviction × 4-bit KV × ~all-INT8 attention** costs only ~3%
acc_norm. Two findings (`docs/findings/all-three-blocks-integration.md`): per-token
*key* demotion is incompatible with ChannelQuant's per-channel key path (keys stay
uniform per-channel), but per-token *value* demotion — the "mixed-precision
retention" lever — recovers ~1pt on 0.5B, but at a **memory cost** (~5.9 vs 4.0
b/val): the ladder only promotes above 4-bit, never below. Trying to make it
memory-neutral by demoting boring tokens to 2-bit craters accuracy with the current
codec (`docs/findings/graded-value-2bit.md`) — so graded is a memory/accuracy trade,
not a free win.

```sh
python analysis/full_stack_integration.py --model Qwen/Qwen2-0.5B --n 1000 --frac 0.25 --recent_ratio 0.5
```

---

## How this fits in LonghornSilicon

```
┌──────────────────────────────────────────────────────────────────────┐
│              LonghornSilicon LLM Inference Accelerator (16FFC)       │
│                                                                      │
│   ┌──────────────────┐          ┌────────────────────────┐          │
│   │  ACU (block 1)   │  scores  │  Token Importance Unit  │          │
│   │  precision ctrl  │─────────▶│  (this repo, block 3)   │          │
│   │  INT8 vs FP16    │          │  H2O accumulated mass   │          │
│   └────────┬─────────┘          │  → keep / demote / evict│          │
│            │  K, V              └───────────┬────────────┘          │
│            ▼                                │ tier signal            │
│   ┌─────────────────────────┐               ▼                       │
│   │  KV Cache Engine        │◀───── keep→CQ-8 / demote→CQ-4 / evict │
│   │  (block 2) ChannelQuant │                                       │
│   └─────────────┬───────────┘                                       │
│                 ▼                                                     │
│   ┌─────────────────────────┐   ┌──────────────────────┐             │
│   │ Memory Hierarchy Ctrl.  │◀─▶│ Off-chip LPDDR5X      │             │
│   │ (block 4)               │   │ (cold KV + weights)   │             │
│   └─────────────────────────┘   └──────────────────────┘             │
└──────────────────────────────────────────────────────────────────────┘
```

The TIU closes the loop on the KV cache: the ACU produces attention scores → the
TIU accumulates per-token importance and rules keep/demote/evict → the KV Cache
Engine applies the resulting precision tier (or frees the slot). Together the three
live blocks turn a linearly-growing FP16 KV cache into a **bounded, mixed-precision**
one.

| Block | Repo | Role |
|---|---|---|
| ACU (Attention Compute Unit) | [attention-compute-unit](https://github.com/LonghornSilicon/attention-compute-unit) | INT8 vs FP16 per tile, MAC array |
| KV Cache Engine | [kv-cache-engine](https://github.com/LonghornSilicon/kv-cache-engine) | ChannelQuant compress/decompress |
| **Token Importance Unit** | **this repo** | Per-token keep/demote/evict (H2O) |
| Memory Hierarchy Controller | not yet | On-die SRAM ↔ off-chip LPDDR5X |

---

## Repo layout

```
token-importance-unit/
├── analysis/          # Python: H2O algorithm study, trace capture, test-vector gen
│   ├── h2o_analysis.py                 # accuracy vs KV-budget sweep (this is the result above)
│   └── h2o_qwen05b_n500.json           # measured curve
├── rtl/               # SystemVerilog DUT + testbenches (29/29 + 40/40) + golden trace
├── openlane/          # LibreLane Sky130 sign-off (0 violations)
├── sw/reference_model/# bit-accurate Python model, parity test, compiler entry point
├── paper/             # block write-up (token_importance_unit.pdf)
└── docs/              # ISA spec, tier handshake, sign-off, SW overview, findings
```

## Roadmap

- [x] Algorithm validated (H2O accumulated-mass on Qwen2; near-lossless to 25% budget)
- [x] Gold config chosen (recent-ratio 0.5, 25% budget)
- [x] All-3-blocks integration verified (TIU+KVCE+APA compose within ~3% of FP16)
- [x] Deep analysis: long-ctx knee, per-head vs shared (keep per-head), accumulator width (SCORE_WIDTH=8b, loss-free −0.002; 10b only for long-ctx margin — `docs/findings/h2o-deep-analysis.md`)
- [x] RTL: distributed-accumulator + serialized-argmin eviction datapath, FF count 95 (synth/CI-pinned, `.github/workflows/ci.yml`; closed-form analytic bound 92 + ~3 yosys-unmerged slot-index FFs — see `rtl/token_importance_unit.sv` header)
- [x] Directed + randomized self-checking testbench (iverilog), 29/29 bit-exact
- [x] **Sky130 sign-off: 0 violations** across all checks (DRC/LVS/antenna/setup/hold/slew/cap/fanout) — `docs/sky130_signoff.md`
- [x] Replay testbench from real Qwen2 attention traces (`sim_realdata`, 40/40 evictions bit-exact)
- [x] TIU→KVCE tier-signal handshake (`tier_keep`), verified with APA in the loop (`docs/tier_handshake.md`)
- [x] Bit-accurate Python reference model at Python↔RTL parity (40/40 evictions on the golden trace) — `sw/reference_model/`
- [x] Compiler-facing ISA / interface spec (`tiu-isa-0.1`) — `docs/isa/token_importance_unit_isa.pdf`
- [x] Paper section with hardware results — `paper/token_importance_unit.pdf`
- [x] Software / reference-model overview — `docs/sw_overview.pdf`

## References

- Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs*, NeurIPS 2023.
- LonghornSilicon ACU sparsity study (`attention-compute-unit/docs/findings/sparsity-controller-finding.md`) — post-softmax attention mass predicts token importance (r≈0.99); pre-softmax proxies do not.

## Known gotchas
Pitfalls that cost time — check before debugging. (Chip-wide gotchas: monorepo-root `README.md`.)

- **LHS box venv is read-only, no numpy/pip.** Use `/home/shadeform/cuda_advisor/.venv/bin/python`
  for numpy; reinstall `iverilog`/`yosys` each session. Prefer pure-Python golden generators.
- **ORFS ASAP7 is 4×-drawn.** Areas read 16× too large unless de-scaled — confirm the SITE size
  (`0.054×0.270`) before quoting µm².
- **`DESIGN_REPAIR_MAX_SLEW_PCT=0` DISABLES slew repair** (passes `-slew_margin 0`) — restore ~20%
  or you get thousands of false max-slew/cap violations.
- **LibreLane escaped-identifier instance naming** takes 3 different forms across
  `PDN_MACRO_CONNECTIONS` regex / `instances` placement / YAML quoting — match each exactly.
