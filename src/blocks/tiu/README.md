# Token Importance Unit (TIU)

**A per-token importance scorer + eviction controller for the KV cache** — block 3 of the
LonghornSilicon (Lambda) decode-attention accelerator. It decides, per cached token, whether
to **keep or evict** its KV entry, so the KV cache stays within a fixed on-die budget as
context grows. Product target: **TSMC 16nm (N16FFC)**; proven on open PDKs.

**Layout** (canonical block template): `sw/ rtl/ pdk/ docs/ research/` (+ `analysis/`,
`paper/`, `DECISIONS.md`). Block root: `src/blocks/tiu/`.

> **Branch model — RTL lives on `rev0`, not `main`.** `main` is a clean scaffold (docs,
> `pdk/` configs, Python golden reference, `results/`) with **no `.sv`/`.v`**. Contributors
> branch from `rev0` and PR into it; a lead blesses and merges to `main`. **To see/build the
> RTL: `git checkout rev0`.** Full model: [`docs/REVISION_SYNC_SOP.md`](../../../docs/REVISION_SYNC_SOP.md) §6a.

---

## TL;DR

| | |
|---|---|
| **What** | Per-token importance scorer + eviction controller for the KV cache |
| **Why** | KV cache grows linearly with context; a fixed on-die budget needs a policy for *which* tokens to drop first |
| **How** | **H2O** — accumulate each token's post-softmax attention mass; keep a recent local window + the top "heavy-hitter" tokens by accumulated mass; evict the rest |
| **Signal** | Post-softmax attention mass (the ACU sparsity study proved pre-softmax proxies fail at r≈0, post-softmax works at r≈0.99) |
| **Integration** | Emits the **tier signal** the KVE consumes. Under KVE's flat CQ-3-rot (INT3) value tier there is no per-token value bit-width to select, so only the **evict/keep** lever survives — the keep→CQ-8 / demote→CQ-4 value-precision reading is retired ([`DECISIONS.md`](DECISIONS.md), [`docs/tier_handshake.md`](docs/tier_handshake.md)) |
| **Verified (algorithm)** | HellaSwag `acc_norm` within **−0.006** of full cache down to **25% KV budget** on Qwen2-0.5B |
| **Status** | **Sky130: metrics clean @40 MHz but GDS not committed → `no-gds` (NOT signed off).** GF180: config-only. See [Status](#status). |

The scorer is an accumulator + a running top-k — a natural streaming datapath. The H2O
algorithm is prior art (Zhang et al., NeurIPS 2023); **the contribution of this block is the
streaming silicon implementation** and its integration with the ChannelQuant tier signal.

---

## How H2O works

The **Heavy-Hitter Oracle** observation: attention mass is highly concentrated — a small,
stable set of tokens receives most of the attention across the whole sequence. Track them and
throw the rest away. Per (layer, head), for each cached key token *j*:

1. **Accumulate** its received attention mass: `acc[i,j] = Σ_{q≤i} A[q,j]` (a running sum,
   one add per token per step — cheap and streaming).
2. Maintain a fixed cache budget of **C** tokens. Once the sequence exceeds C, keep a **recent
   local window** of L tokens plus the **top (C − L) heavy hitters** by accumulated mass, and
   **evict** everything else. Evicted tokens' K/V are never attended to again.

---

## Algorithm result — verified on Qwen2

HellaSwag `acc_norm`, n=500, H2O eviction on every layer/head of Qwen2-0.5B (recent-window
share = 50% of budget). "KV budget" is cache size C as a fraction of sequence length. Source:
[`analysis/h2o_qwen05b_n500.json`](analysis/h2o_qwen05b_n500.json) (via
[`analysis/h2o_analysis.py`](analysis/h2o_analysis.py)).

| KV budget | acc_norm | Δ vs full cache |
|---|---|---|
| 100% (full) | 0.498 | — |
| 75% | 0.490 | −0.008 |
| 50% | 0.496 | −0.002 |
| 35% | 0.496 | −0.002 |
| **25%** | **0.492** | **−0.006** |
| 15% | 0.454 | −0.044 |
| 10% | 0.376 | −0.122 |

**H2O holds accuracy to within −0.006 of the full cache down to a 25% KV budget, then falls
off sharply below ~15%.** A cache of ~25–30% of context is near-lossless on this workload.
Gold config: **recent-window ratio 0.5, KV budget 25%** ([`docs/findings/h2o-analysis.md`](docs/findings/h2o-analysis.md)).

```sh
python analysis/h2o_analysis.py --model Qwen/Qwen2-0.5B --n 500
```

### All three blocks together

Composed in chip order — KVCE decompress → scores → **TIU** keep/evict → APA route — on Qwen2
(HellaSwag n=1000, gold config frac 0.25 / recent-ratio 0.5;
[`analysis/full_stack_qwen05b_n1000.json`](analysis/full_stack_qwen05b_n1000.json) +
`full_stack_qwen15b_n1000.json` via
[`analysis/full_stack_integration.py`](analysis/full_stack_integration.py)), Δ vs the FP16
full-cache baseline:

| config | Qwen2-0.5B | Qwen2-1.5B |
|---|---|---|
| TIU evict (25% budget) | −0.016 | −0.034 |
| KVCE (cq4+) | −0.015 | −0.003 |
| APA | +0.001 | −0.002 |
| **ALL 3 stacked** | **−0.033** | **−0.030** |
| ALL 3 + graded value demotion | −0.023 | −0.029 |

Stacking **75% cache eviction × 4-bit KV × ~all-INT8 attention** costs only ~3% acc_norm.
Per-token *key* demotion is incompatible with ChannelQuant's per-channel key path; per-token
*value* demotion recovers ~1 pt on 0.5B but at a **memory cost** (~5.9 vs 4.0 b/val) — graded
is a memory/accuracy trade, not a free win
([`docs/findings/all-three-blocks-integration.md`](docs/findings/all-three-blocks-integration.md),
[`docs/findings/graded-value-2bit.md`](docs/findings/graded-value-2bit.md)).

---

## How this fits in LonghornSilicon

```
┌──────────────────────────────────────────────────────────────────────┐
│              LonghornSilicon LLM Inference Accelerator (N16FFC)       │
│                                                                      │
│   ┌──────────────────┐          ┌────────────────────────┐          │
│   │  ACU (block 1)   │  scores  │  Token Importance Unit  │          │
│   │  precision ctrl  │─────────▶│  (this block, block 3)  │          │
│   │  INT8 vs FP16    │          │  H2O accumulated mass   │          │
│   └────────┬─────────┘          │  → keep / evict         │          │
│            │  K, V              └───────────┬────────────┘          │
│            ▼                                │ tier signal (evict/keep)│
│   ┌─────────────────────────┐               ▼                       │
│   │  KV Cache Engine        │◀───── keep / evict                    │
│   │  (block 2) ChannelQuant │                                       │
│   └─────────────┬───────────┘                                       │
│                 ▼                                                     │
│   ┌─────────────────────────┐   ┌──────────────────────┐             │
│   │ Memory Hierarchy Ctrl.  │◀─▶│ Off-chip LPDDR5X      │             │
│   │ (block 4)               │   │ (cold KV + weights)   │             │
│   └─────────────────────────┘   └──────────────────────┘             │
└──────────────────────────────────────────────────────────────────────┘
```

> **Note (CQ-3-rot):** the `keep→CQ-8 / demote→CQ-4` value-precision ladder is the
> pre-CQ-3-rot design. KVE now stores a single flat rotated-INT3 value tier, so there is no
> per-token value bit-width to select — the value-precision role of `tier_keep` is retired and
> only the **evict/keep** lever remains. See [`docs/tier_handshake.md`](docs/tier_handshake.md).

| Block | Repo | Role |
|---|---|---|
| ACU (Attention Compute Unit) | [`src/blocks/acu`](../acu) | INT8 vs FP16 per tile, MAC array |
| KV Cache Engine | [`src/blocks/kve`](../kve) | ChannelQuant compress/decompress |
| **Token Importance Unit** | **this block** | Per-token keep/evict (H2O) |
| Memory Hierarchy Controller | spec-only stub | On-die SRAM ↔ off-chip LPDDR5X |

---

## Layout

```
src/blocks/tiu/
├── rtl/                 # (rev0 only) SystemVerilog DUT + tb/ (29/29 directed + 40/40 replay) + golden trace
├── pdk/                 # sky130/openlane/ (metrics, no GDS) · gf180/librelane/ (chipathon, config-only)
├── sw/reference_model/  # bit-accurate Python model + parity test + compiler entry point
├── analysis/            # Python H2O studies, trace capture, test-vector gen
├── docs/                # ISA spec, tier handshake, sign-off note, findings/
├── paper/               # block write-up (token_importance_unit.pdf)
├── research/            # the "why": rationale, dead ends
└── DECISIONS.md         # settled calls
```

---

## Status

Per the authoritative sign-off matrix — [`docs/PROGRESS.md`](../../../docs/PROGRESS.md)
(generated) and [`docs/REVISION_SYNC_SOP.md`](../../../docs/REVISION_SYNC_SOP.md) §5.2 for
the sign-off definitions.

| PDK | Macro | Status | Die | Freq |
|---|---|---|---|---|
| **sky130** | `token_importance_unit` | **no-gds** (metrics clean, but no committed GDS) | 15,072 µm² (0.015 mm²) | 40 MHz |
| gf180 | `token_importance_unit` | config-only (declared, not run) | — | — |

**Honest status: the RTL and algorithm are done and verified; the block is NOT signed off.**
The Sky130 LibreLane run produced **clean metrics** — DRC/LVS/antenna/setup/hold/slew/cap/
fanout all **0** across corners (`pdk/sky130/openlane/token_importance_unit/results/sky130_signoff_metrics.json`,
[`docs/sky130_signoff.md`](docs/sky130_signoff.md)) — but the **GDSII is not committed**, and
§5.2 requires a committed GDS for "signed off". Status is therefore **`no-gds`**: metrics but no
GDS. To reach full sign-off, re-run the flow and commit the GDS artifact.

**Verified (algorithm + RTL):**
- Algorithm validated on Qwen2 (near-lossless to 25% budget); gold config chosen.
- All-3-blocks integration composes within ~3% of FP16.
- RTL: distributed-accumulator + serialized-argmin eviction datapath; directed +
  randomized self-checking TB **29/29 bit-exact**; real-Qwen replay **40/40 evictions
  bit-exact** (`sim_realdata`); Python↔RTL reference parity **40/40** on the golden trace.
- Deep analysis: accumulator width `SCORE_WIDTH=8b` loss-free (−0.002), per-head kept
  ([`docs/findings/h2o-deep-analysis.md`](docs/findings/h2o-deep-analysis.md)).
- Compiler-facing ISA (`tiu-isa-0.1`, [`docs/isa/`](docs/isa/)), reference model
  ([`sw/reference_model/`](sw/reference_model/)) and paper section ([`paper/`](paper/)) shipped.
- TIU→KVCE tier handshake (`tier_keep`) verified with APA in the loop
  ([`docs/tier_handshake.md`](docs/tier_handshake.md)).

**Not yet:** commit the Sky130 GDS (→ signed-off), GF180 hardening run, TSMC 16FFC sign-off.
Chip roadmap: [`docs/ROADMAP.md`](../../../docs/ROADMAP.md).

---

## Reproduce

```sh
git checkout rev0
cd src/blocks/tiu/rtl
make sim          # functional sim, 29/29
make sim_realdata # real-Qwen trace replay, 40/40 evictions bit-exact
make synth        # Yosys synth smoke
# harden:
librelane pdk/sky130/openlane/token_importance_unit/config.json
```

---

## Known gotchas
Pitfalls that cost time — check before debugging. (Chip-wide gotchas: monorepo-root `README.md`.)

- **LHS box venv is read-only, no numpy/pip.** Use `/home/shadeform/cuda_advisor/.venv/bin/python`
  for numpy; reinstall `iverilog`/`yosys` each session. Prefer pure-Python golden generators.
- **ORFS ASAP7 is 4×-drawn.** Areas read 16× too large unless de-scaled — confirm the SITE size
  (`0.054×0.270`) before quoting µm².
- **`DESIGN_REPAIR_MAX_SLEW_PCT=0` DISABLES slew repair** (passes `-slew_margin 0`) — restore
  ~20% or you get thousands of false max-slew/cap violations.
- **LibreLane escaped-identifier instance naming** takes 3 different forms across
  `PDN_MACRO_CONNECTIONS` regex / `instances` placement / YAML quoting — match each exactly.

---

## References

- Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs*, NeurIPS 2023.
- LonghornSilicon ACU sparsity study — post-softmax attention mass predicts token importance
  (r≈0.99); pre-softmax proxies do not.
