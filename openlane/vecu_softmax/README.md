# Sky130 OpenLane / LibreLane flow — `vecu_softmax`

End-to-end open-source RTL → GDSII flow for `vecu_softmax`, the **VecU decode
online-softmax slice** (FlashAttention-style running-max / running-sum with a
64-entry exp LUT, fp32 accumulator, fp16 reciprocal), targeting SkyWater
**Sky130A**. Same flow and tuning family as `../mate_pv_fp16` / `../mate_qkt` — it
reuses the identical IEEE fp16/fp32 primitives (fp16↔fp32, fp32_add/mul,
fp32→fp16), plus one new op (fp16_div for 1/ℓ).

> 130 nm Sky130 proxy, used for 16 nm estimates — Lambda targets TSMC 16 nm.

This is one of the two **flagship-parity backfills** (audit priority #1): `mate_qkt`
and `vecu_softmax` had GF180 sign-off but no Sky130, so the 130 nm flagship could not
do Q·Kᵀ scoring or softmax in silicon terms. This closes the softmax half. It is the
**rebalanced multi-cycle** revision (commit `2c458aa`): the earlier 3-stage pipeline's
longest stage (~263 ns @ GF180 ss) forced a 300 ns clock + an aggressive resize that
~2×'d the cell count; the micro-sequenced FSM executes **one fp32 op per cycle** (each
reg-to-reg path holds at most one fp32 add or multiply), bit-exact to
`sw/reference_model/vecu_softmax_ref.py` (added latency only — decode is
latency-tolerant, the consumer waits on `w_valid`).

## Run it

Requires Docker (~25 GB free disk); this run used LibreLane **3.0.5** on the
Sky130A PDK (ciel version `8afc8346`).

```sh
ciel enable --pdk-family sky130 8afc8346a57fe1ab7934ba5a6056ea8b43078e71
cd openlane/vecu_softmax
librelane --docker-no-tty --dockerized --pdk sky130A config.json
```

## Config

Based on `mate_pv_fp16/config.json` (same FP16 datapath family):

- `CLOCK_PERIOD` **105 ns (9.5 MHz)** — even after the micro-sequence rebalance, the
  binding reg-to-reg path is a **single fp32 op** (add or multiply, the same primitive
  that clocks `mate_pv_fp16` at ~76 ns pre-route). On Sky130 that op plus routing is
  ~98 ns at the slow `ss_100C_1v60` corner, so 105 ns closes with **+7.0 ns** of
  slack. The routed fp32 datapath degrades ~16 ns from pre-route (the exp-LUT +
  rescale + reciprocal logic routes denser than the P·V tile), which set the 105 ns
  close point. So Sky130 is fp32-op-bound like the sibling FP16 tiles — it does **not**
  clock faster than them (the GF180 "~½-length stages" note is relative to the old
  263 ns stage; the fp32 op itself is unchanged).
- `SYNTH_PARAMETERS: ["N=4"]` — score-buffer depth 4 for the physical proxy.
  Functional default is N=16 (deeper score buffer → larger, but the datapath / fmax
  are unchanged; N only sizes the row buffer).
- `FP_CORE_UTIL 45` / `PL_TARGET_DENSITY_PCT 55`, `DESIGN_REPAIR_MAX_CAP_PCT 60` +
  `RUN_POST_GRT_DESIGN_REPAIR` — the FP16 family recipe; drives **Max-Cap to zero**.
- `PL_/GRT_RESIZER_HOLD_SLACK_MARGIN 1.0` — this block is ~4× the cell count of the
  MAC tiles (50.7 k cells), and at the family's default 0.4 ns hold margin the signoff
  STA left 17 hold paths marginally negative (−0.55 ns worst); 1.0 ns drives every
  hold path positive (worst **+0.75 ns**) while setup keeps +7 ns of headroom.

`src/vecu_softmax.sv` is the block top (kept in sync with `rtl/vecu_softmax.sv`).

## Sign-off — Sky130A (N=4 proxy)

**All six physical checks are zero, multi-corner across all 9 IPVT corners**
(`{min,nom,max} × {tt_025C_1v80, ss_100C_1v60, ff_n40C_1v95}`). Committed under
`results/` (`vecu_softmax.gds.gz` — gzipped, `gunzip` to open in KLayout/Magic, since
this block's flat GDS is ~65 MB — plus `vecu_softmax.png` render + `sky130_signoff_metrics.json`).

| check | value |
|---|---|
| setup violations | **0** (WNS 0, all 9 corners; worst slack +7.0 ns @ ss) |
| hold violations | **0** (WNS 0; worst slack +0.75 ns) |
| Magic DRC / KLayout DRC | **0 / 0** |
| LVS (Netgen, incl. device diff) | **0** |
| antenna | **0** |
| **max-cap** | **0** (all 9 corners) |

Also clean: power-grid IR = 0, XOR diff = 0. Reported honestly (not one of the six):
`max_slew = 2476`, `max_fanout = 13` — ss-corner residual only (the FP16 family's
register-tree transition item; setup/hold/DRC/LVS unaffected), same class as
`mate_pv_fp16` (734).

| metric | value | notes |
|---|---|---|
| **fmax** | 9.5 MHz (105 ns constrained) | OpenSTA confirms all paths meet at every corner; the ss reg-to-reg path is ~98 ns → ~10.2 MHz intrinsic. Single fp32 op/cycle — fp32-op-bound like the sibling FP16 tiles. |
| setup / hold WS | +7.0 / +0.75 ns | worst across all corners (ss / fast). Zero setup and hold violations. |
| die area | 726,020 µm² (0.726 mm²) | floorplanner: std-cell area ÷ target util + IO margin. |
| std-cell area | 388,016 µm² (50,753 cells) | exp-LUT + fp32 add/mul + fp16 reciprocal + online-softmax FSM + score buffer + hold-repair buffers. |
| sequential (FF) | 858 | running max/sum registers + score buffer + reused intermediates + FSM. |
| core utilization | 55.6 % | |
| total power | ~10.2 mW | OpenSTA estimate @ 1.8 V / 9.5 MHz, default toggle (no workload VCD) — an estimate, not measured. |

**Derivation caveats:** (1) **N=4 proxy** (score-buffer depth 4); the functional block
is N=16 — a deeper score buffer, so more FFs/area, but the datapath and fmax are
unchanged (N sizes the buffer, not the critical path). (2) 130 nm Sky130A, 1.8 V,
timing at the slow corner. (3) power is at this block's own 105 ns clock under an
assumed toggle rate.

## Where it sits in the FP16 family

`vecu_softmax` is the largest and slowest of the three FP16 blocks in this repo, but
for the same underlying reason: all three are bounded by a single-cycle IEEE fp32
operation at the slow corner (~76 ns pre-route). `mate_qkt`/`mate_pv_fp16` are one
MAC-reduce recurrence; `vecu_softmax` adds the exp-LUT + rescale + reciprocal, which
routes denser and pushed the clean-close point to 105 ns. The honest takeaway: decode
softmax is ~10 MHz-class at 130 nm and, like the MACs, scales to ~60 MHz-class at
16 nm (see the `mate_pv_fp16` porting note; same ratio basis).
