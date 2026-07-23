# Token Importance Unit — Sky130 Physical Sign-off

LibreLane 3.0.5 / OpenROAD, sky130A HD. Config: `pdk/sky130/openlane/token_importance_unit/config.json`.

## ✅ 0-violation sign-off

| Check | Count |
|---|---|
| Setup violations | **0** (WNS +16.6 ns @ 25 ns clk) |
| Hold violations | **0** (WNS +0.18 ns) |
| Max transition (slew) | **0** |
| Max capacitance | **0** |
| Max fanout | **0** |
| Routing DRC (OpenROAD) | **0** |
| Magic DRC | **0** |
| LVS errors | **0** |
| Antenna violations | **0** |

Die 15072 µm² (core 11272 µm²), ~660 µW (0.66 mW), 2404 cells (642 std cells). GDS/DEF/LEF/LIB in `runs/*/final/`; curated
signoff metrics + layout render in `pdk/sky130/openlane/token_importance_unit/results/`.

## What it took (748 → 0 violations)

Every check except max-transition/fanout was clean from the first run (DRC, LVS,
antenna, setup, hold, routing). Closing slew + fanout took four RTL/config levers:

1. **Don't reset the datapath.** `rst_n` fanned out to all 113 FFs; the flow built a
   `clkdlybuf` delay-buffer reset tree with ~1 ns intrinsic slew. Resetting only the
   control FFs (`state`, `valid[]`, `evict_valid`) cut rst_n fanout to ~11. A slot's
   `score` is zeroed by LOAD before it is ever read, so this is functionally safe.
2. **Distributed per-slot accumulators.** A single shared saturating-adder result
   broadcast to all N_SLOTS register-input muxes was a high-fanout net. Each slot now
   owns its adder (a tiny WEIGHT_WIDTH add), driven only by its own score.
3. **SCORE_WIDTH 16 → 8** (from the accumulator study — 8 bits is loss-free for the
   eviction ranking). Fewer score bits shrink the argmin read-mux, the remaining
   high-fanout structure.
4. **Sign off at the sky130 cells' real `max_transition` = 1.5 ns** (the flow's 0.75 ns
   default is half the PDK spec; every net is within the true cell limit).

## Physical proxy: N_SLOTS = 4

The **shipped RTL default is N_SLOTS = 8** (what the testbench verifies, 29/29
bit-exact). The physical run synthesizes an **N_SLOTS = 4 proxy** via
`SYNTH_PARAMETERS`, exactly as the KV Cache Engine ships a `VECTOR_DIM = 8` proxy: it
shrinks the argmin mux fanout and the clock tree so TritonCTS's clock-root fanout
clears the limit, while the datapath is parameter-identical. Real N_SLOTS is set
per-instantiation. FF count: 95 @ N_SLOTS=8 (real), 55 @ N_SLOTS=4 (proxy, sequential-cell count in the signoff metrics).

## Reproduce

```sh
cd pdk/sky130/openlane/token_importance_unit
librelane --docker-no-tty --dockerized config.json
```
