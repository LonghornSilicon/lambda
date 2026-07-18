# Token Importance Unit — Sky130 Physical Sign-off (first pass)

LibreLane 3.0.5 / OpenROAD, sky130A HD, `token_importance_unit` (N_SLOTS=8,
SCORE_WIDTH=10). Config: `openlane/token_importance_unit/config.json`.

## Status: routes clean; residual max-transition on 2 nets

| Check | Count | |
|---|---|---|
| Setup violations | **0** | ✅ (WNS +14.0 ns @ 25 ns clk) |
| Hold violations | **0** | ✅ (WNS +0.18 ns) |
| Routing DRC (OpenROAD) | **0** | ✅ |
| Magic DRC | **0** | ✅ |
| LVS errors | **0** | ✅ |
| Antenna violations | **0** | ✅ |
| Max cap | 2 | ⚠ |
| Max fanout | 1 | ⚠ `clkbuf_0_clk` fanout 16 (limit 10) |
| Max transition (slew) | 22 | ⚠ two combinational nets @ ~1.63 ns vs 1.5 ns PDK limit |

GDS/DEF/LEF/LIB emitted (`runs/*/final/`), curated signoff metrics + layout render
in `openlane/token_importance_unit/results/`. Die 19321 µm², ~0.0012 mW, 2971 cells.

## The residual, honestly

Slew was driven from **748 → 25** total violations over the bring-up by: sizing the
accumulator to SCORE_WIDTH=10 (fewer FFs/fanout), not resetting the `score[]`
datapath (rst_n fanout 113 → ~11, killing the delay-buffer reset tree), and signing
off at the sky130 cells' real `max_transition` = 1.5 ns (the flow's 0.75 ns default
is half the PDK spec).

The last 22 slew violations are **two combinational nets** (the serialized-argmin
mux output / wide compare) at ~1.63 ns — only ~8% over the 1.5 ns limit, and only in
the slowest corner (ss_100C_1v60). The resizer will not buffer them further because
the fanout is structural: `scan_idx` and the 8:1 score read-mux spread one signal
across the datapath. The clock-root fanout (16) is the same class of issue in CTS.

**This is an RTL fix, not a config tweak:** register the argmin mux output (break the
comparator's combinational fanout into a pipelined compare), which drops the wide-mux
transition and lets CTS balance the clock. That is the next RTL iteration; the current
netlist is functionally correct (sim 20/20) and manufacturable (DRC/LVS/antenna clean).

## Reproduce

```sh
cd openlane/token_importance_unit
librelane --docker-no-tty --dockerized config.json
```
