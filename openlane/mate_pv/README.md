# Sky130 OpenLane / LibreLane flow — `mate_pv`

End-to-end open-source RTL → GDSII flow for `mate_pv`, the INT8 P·V MAC tile
(INT32 accumulator), targeting SkyWater Sky130A. Same flow and tuning as
`../precision_controller` (the signed-off reference), so the two blocks reach
GDSII the same way.

> 130nm Sky130 proxy, used for 16nm estimates — Lambda targets TSMC 16nm.

## Run it

Requires Docker (~25 GB free disk) and `pip install librelane`.

```sh
cd openlane/mate_pv
librelane --docker-no-tty --dockerized config.json
```

Flag order matters: `--docker-no-tty` must precede `--dockerized`. First
invocation downloads the Sky130A PDK (~500 MB via Ciel) and the LibreLane
Docker image (~6 GB); subsequent runs reuse both caches.

## Config

Based on `precision_controller/config.json`, with the differences a real MAC (vs
a trivial comparator) needs to close the Sky130 SS corner:

- `CLOCK_PERIOD` **25 ns (40 MHz)** — same clock as the TIU block. The
  combinational `INT8-mult → INT32-add → FF` path is ~15-18 ns at the slow
  (`ss_100C_1v60`) corner, so 40 MHz is the honest signed-off point; tighter
  targets left OpenROAD sizing short (a ~−0.9 ns residual that did not scale).
  Closing a faster clock would take an RTL pipeline stage.
- `SYNTH_PARAMETERS: ["N=4"]` — synthesize **4 lanes** for the physical proxy
  run (same pattern as TIU's `N_SLOTS=4`). This halves `a_data`'s fanout tree,
  which was driving the Max Cap violations. The functional RTL default is N=8;
  the bit-exact sim and the `expected-ff-count` synth gate use N=8 (513 FFs).
- `IO_DELAY_CONSTRAINT: 2` (vs 5) — the inputs feed straight into the arithmetic,
  so a 5 ns input budget was eating a third of the cycle.

`src/mate_pv.sv` is the block top (kept in sync with `rtl/mate_pv.sv`).

## Sign-off — clean at 40 MHz, Sky130A

`results/` holds the committed sign-off (all six physical checks **zero**):

| metric | value |
|---|---|
| setup / hold violations | 0 / 0 (WS +6.8 / +0.21 ns @ ss) |
| DRC / LVS / antenna / Max-Cap | 0 / 0 / 0 / 0 |
| die area | 301.76 × 312.48 µm |
| std cells / sequential | 5726 / 257 (N=4) |
| core utilization | 59.2 % |
| total power | ~8.8 mW |

`results/mate_pv.gds` + `results/mate_pv.png` (render) +
`results/sky130_40MHz_signoff_metrics.json`. `runs/` is gitignored. The
`openlane-sky130` CI gate re-runs this config and asserts the six checks are zero,
same as every other block.
