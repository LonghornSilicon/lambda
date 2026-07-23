# KVE handoff — grouped per-channel key-path integration

> **✅ DONE (master `4f92c9e`, 2026-07-03).** The grouped per-channel-INT4 key path
> is serialized, wired into the top, and signed off — **all CI gates green**
> (1 functional, 3 synth, 4 formal, 5 reference, 6 OpenLane). `make sim_top` is
> bit-exact for per-token INT4 V and grouped CQ-4+ keys; `make sim_kpath` 6/6. See
> the 2026-07-03 NOTES.md entry for the full record. The plan below is retained for
> provenance. Only open follow-up: **partial-group flush (g<G)** — datapath supports
> it; the top's stream framing auto-flushes at full G only.


**Status at handoff (master `3537727`, 2026-07-02): CI fully green.** Gates green:
1 functional, 3 synth (FF=8210), 4 formal (RTL≡netlist), 5 reference, 6 OpenLane
Sky130 sign-off. The **value path** (per-token INT4/INT8 V + per-token CQ-8 K) is
integrated into the top, synthesizable (no `real`), bit-exact, and physically
signed off. This doc is the plan to finish the revamp: wire the **grouped
per-channel-INT4 key path** (CQ-4 / CQ-4+) + outlier isolation into the top.

This work is **CPU-only** (iverilog / yosys / OpenLane CI) — no GPU needed.

---

## The goal

Route the KEY stream through `cq_key_path` (per-channel INT4, grouped by G, + static
outlier-channel isolation with an fp16 sidecar) instead of the current per-token
CQ-8 key handling, and get all gates green again. `cq_key_path` is already built and
**golden-verified standalone** (`make sim_kpath`, 6/6 bit-exact) — the work is
*serialization + top integration + re-verification*, not new algorithm.

---

## Step-by-step plan

1. **Serialize `cq_key_path`** (rtl/cq_key_path.sv). It currently instantiates **D
   parallel** combinational quant units (`cq_quant_comb`, generate loop `g_quant`).
   Replace them with **one shared sequential `cq_quant_unit_syn`** walked across the
   D channels — mirror `cq_value_path.sv` exactly (its `S_QSTART`/`S_QWAIT` states +
   `q_start`/`q_done` handshake + shift-register code assembly). A single parallel
   combinational divide cone is what made OpenLane's resizer hang; the value path is
   the proven template. This is the biggest sub-task.

2. **Wire `cq_key_path` into the top** (rtl/kv_cache_engine.sv). The AXIS input
   carries K vs V on `s_axis_kv_tuser` (0/1). Today both go through the value path.
   Route keys → `cq_key_path` (accumulate a group of G tokens → per-channel scales →
   grouped INT4 payload + outlier sidecar), values → `cq_value_path` (unchanged).

3. **Outlier-mask ROM.** Instantiate a static ROM from the vendored masks
   (`rtl/tb/testvectors/channelquant/masks/`, k=2 per contract v0.2) feeding
   `cq_key_path`'s `outlier_mask`. Lane is bypassable at D=128 (contract note).

4. **CQ-4+ fp16 sidecar.** Store the top-k outlier key channels as fp16 alongside
   the INT4 payload; reconstruct on decompress. `cq_key_path` already emits the
   sidecar (verified in `sim_kpath`) — the top must store/replay it.

5. **SRAM layout.** Keys need {D per-channel fp16 scales} + grouped INT4 payload +
   sidecar, vs the value path's {1 scale + packed payload}. Recompute the SRAM word
   layout / `SRAM_WIDTH` and the AXIL INFO regs (SCALE_DEPTH, RESID_DEPTH already
   exist). Keep the fp32 decompress bus (contract §1).

6. **Re-verify** (see checklist below).

---

## Key files and their roles

| File | Role |
|---|---|
| `cq_value_path.sv` | **The template.** Serialized value path: FSM `S_IDLE→S_WAIT→S_QSTART→S_QWAIT(×D)→S_EMIT`, shared sequential quant, shift-register `out_codes`, static `out_pay` pack in `S_EMIT`. Copy this structure. |
| `cq_key_path.sv` | Parallel key path to be serialized. FSM `S_COLLECT→S_SCALE→S_EMIT→S_DONE`. Uses `residual_buffer` + `scale_bank` + `amax_unit`(key mode) + **D× `cq_quant_comb`** (→ replace with 1 shared `cq_quant_unit_syn`). |
| `cq_units_syn.sv` | `cq_quant_unit_syn` (SEQUENTIAL, `.clk/.rst_n/.start/.done`), `cq_scale_unit_syn`, `cq_dequant_unit_syn`, and `cq_quant_comb` (combinational — delete its use once the key path is serialized). |
| `amax_unit.sv` | Value mode = per-token amax via **balanced max-tree**; key mode = per-channel running max (used by key path). Both verified. |
| `scale_bank.sv`, `residual_buffer.sv` | Key-path support (freeze group scales / buffer group tokens). |
| `kv_cache_engine.sv` | Top FSM + SRAM + AXIL/AXIS. Currently routes K+V through the value path. |
| `tb/tb_key_path.sv` | `sim_kpath` — grouped-key golden parity. Update to the sequential handshake after serializing. |
| `tb/tb_top_stream.sv` | `sim_top` — currently CQ-8 K+V end-to-end. Extend to grouped keys. |

---

## Patterns to reuse (hard-won this session)

- **Sequential divider handshake** — `cq_quant_unit_syn` takes ~24 cycles/channel
  (pulse `start` with operands valid; wait `done`). See `cq_value_path.sv` S_QSTART/
  S_QWAIT for how to drive one shared unit across D channels.
- **Wide reductions → balanced tree** via `generate` + **wire arrays** (see
  `amax_unit.sv` `ml[...]`), never a linear scan (was 953 levels) and never a
  procedural `reg` array in `always @*`.

## Gotchas that WILL bite you

- **mem2reg "conflicting-constant" = LibreLane's "2 Yosys check errors."** A
  combinational **`reg [..] arr [0:N]` written in `always @*`** gets mem2reg'd into
  DFFs with constant-conflicting drivers. Your local `synth -flatten; check` says 0
  (post-opt), but LibreLane checks *pre-opt* and fails. Reproduce locally with
  `yosys -p "read_verilog -sv <files>; hierarchy -top <mod>; proc; check"` and count
  `conflicting with a constant` — must be **0**. Fix: inline scratch reg or
  generate+wire arrays (both used this session).
- **OpenLane has a hard 30-min job cap** in the SHARED `LonghornSilicon/.github`
  `block-ci.yml` — you can't override it from this repo. Full-size D=64/depth-16 P&R
  overruns it (flop-based SRAM, no Sky130 macro). Gate 6 passes as a **proxy** via
  `pdk/sky130/openlane/kv_cache_engine/config.json` → `"SYNTH_PARAMETERS":
  ["SRAM_DEPTH=4","VECTOR_DIM=32"]` (LibreLane-only chparam) + `CLOCK_PERIOD: 100`.
  **Adding the key path grows the design → you may need to shrink the proxy params
  further** to keep routing under the cap (watch DetailedRouting time in the log).
- **FF-count gate**: CI apt-yosys 0.33 ≠ local conda 0.65 by a few FFs. After any RTL
  change, read the actual from the gate-3 log (sum the `$_*DFF*` cell counts) and set
  `expected-ff-count` in `.github/workflows/ci.yml`.
- **New RTL files need TWO registrations**: `extra-rtl-sources` in `ci.yml` AND a
  force-added symlink in `pdk/sky130/openlane/kv_cache_engine/src/` (the glob is gitignored).
- **Directed TBs** (`tb_kv_cache_engine`, `tb_realdata`) need long post-stream waits
  (`TOKWAIT ≈ 26·D`) because compress is now multi-cycle — don't use short repeats.
- **LibreLane** installs via `pip install librelane` (v3.0.4). Introspect any flow
  step's config vars with:
  `python3.10 -c "from librelane.steps import Step; c=Step.factory.get('OpenROAD.ResizerTimingPostCTS'); [print(v.name, v.default) for v in c.config_vars]"`

## Toolchain on a fresh machine

iverilog 12.0 + yosys are needed (CPU-only). The `.tools` micromamba env is
per-host — set it up on the new box: `. rtl/eda-env.sh` (or see the EDA-toolchain
memory / repo notes). CI needs nothing extra — it installs iverilog/yosys/librelane
itself. Git author for this repo: `themoddedcube <themoddedcube@gmail.com>`; commit
+ push milestones straight to `master`.

## Verification checklist (all must be green before pushing)

- [ ] `make sim_kpath` bit-exact (after serializing — update the TB to the handshake)
- [ ] `make sim_top` extended to grouped keys, bit-exact
- [ ] `make sim sim_realdata sim_vpath sim_amax sim_syn` still pass
- [ ] `proc; check` → **0** `conflicting with a constant` on the full top
- [ ] 0 latches, no `real`, `expected-ff-count` updated to CI-0.33 value
- [ ] CI: gates 1/3/4/5 green; gate 6 (shrink proxy params if DetailedRouting nears
      the cap)

---

*Full context in the two-lane project memory and `rtl/NOTES.md`. The ChannelQuant
lane hand-off report (bit-exact numbers + scope boundaries) was delivered
2026-07-02.*
