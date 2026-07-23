# Build guide — Lambda ACU (GF180 multi-macro → workshop padring)

This is the operational counterpart to the top-level `README.md`. It covers
(1) verifying the harness, (2) hardening each block as a standalone GF180
LibreLane macro, and (3) integrating into the Chipathon 2026 workshop padring.

Everything runs inside the chipathon toolchain container
(`hpretl/iic-osic-tools` / `iic-osic-tools`, which ships LibreLane, yosys,
OpenROAD, Magic, Netgen, KLayout, iverilog, cocotb) or the padring fork's
`nix-shell`.

---

## 0. Verify the harness (cocotb smoke test)

Elaborates the workshop-slot `chip_core` (via `tb/chip_core_wrap.sv`, which pins
the pad widths to `SLOT_WORKSHOP` = 1/20/60) and drives one SPI `START` frame end
to end through the serial loader.

```bash
cd tb
make test-smoke      # 2 tests: reset/pad-dir sanity + SPI START handshake
make clean
```

Expected: `TESTS=2 PASS=2`. Requires `cocotb` + `icarus-verilog` on `PATH`.

---

## 1. Harden a block as a standalone GF180 macro (LibreLane Classic)

Each block owns its GF180 config **block-major** under
`<block>/pdk/gf180/librelane/<macro>.yaml`, pointing at that block's own `rtl/`.
The simplest path is `scripts/harden.sh <macro>` (it locates the config by macro
name anywhere under the worktree, mounts the whole repo, and runs it from the
config's dir). From the repo root, inside the container, the equivalent explicit
invocations are:

```bash
librelane acu/mate/pdk/gf180/librelane/mate_pv.yaml       # → runs/.../final/{gds,lef,lib,nl.v}
librelane acu/precision_controller/pdk/gf180/librelane/precision_controller.yaml
librelane tiu/pdk/gf180/librelane/token_importance_unit.yaml
librelane kve/pdk/gf180/librelane/kve.yaml
librelane kve/pdk/gf180/librelane/kve_store_gf180.yaml    # real gf180 SRAM store
librelane acu/mate/pdk/gf180/librelane/mate_pv_fp16.yaml
```

Each produces a reusable `GDS / LEF / Liberty / .nl.v` view set. Collect them
(e.g. `--save-views-to ../build/<macro>`) for the chip-top `MACROS:` merge and
for gate-level cocotb.

> **GF180 re-timing.** The `CLOCK_PERIOD` in every real yaml is **ported from the
> Sky130 signoff** and is a starting point only. GF180MCU (180 nm) is slower than
> Sky130 (130 nm); expect to relax the period until the Classic flow closes STA,
> then record the achieved frequency. This is the `TODO re-time for GF180` note
> in each config.

`mate_qkt.yaml` and `vecu_softmax.yaml` are **stubs** — their `VERILOG_FILES`
point at sources that do not exist yet. They will not run until the RTL lands.

## 2. Per-macro gate-level verification (cocotb, gf180 cells)

Follow the template's `test-*-gl` pattern (see `tb/timescale.v` and the
sscs-chipathon-2026 multi-macro `tb/Makefile`): compile `timescale.v` +
`primitives.v` + `gf180mcu_fd_sc_mcu7t5v0.v` + the macro's `*.nl.v` and re-run
the block's cocotb test against the synthesized netlist. Per-macro GL targets are
added to `tb/Makefile` as each macro is hardened.

## 3. Integrate into the workshop padring

Target fork: **`Mauricio-xx/chipathon-2026-gf180mcu-padring`** (workshop slot:
`NUM_INPUT_PADS=1`, `NUM_BIDIR_PADS=20`, `NUM_ANALOG_PADS=60`, die 2935×2935 µm).

```bash
git clone https://github.com/Mauricio-xx/chipathon-2026-gf180mcu-padring
cd chipathon-2026-gf180mcu-padring
```

Then:

1. **Swap the core, keep the ports.** Replace `src/chip_core.sv` with this repo's
   `rtl/chip_core.sv` (the pad port list is identical — copied verbatim from the
   fork). Add `rtl/lambda_acu.sv` and `rtl/spi_loader.sv` to the fork's
   `librelane/config.yaml` `VERILOG_FILES`.
2. **Merge the hardened macros.** Add each block's `final/` views to
   `librelane/config.yaml` under `MACROS:` (with placement) and add matching
   `PDN_MACRO_CONNECTIONS` entries; add the block-level `define_pdn_grid` blocks
   to `librelane/pdn_cfg.tcl`. (This mirrors what the multi-macro template
   notebook does for its counter+ALU macros.)
3. **Wire the datapath.** In `rtl/lambda_acu.sv`, instantiate the macros at their
   reserved `TODO macro:` sites, wire the SPI Q/K/V/OUT byte buffers to each
   macro's stream ports, and replace the placeholder sequencer with the real
   block-chain FSM.
4. **Build to GDS.**
   ```bash
   SLOT=workshop make librelane      # Chip flow + Magic DRC + Netgen LVS
   ```
   Runtime is a couple of hours for full signoff (per the fork README).

## Notes

- The block RTL each GF180 config points at lives in that block's own `rtl/`
  (`kve/rtl/`, `tiu/rtl/`, `acu/*/rtl/`); provenance of the chip-imported copies
  is in [`../PROVENANCE.md`](../PROVENANCE.md). Fix RTL in the block, not here.
- The Sky130 signoff configs the GF180 yamls were ported from live block-major in
  `<block>/pdk/sky130/openlane/<block>/config.json` — consult them for the knobs
  (hold-slack margins, transition/fanout constraints) each block needed to close.
