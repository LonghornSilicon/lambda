# RTL provenance — `rtl/blocks/`

The Verilog in this directory is **copied, unmodified**, from the LonghornSilicon
sibling design repositories. This chipathon submission repo is the integration /
tapeout harness; those repos remain the source of truth for each block's RTL,
testbenches, and per-block sign-off. Re-sync from upstream when the block RTL
changes — do not edit these copies in place (fix upstream, re-copy).

Copied on 2026-07-21 from (all read-only; not modified by this repo):

| file(s) in `rtl/blocks/`            | source repo / path                                             | src branch @ commit        | RTL status                          |
|-------------------------------------|----------------------------------------------------------------|----------------------------|-------------------------------------|
| `kve/*.sv` (16 files)               | `kv-cache-engine/rtl/*.sv` (non-tb)                             | `rtl` @ `dacebbd`          | REAL — Sky130-signed (ChannelQuant KV codec) |
| `token_importance_unit.sv`          | `token-importance-unit/rtl/token_importance_unit.sv`           | `rtl` @ `6c24975`          | REAL — Sky130-signed (H2O eviction) |
| `precision_controller.sv`           | `adaptive-precision-attention/rtl/precision_controller.sv`     | `pdk-asap7` @ `e501d90`    | REAL — Sky130-signed                |
| `mate_pv.sv`                        | `adaptive-precision-attention/rtl/mate_pv.sv`                  | `pdk-asap7` @ `e501d90`    | REAL — Sky130-signed (INT8 P·V)     |
| `mate_pv_fp16.sv`                   | `adaptive-precision-attention/rtl/mate_pv_fp16.sv`             | `pdk-asap7` @ `e501d90`    | REAL — Sky130-signed (FP16 P·V)     |
| `mate_qkt.sv`                       | `architecture/rtl/blocks/acu/mate_qkt.sv` (vendored)          | `attention-compute-unit` `rtl` @ `93e9960` | REAL — decode Q·Kᵀ scoring (passed RTL cosim) |
| `vecu_softmax.sv`                   | `architecture/rtl/blocks/acu/vecu_softmax.sv` (vendored)      | `attention-compute-unit` `rtl` @ `4a30d93` | REAL — decode online-softmax (passed RTL cosim) |

`mate_qkt.sv` / `vecu_softmax.sv` were copied on 2026-07-21 from the byte-identical
vendored copies in the `architecture` repo (`rtl/blocks/acu/`), which themselves
originate on `attention-compute-unit` `rtl` at the commits above. Both are
synthesis-clean (no `real`/`$fscanf`); `librelane/{mate_qkt,vecu_softmax}.yaml`
harden them on GF180 at N=8. `mate_qkt.sv` is verbatim.

**`vecu_softmax.sv` — one synth-compat patch (documented divergence):** the
vendored source declares the fp16-subnormal-normalize loop counter `gi_unused`
at **module scope** but uses it **only inside** `function automatic fp16_to_fp32`.
A module-scope reg written from an automatic function makes yosys **latch-infer**
it → 320 "multiple conflicting drivers" check errors that abort GF180 synthesis.
The patch moves the `integer gi_unused;` declaration **into that function** (so it
is per-call, as an automatic-function local). This is **behaviorally and
simulation identical** (the counter is dead outside the loop) and is the **only**
divergence from the vendored source — flagged inline in `vecu_softmax.sv` with a
`SYNTH-COMPAT PATCH` comment. Fix belongs upstream on `attention-compute-unit`.

Note: the `kve/*.sv` set is the full non-testbench RTL of the KV-cache engine
(the top `kv_cache_engine.sv` plus its ChannelQuant / WHT / SRAM-controller
submodules). The LibreLane config `librelane/kve.yaml` hardens the top with the
gate-proxy parameters.
