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

Not yet copied (RTL does not exist upstream yet):

| block           | status                         |
|-----------------|--------------------------------|
| `mate_qkt`      | Phase 1 — RTL IN PROGRESS      |
| `vecu_softmax`  | Phase 2 — RTL NOT STARTED      |

Note: the `kve/*.sv` set is the full non-testbench RTL of the KV-cache engine
(the top `kv_cache_engine.sv` plus its ChannelQuant / WHT / SRAM-controller
submodules). The LibreLane config `librelane/kve.yaml` hardens the top with the
gate-proxy parameters.
