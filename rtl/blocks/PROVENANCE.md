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
| `vecu_softmax.sv`                   | `architecture/rtl/blocks/acu/vecu_softmax.sv` (vendored, **pipelined**) | `attention-compute-unit` `rtl` @ `f36daa2` (architecture `rtl` @ `5aa6298`) | REAL — decode online-softmax, 3-stage pipeline (passed RTL cosim) |

`mate_qkt.sv` / `vecu_softmax.sv` were copied from the byte-identical vendored
copies in the `architecture` repo (`rtl/blocks/acu/`), which themselves originate
on `attention-compute-unit` `rtl` at the commits above. Both are synthesis-clean
(no `real`/`$fscanf`); `librelane/{mate_qkt,vecu_softmax}.yaml` harden them on
GF180 at N=8. **Both are now VERBATIM** — no local edits.

`vecu_softmax.sv` was **re-copied 2026-07-21** as the **pipelined** version
(`attention-compute-unit` `rtl` `f36daa2`): the two-serial-fp32-multiply exp
chain that missed ss-corner setup is now split across 3 feed-forward register
stages (+358 FFs, throughput unchanged at 1 score/cycle, bit-exact; the +3-cycle
latency is data-independent and transparent to the result handshake). This also
**removed the earlier `gi_unused` synth-compat patch** — the pipelining rewrite
no longer has the module-scope loop counter, so the source is now verbatim and
synthesizes cleanly with the default yosys frontend (no `USE_SLANG`, no edits).

Note: the `kve/*.sv` set is the full non-testbench RTL of the KV-cache engine
(the top `kv_cache_engine.sv` plus its ChannelQuant / WHT / SRAM-controller
submodules). The LibreLane config `librelane/kve.yaml` hardens the top with the
gate-proxy parameters.

`kve/kv_sram.sv` + the updated `kve/sram_controller.sv` are re-synced from the
kv-cache-engine repo `rtl` refactor that puts the KV-store array behind a
swappable `kv_sram` memory interface (behavioral default). **`kve_gf180_sram/`**
is **chipathon-authored (NOT vendored):** `kv_sram.sv` there is the GF180 view of
the same module — it tiles the real `gf180mcu_fd_ip_sram__sram512x8m8wm1` hard
macro to the identical interface; `*__bb.v` is the macro blackbox stub for
lint/synth. `maglef_drc/…mag` is a **local copy of the PDK maglef with one
sub-min-width Metal3 vendor-abstract pin widened to min-width**, used only as the
`MAGIC_DRC_MAGLEFS` DRC-blackbox view (the vendor GDS is signed-off clean; not
used for LVS/connectivity). `librelane/kve_store_gf180.yaml` hardens it with real
SRAM macros placed — clean 6-check signoff (DRC=0, LVS=0); see
`docs/gf180_gls_report.md` §4.
