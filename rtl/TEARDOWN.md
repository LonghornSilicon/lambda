# RTL teardown manifest — TurboQuant+ → ChannelQuant

Tracks the datapath conversion of the KVE block from the TurboQuant+ codec to
ChannelQuant. Full plan: [`../findings/channelquant_block_revamp.md`](../findings/channelquant_block_revamp.md).
Algorithm contract + golden vectors **landed 2026-06-22** (channelquant commit
`08d5287`), vendored hermetically at
[`tb/testvectors/channelquant/`](tb/testvectors/channelquant/README.md) — the
3-way parity dependency is unblocked. (Upstream source of truth:
`../../channelquant/docs/HW_CONTRACT.md` + `../../channelquant/reference/testvectors/`.)

> **Simulator available locally.** iverilog/vvp 12.0 (the tool of record) +
> verilator, provisioned per-host into `<lhs>/.tools` — `. rtl/eda-env.sh` puts
> them on PATH (portable: conda-forge on x86_64, from-source on aarch64; see
> NOTES.md). All three testbenches are green: `make sim` 17/17, `make sim_cq`
> **9/9 bit-exact**, `make sim_realdata` PASS. The verified-build gate is cleared,
> and the ChannelQuant compute cores have replaced the TurboQuant+ codec. The
> remaining teardown work is the P2 streaming FSM/SRAM and P4 synth — each step
> confirmed by `make sim` + golden-vector parity before commit.

## Status legend
`[ ]` not started · `[~]` skeleton added (inert) · `[x]` done & verified

## Delete (TurboQuant+-only) — DONE
- [x] `rotation_unit.sv` — deleted (commit on master). Archived on legacy branch.
- [x] `qjl_unit.sv` — deleted. Archived on legacy branch.

## Repurpose / replace
- [x] `norm_unit.sv` → deleted; per-axis amax→scale is **`cq_scale_unit`** (cq_units.sv).
- [x] `quantizer.sv` — deleted; uniform signed INT4/INT8 round-half-even+clamp is
      **`cq_quant_unit`** (cq_units.sv). No centroid ROM.
- [x] `decompressor.sv` — deleted; `q*scale` (+ FP16 outlier passthrough) is
      **`cq_dequant_unit`**. No inverse-WHT / JL.
- [x] `packer.sv` — deleted; INT4 nibble lane is **`cq_pack2`** (INT8 = raw byte).
- [~] `sram_controller.sv` — unchanged shell; scale storage + residual-group buffer
      management is the streaming P2 work (behavioral cores exist; not yet streamed).
- [x] `kv_cache_engine.sv` (top) — CSR map swapped to ChannelQuant; codec files
      removed from `RTL_SRC`. (Datapath is still the passthrough store as on the
      predecessor; streaming the cq cores through the FSM is P2.)

## Add (new ChannelQuant blocks)
- [x] datapath compute cores `cq_units.sv` (+ `cq_fp_pkg.sv`) — scale/quant/dequant/
      pack, **bit-exact vs golden vectors** (tb_channelquant.sv, all 9, all tiers).
- [x] `amax_unit.sv` — **implemented + verified** (P2). Synthesizable per-axis amax
      (value per-token / key per-channel over a group, partial finals); no `real` —
      max|fp16| is an unsigned max on the 15-bit magnitude field. Its amax → the
      proven `cq_scale_unit` reproduces the golden scales bit-exactly, all 9 vectors
      (`make sim_amax`, tb_amax_unit.sv).
- [x] `cq_value_path.sv` — **implemented + verified** (P2, integration-first). The
      reusable per-token VALUE datapath: amax_unit → cq_scale_unit → D× cq_quant_unit
      → pack (compress) + D× cq_dequant_unit (decompress). Streams whole value
      tensors bit-exact vs golden `val_scales` + `val_payload` + `expected_v_hat`,
      all 9 vectors (`make sim_vpath`, tb_value_path.sv). The top will instantiate it.
- [x] `residual_buffer.sv` / `scale_bank.sv` — **implemented** (P2). Real modules:
      residual_buffer = indexed fp16 group hold (G tokens); scale_bank = D-wide
      parallel per-channel scale register bank. Instantiated by cq_key_path.
- [x] `cq_key_path.sv` — **implemented + verified** (P2). Per-channel grouped KEY
      datapath with the group FSM (COLLECT→SCALE→EMIT→DONE): residual_buffer +
      amax_unit(key) + D× cq_scale_unit + scale_bank + D× cq_quant_unit + INT4 pack
      (with the outlier-mask keep/exclude), + D× cq_dequant_unit decompress. Streams
      whole key tensors bit-exact vs golden `key_scales` + `key_payload` +
      `expected_k_hat` + `sidecar`, full g=G and partial g<G groups, CQ-4/CQ-4+
      (`make sim_kpath`, tb_key_path.sv). Outlier channels excluded from INT4; the
      FP16 sidecar identity is the top's job. The top will instantiate it.
- [ ] outlier-mask ROM — static per-layer top-k key-channel indices (CQ-4+). The
      mask format is exercised by the parity TB; the ROM load IF is P2.

## CSR / ISA changes (top-level + docs/isa) — DONE
- [x] REMOVED `INFO_PQ_BITS`, `INFO_QJL_BITS`.
- [x] ADDED `INFO_TIER` (0=CQ-8,1=CQ-4,2=CQ-4+), `INFO_GROUP` (G), `INFO_OUTLIER_K`,
      `INFO_SCALE_DEPTH` (=D), `INFO_RESID_DEPTH` (=G). `INFO_DIM` already exposes D.
- [x] BUMPED `INFO_VERSION` → v0.2.0.0 (incompatible codec — ISA major).
- [ ] outlier-mask load interface — P2 (with the streaming key path).

## Build / CI
- [x] `RTL_SRC` = top + sram + `cq_units.sv`; deleted codec removed. `make sim`
      green (17/17), `make sim_cq` green (9/9 bit-exact), `make sim_realdata` green.
- [x] `genus.tcl` / `synth.ys` file lists + notes updated (cores are behavioral —
      synthesizable fp16 lowering is P4; OpenLane top/IO unchanged for the shell).
- [x] Update expected FF-count assertion (CI gate 3): 5575 → **19559**. The revamped
      top synthesizes cleanly (Yosys, 0 CHECK problems); the jump is the transitional
      passthrough store holding a raw fp16 vector (SRAM_WIDTH = D·COORD_WIDTH = 1024
      vs the old 288-bit compressed word). P2 (compressed streaming store) will
      shrink it — revisit the gate then. (This gate had been red on master since the
      top-swap commit; now green.)

## Verification (golden vectors landed; SV simulator now local — see eda-env.sh)
- [x] SV parity vs the Python reference: **all 9 golden vectors bit-exact** (scales,
      packed payload, and reconstructed K/V_hat), CQ-8/CQ-4/CQ-4+, D∈{64,128}, full
      and partial key groups, CQ-4+ outlier lane. `make sim_cq`.
- [x] 3-way Python ↔ C++ ↔ SV: **all three legs bit-exact** on the 9 golden vectors.
      C++ leg = `sw/reference_model/channelquant_ref.{hpp,cpp}` (1:1 port of the SV
      cores), checked by `test_channelquant_ref.cpp` — `make -C sw/reference_model
      test-cq`. Python verified upstream; SV via `make sim_cq`.
- [ ] `tb_realdata.sv`: captured Qwen2 K/V trace, reconstructed rMSE within tol.
- [ ] Synth (Sky130 → 16FFC); compare area/Fmax vs the TurboQuant+ baseline on
      `legacy/turboquant-plus` (expect smaller — no WHT, no JL).
