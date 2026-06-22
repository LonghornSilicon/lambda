# RTL teardown manifest — TurboQuant+ → ChannelQuant

Tracks the datapath conversion of the KVCE block from the TurboQuant+ codec to
ChannelQuant. Full plan: [`../findings/channelquant_block_revamp.md`](../findings/channelquant_block_revamp.md).
Algorithm contract + golden vectors **landed 2026-06-22** (channelquant commit
`08d5287`), vendored hermetically at
[`tb/testvectors/channelquant/`](tb/testvectors/channelquant/README.md) — the
3-way parity dependency is unblocked. (Upstream source of truth:
`../../channelquant/docs/HW_CONTRACT.md` + `../../channelquant/reference/testvectors/`.)

> **Simulator now available locally (2026-06-22).** iverilog/vvp 12.0 built from
> source + verilator via micromamba on this aarch64 host — `. rtl/eda-env.sh` puts
> them on PATH. Baseline `make sim` is green (14/14 on the TurboQuant+ TB), so the
> verified-build gate is **cleared**. Steps that change the elaborated design
> (deletions, top rewiring, `RTL_SRC` edits) may now proceed, each confirmed by
> `make sim` + golden-vector parity before commit. As of this line the datapath is
> still TurboQuant+ and the ChannelQuant blocks are still inert skeletons.

## Status legend
`[ ]` not started · `[~]` skeleton added (inert) · `[x]` done & verified

## Delete (TurboQuant+-only, remove during the verified rewire step)
- [ ] `rotation_unit.sv` — Walsh–Hadamard butterfly. No rotation in ChannelQuant.
- [ ] `qjl_unit.sv` — 1-bit JL sign projection. No residual sketch.

## Repurpose / replace
- [ ] `norm_unit.sv` → folded into **`amax_unit.sv`** (per-axis amax, not L2 norm).
- [ ] `quantizer.sv` — replace 3-bit Lloyd-Max nearest-centroid with uniform
      signed INT4/INT8 round+clamp (drops the centroid ROM).
- [ ] `decompressor.sv` — simplify to `q * scale` (+ re-insert FP16 outlier
      channels); remove inverse-WHT and JL reconstruction.
- [ ] `packer.sv` — re-lane to 4/8-bit + per-axis scale sidecar + FP16 outlier lane.
- [ ] `sram_controller.sv` — add scale storage + residual-group buffer management.
- [ ] `kv_cache_engine.sv` (top) — rewire datapath; new CSR fields (see below).

## Add (new ChannelQuant blocks)
- [~] `amax_unit.sv` — per-token (V) / per-channel (K) amax. Skeleton added.
- [~] `residual_buffer.sv` — FP16 hold for the in-flight key group. Skeleton added.
- [~] `scale_bank.sv` — per-channel K scales + per-token V scale FIFO. Skeleton added.
- [ ] outlier-mask ROM — static per-layer top-k key-channel indices (CQ-4+).

## CSR / ISA changes (top-level + docs/isa)
- [ ] REMOVE `INFO_PQ_BITS`, `INFO_QJL_BITS`.
- [ ] ADD `INFO_GROUP_SIZE` (G), `INFO_OUTLIER_K` (k; 0 ⇒ CQ-4), `INFO_SCALE_FMT`,
      `INFO_HEAD_DIM` (D, parameterized), `CFG_TIER` (0=CQ-8,1=CQ-4,2=CQ-4+).
- [ ] BUMP `INFO_VERSION` (incompatible codec — ISA major).
- [ ] ADD outlier-mask load interface.

## Build / CI
- [ ] Add the new modules to `RTL_SRC` (Makefile) only when they elaborate clean.
- [ ] Update `synth.ys`, OpenLane `config.json` top/IO if ports change.
- [ ] Update expected FF-count assertions (CI gate 3) after synth lands.

## Verification (golden vectors landed; gated only on an SV simulator on PATH)
- [ ] Unit-test each new block vs the Python reference (behavioral).
- [ ] 3-way Python ↔ C++ ↔ SV bit-exact parity, compress + decompress, all tiers.
- [ ] `tb_realdata.sv`: captured Qwen2 K/V trace, reconstructed rMSE within tol.
- [ ] Synth (Sky130 → 16FFC); compare area/Fmax vs the TurboQuant+ baseline on
      `legacy/turboquant-plus` (expect smaller — no WHT, no JL).
