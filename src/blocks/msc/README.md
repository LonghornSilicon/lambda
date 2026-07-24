# MSC — Memory Subsystem Controller (canonical MHC)

**Status: spec-only stub — no RTL yet.** Placeholder in the canonical block taxonomy so the full
chip is legible at a glance (`src/blocks/`). Spec lives in [`../../../arch.yml`](../../../arch.yml)
(block `MSC`); progress tracked in [`../../../docs/PROGRESS.md`](../../../docs/PROGRESS.md).

**Function (from arch.yml):** LPDDR5X x16 controller + 4-port SRAM crossbar (MatE/VecU/KVE/host) +
128-entry block table for vLLM-style KV paging + DMA descriptor engine + on-demand KVE per-channel
dequant trigger. Maps to the canonical Memory Hierarchy Controller (MHC, Block 4). Area ≈ 0.18 mm².

When RTL work begins, this block fills out to the canonical template (`rtl/ sw/ pdk/ docs/ research/`)
per [`../../../docs/REVISION_SYNC_SOP.md`](../../../docs/REVISION_SYNC_SOP.md) §5.
