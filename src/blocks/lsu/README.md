# LSU — Layer Sequencer

**Status: spec-only stub — no RTL yet** (assembly/ISA sketch only). Placeholder in the canonical
block taxonomy. Spec: [`../../../arch.yml`](../../../arch.yml) (block `LSU`); chip-level ISA headers
land in [`../../isa/`](../../isa/); progress: [`../../../docs/PROGRESS.md`](../../../docs/PROGRESS.md).

**Function (from arch.yml):** minimal in-order RISC that walks the pre-compiled model schedule —
dispatches MatE GEMMs, KVE compress/decompress, VecU softmax/RMSNorm/RoPE, DMA prefetches. 32-bit
fixed-width ISA, 32 instructions, 16×32b GPR, 4K microcode. Area ≈ 0.10 mm².

Fills out to the canonical block template when RTL work begins
(see [`../../../docs/REVISION_SYNC_SOP.md`](../../../docs/REVISION_SYNC_SOP.md) §5).
