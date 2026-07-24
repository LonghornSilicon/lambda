# src/isa/ — chip-level ISA

Chip-level instruction/CSR definitions shared across blocks: the LSU opcode set, the CSR map, and
the unified programming interface. **Per-block ISAs live with their block** (`src/blocks/<block>/docs/isa/`)
and are versioned independently; this directory holds the *chip-level* aggregation.

**Versioning** (see [`../../docs/documentation_standard.md`](../../docs/documentation_standard.md)):
- Unified chip ISA: **`lh-isa-0.1`** — aggregates the per-block ISAs below. Documented in
  [`../../docs/compiler_programming_guide.md`](../../docs/compiler_programming_guide.md).
- Per-block ISAs (source of truth in each block):
  - `kv-isa-0.2` → [`../blocks/kve/docs/isa/`](../blocks/kve/docs/isa/)
  - `tiu-isa-0.1` → [`../blocks/tiu/docs/isa/`](../blocks/tiu/docs/isa/)
  - `pc-isa-0.2` → [`../blocks/acu/precision_controller/docs/isa/`](../blocks/acu/precision_controller/docs/isa/)
  - LSU ISA → [`../blocks/lsu/`](../blocks/lsu/) (spec-only until LSU RTL begins)

Chip-level headers (`csr_map.h`, `lsu.h`, `vecu_microcode.h`) land here as the LSU + CSR design
matures. An ISA change bumps the block's `<block>-isa-MAJOR.MINOR` and is recorded in a chip
revision ([`../../docs/REVISIONS.md`](../../docs/REVISIONS.md)).
