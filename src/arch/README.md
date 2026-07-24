# src/arch/ — architecture spec pointer

The canonical **architecture spec** and **chip-level ISA** live in the standalone
[`LonghornSilicon/lambda-arch`](https://github.com/LonghornSilicon/lambda-arch) repo (the spec /
"why" layer: `arch.yml` source, `STATUS`, `floorplan`, `dataflow_walkthrough`, the paper, and HLS
exploration). This monorepo is the **implementation / "how"** layer.

- **Chip-level ISA** (LSU opcodes, CSR map, the unified `lh-isa`) — owned by `lambda-arch`.
- **Per-block ISA** — stays with each block: `src/blocks/<block>/docs/isa/` (`kv-isa`, `tiu-isa`,
  `pc-isa`), versioned independently and recorded in chip revisions (`docs/REVISIONS.md`).
- **Machine-readable arch** — `arch.yml` at this repo root is the implementation-facing copy;
  `lambda-arch` is its canonical narrative home.

`src/isa/` was folded into `src/arch/` in the 2026-07 reorg (`docs/REORG_NOTES.md`) so there is one
spec home, not two. When an ISA changes, bump the block's `<block>-isa-MAJOR.MINOR` and log it in the
next revision.
