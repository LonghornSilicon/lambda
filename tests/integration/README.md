# tests/integration/ — cross-block integration tests

Home for chip-level, cross-block tests (multi-block cosim, end-to-end decode traces). **Per-block
unit tests + reference-model parity live with their block** (`src/blocks/<block>/rtl/tb/`,
`src/blocks/<block>/sw/reference_model/test_*`); this directory is for tests that span blocks.

**Current cross-block cosim** lives in [`../../chip/verif/`](../../chip/verif/)
(`tb_chip_cosim.sv` + real-Qwen vectors + `Makefile`); run with `make -C chip/verif cosim`. That
harness currently vendors block RTL copies under `chip/verif/blocks/` — flagged for re-point to the
canonical `src/blocks/<block>/rtl/` (single-source-of-truth per
[`../../docs/REVISION_SYNC_SOP.md`](../../docs/REVISION_SYNC_SOP.md) §5.2).

As integration matures, chip-level test entry points and the full decode-loop trace harness land
here, referencing the canonical block RTL directly.
