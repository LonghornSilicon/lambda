# chip/ DECISIONS — cross-block integration

Append-only; never delete, mark superseded. Format: what · why · YYYY-MM-DD.
Chip-level (cross-block) decisions only; per-block calls live in `src/blocks/<block>/DECISIONS.md`;
chip-wide arch calls in the root `DECISIONS.md`.

- **Chipathon 2026 submission scope = KV-compression COPROCESSOR only** (KVE value compressor + TIU
  + precision_controller + SPI loader), NOT the full attention datapath · the full fp16 datapath
  does not fit the fixed 2051×2051 (4.21 mm²) workshop core — routing congestion >1.0 even at the
  minimal tile; the coproc closes comfortably (~0.7 mm² in the core) · 2026-07-23.
- **Coproc sized at Dh=2 / L=2** for wide routing margin (~30% util) · gives comfortable congestion
  headroom on the fixed core after larger tiles stalled routing · 2026-07-23.
- **Post-CTS resizer nulled in the coproc flow** · it stalls and timing is already met without it ·
  2026-07-23.
- **`chip/verif/` cosim vendors block RTL copies under `verif/blocks/{kve,tiu}`** (ACU pulled live
  from `src/blocks/acu/*/rtl/`) · a pragmatic bridge; flagged for re-point to single-source
  `src/blocks/<block>/rtl` — the copy-drift the monorepo is meant to eliminate · 2026-07-23.
- **`chip/` stays at top level through the 2026-07 reorg** (not moved under `src/blocks/`) · it is
  the cross-block *integration* block, not a functional block; the canonical template exempts it
  (README + DECISIONS only). See `docs/REORG_NOTES.md` · 2026-07-23.
