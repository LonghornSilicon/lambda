# DECISIONS — ACU (Attention Compute Unit)

Settled calls, append-only. Format: *what · why · date*. Do not re-litigate unless the premise
changed. Sub-block-specific decisions now live in `mate/`, `vecu/`, `precision_controller/DECISIONS.md`
(migrated there at import 2026-07-22, per the parked seed).

## ACU-wide
- **The ACU is a multi-block unit → block-major sub-folders** (`mate/`, `vecu/`,
  `precision_controller/`), each self-contained (`sw/ rtl/ pdk/ docs/ research/`) · matches the
  monorepo block-major decision and makes each sub-block a complete mirror · 2026-07-22.
- **Canonical RTL is the ACU sub-blocks, not the chipathon copies** · `chipathon-lambda-acu` kept
  hand-synced `.sv` copies (tracked in `PROVENANCE.md`); the monorepo keeps ONE copy per tile and
  `pdk/gf180` references it by path. At import all 23 copies were byte-identical to canonical · 2026-07-22.
- **The APA RL research is chip-wide, parked at `research/apa-precision-policy/`** · it birthed the
  whole ACU (entropy → ratio-gate discovery) but is not block RTL · 2026-07-22.
- **Sky130 = flagship dev/proof PDK; GF180 = 2026 tape-out PDK; ASAP7 = predictive bracket only** ·
  the SSCS Chipathon 2026 shuttle is GF180MCU · 2026-07-21.
- **Online-softmax citation = Milakov & Gimelshein 2018**, not FlashAttention-3 · the recurrence
  predates FA; FA added the tiling · 2026-07-21.
