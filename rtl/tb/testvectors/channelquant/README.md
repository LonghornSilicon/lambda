# ChannelQuant → KVCE handoff: contract + golden vectors

**Vendored from the `channelquant` lane @ commit `08d5287` (see `SOURCE_COMMIT`).**
This is the verified algorithm handoff that unblocks KVCE verification
(`findings/channelquant_block_revamp.md` §1, P3 — 3-way parity). The
`channelquant` repo remains the source of truth; this is a pinned copy so the
parity harness is hermetic. Re-sync only on a contract/vector version bump.

## What landed

| Artifact | What it is |
|---|---|
| `HW_CONTRACT.md` | Algorithm→silicon interface contract: exact quant rule (round-half-to-even, clamp INT4 [−8,7] / INT8 [−128,127]), `EPS=2^-14`, fp16 scales, per-tier bit/packing layout (§5), group-flush semantics (§3), static outlier-mask format (§4), parity acceptance (§8). **Implement against this; if it is silent, ask the channelquant lane — do not invent.** |
| `*.npz` (9) | Golden vectors: `input_K/V`, expected packed payload (`key/val_payload`, `*_scales`, `sidecar`, `outlier_*`), expected reconstructed `expected_K/V_hat`. Source of truth. |
| `manifest.json` | Per-vector tier/recipe/D/T/G, group byte offsets, payload SHAs, rms. |
| `hex/<vector>/*.hex` | The same data as flat `$readmemh`-loadable images (one value/line, MSB-first; `u8`=2, `f16`=4 raw half, `f32`=8 raw single hex). See `hex/INDEX.md`, checksums in `hex/SHA256SUMS`. |

## Coverage (contract §8 — all required cases present)
- Tiers: **CQ-8 / CQ-4 / CQ-4+** (k=2, static mask applied).
- Key groups: **full** (`d64_T128_G64`, 2×g=64), **partial** (`d64_T70_G64`, g=6;
  `d128_T100_G128`, g=100).
- Head dim: **D ∈ {64, 128}**.
- Both **compress** (`*_payload` + scales + sidecar) and **decompress**
  (`expected_*_hat`).

## Verified before handoff (channelquant Phase 1, commit `08d5287`)
- Reference reproduces the c17 evidence **bit-exactly** (max |Δ|=0.000 over 6
  variants × Qwen2-{0.5B,1.5B,7B}, HellaSwag acc_norm n=250). The INT4 collapse
  and the per-channel recovery are real.
- `torch` fake-quant == `numpy` pack→unpack **bit-exact** for every tier (7/7
  reference unit tests); generator self-asserts `numpy==torch` keys per vector.
- The `.hex` images round-trip bit-exactly back to the `.npz`.

## How to consume for parity (P3)
1. Python/C++ reference (`sw/reference_model/`) reads the `.npz` directly.
2. SV testbench (`rtl/tb/`) `$readmemh`s `hex/<vector>/*.hex`.
3. Bit-exact match required on **both the packed byte stream and the
   reconstructed tensors**, all three tiers (contract §8). Byte-stream-only or
   tensor-only matches are non-conformant.

## Open items to confirm with the channelquant lane (do not guess)
- **Decompress bus format.** `expected_K/V_hat` are exported as exact **fp32**
  (`int × fp16_scale`, lossless). If the KVCE decompress bus carries fp16, compare
  after casting to fp16 — and pin which the RTL emits (contract §1 fixes scale
  format as fp16 but does not yet fix the product format). Flagged in
  `HW_CONTRACT.md` "Open items".
- **Final `G`.** Vectors use G=64 (D=64) / G=128 (D=128). The shipped `G` comes
  from the channelquant Phase-2 group-size Pareto — re-sync if it changes.
- **CQ-4+ at scale.** CQ-4+ = outlier on **K only** + per-token V. At 7B its
  HellaSwag point estimate sits −0.032 below the both-sides outlier row (inside
  overlapping n=250 CIs); the headline n≥1000 run is channelquant Phase 3. Does
  not affect bit-exact parity — noted so the joint paper's accuracy column is honest.
