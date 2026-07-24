# REVISIONS.md — Lambda chip revision manifest

Append-only ledger of **coordinated cross-block revisions**. Each entry decodes one `rev-RN[.5]`
git tag into human-readable terms: what baseline/assembly it is, and every block's exact state at
that pin. See `docs/REVISION_SYNC_SOP.md` for the model, triggers, and cut procedure.

**Rule:** the monorepo tag is the atomic source of truth (one tag pins all blocks). This file is its
legible decode. Never edit a past entry except to append a correction note. Integer baselines are
immutable; `.5` assembly entries may be re-cut as assembly advances.

Legend — sign-off: `signed-off` = GDS + all headline checks 0 (DRC/LVS/antenna/setup/hold);
`route-clean` = ASAP7/ORFS GDS, routing+antenna clean, **no Magic-DRC/LVS** (not full sign-off);
`config-only` = flow declared, not run; `prose-only` = claimed in a report, no committed metrics JSON.

---

## R1 — Proxy-PDK block baseline

- **Tag:** `rev-R1` — **cut** (annotated tag at `cb08d1b`, local; push with `git push origin rev-R1`) · **Kind:** baseline (frozen)
- **Date:** 2026-07-23 · **Monorepo anchor:** `cb08d1b` (2026-07-23 23:50Z) · **Structure:** block-major (pre-reorg; the `src/blocks/` reorg is the R1.5 assembly below)
- **arch.yml doc version:** `0.3-clean`
- **Milestone:** compute + memory blocks signed off on open PDKs (sky130 primary); GF180 Chipathon
  KV-coproc full-chip GDS closed. First coordinated baseline — the seed revision.
- **Track:** proxy (GF180/Sky130). N16 product track not yet started (kicks off ~R2).

| Block | Last-touch SHA | ISA | sky130 | gf180 | asap7 | Sign-off notes |
|---|---|---|---|---|---|---|
| `kve` | `3074d92` | `kv-isa-0.2` | **signed-off** @10 MHz (die 0.236 mm²) | config-only | — | 1 ss-corner max-cap near-miss (`SIGNOFF.md`); 3-way Py↔C++↔SV parity claimed |
| `tiu` | `5b51900` | `tiu-isa-0.1` | **no-gds**: metrics clean @40 MHz (die 0.015 mm²) but **GDS not committed** | config-only | — | headline checks all 0 + layout.png present, but no `*.gds*` in the tree — commit the GDS to reach signed-off. Py-only ref (29/29 + 40/40 replay claimed) |
| `src/blocks/acu/mate` | `3074d92` | — | **signed-off** ×3: pv@71, qkt@12.5, fp16@11.8 MHz | config-only | route-clean: pv@2 GHz, fp16@286 MHz | asap7 = route-clean only; FP16 parity is rel-err<5e-3 (not bit-exact, by design) |
| `src/blocks/acu/vecu` | `cb08d1b` | — | **signed-off**: rope, softmax @9.5 MHz | **signed-off**: rmsnorm, rope @3.85 MHz | — | half-covered: rmsnorm sky130 = config-only; softmax gf180 = config-only. Py-only ref, no committed parity test |
| `src/blocks/acu/precision_controller` | `3074d92` | `pc-isa-0.2` | **signed-off** @80 MHz (die 0.0086 mm², tightest slack +0.072 ns) | config-only | route-clean @1.18 GHz | Py+C++ parity claimed. **Drift risk:** byte-identical twin in `attention-compute-unit/` (no auto-sync) |
| `chip` | `17f5a6c` | `lh-isa-0.1` (unified) | — | full-chip KV-coproc GDS present (**prose-only** sign-off) | — | scope honestly downsized to coproc (fp16 datapath didn't fit 2051×2051 core). No committed metrics JSON |

**Known-open at R1 (carried into R1.5 assembly):**
1. gf180 per-block hardening artifacts missing for kve/tiu/mate/precision (numbers exist prose-only
   in `chip/pdk/gf180/docs/gf180_gls_report.md`).
2. chip full-chip sign-off has no machine-readable metrics JSON.
3. Two dangling configs: `vecu_rmsnorm` (sky130), `vecu_softmax` (gf180).
4. Metrics filenames not yet normalized to the SOP §5.2 schema (`kve` uses `metrics.json`).
5. Chip-level status docs drifted from block reality (RoPE/RMSNorm) — to be regenerated from
   ground truth via `scripts/gen_progress.py`.
6. **`tiu` sky130 GDS is not committed** — metrics are clean but no `*.gds*` artifact exists in the
   tree (found by `gen_progress.py`, correcting an earlier "GDS present" claim). Commit the GDS or
   mark it explicitly.

_Data source: per-block `*_metrics.json` / `SIGNOFF.md` (verified 2026-07-23); block SHAs from
`git log -1 -- <block>`; ISA versions from `<block>/docs/isa/`._

---

<!-- Next: R1.5 — full-chip GF180 assembly. Cut when blocks integrate into the full-chip build and
     remaining per-block GF180 hardening closes. See docs/ROADMAP.md. -->

---

## R1.5 — src/blocks reorg + Revision-Sync SOP + metrics normalization

- **Tag:** `rev-R1.5` · **Kind:** assembly
- **Date:** 2026-07-23 · **Monorepo anchor:** `fd78de6` (`fd78de69a3fd19715bb3007141565437c3e7fdf1`)
- **arch.yml doc version:** `0.3-clean`
- **Milestone:** src/blocks reorg + Revision-Sync SOP + metrics normalization

| Block | SHA | ISA | Sign-off summary |
|---|---|---|---|
| `kve` | `7589254` | `kv-isa-0.2` | **signed-off**: kv_cache_engine@sky130; **config-only**: kve@gf180, kve_store_gf180@gf180 |
| `tiu` | `7589254` | `tiu-isa-0.1` | **no-gds**: token_importance_unit@sky130; **config-only**: token_importance_unit@gf180 |
| `acu/mate` | `7589254` | `—` | **signed-off**: mate_pv@sky130, mate_pv_fp16@sky130, mate_qkt@sky130; **route-clean**: mate_pv@asap7, mate_pv_fp16@asap7; **config-only**: mate_pv@gf180, mate_pv_fp16@gf180, mate_qkt@gf180 |
| `acu/vecu` | `d401d66` | `—` | **signed-off**: rmsnorm@gf180, rope@gf180, rope@sky130, vecu_softmax@sky130; **config-only**: rmsnorm@sky130, vecu_softmax@gf180 |
| `acu/precision_controller` | `7589254` | `pc-isa-0.2` | **signed-off**: precision_controller@sky130; **route-clean**: precision_controller@asap7; **config-only**: precision_controller@gf180 |
| `chip` | `d401d66` | `lh-isa-0.1 (unified)` | **prose-only**: chip_top_coproc@gf180 |

_Generated by scripts/cut_revision.py from ground-truth artifacts._
