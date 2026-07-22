# research/ — chip-wide research (stub)

Chip-wide research and exploration context (design rationale, dead ends, benchmarks) that isn't
owned by a single block. Per-block research lives in `<block>/research/` and rides along in that
block's mirror.

To land here (per `docs/repo_reorg_plan.md`, decision #2):
- The **Adaptive Precision Attention (APA) RL project** — the evolutionary policy/kernel search that
  discovered entropy as the per-block quantization discriminator (currently in the
  `attention-compute-unit` repo under `phase1_policy/`, `phase2_kernel/`, `analysis/`). Import it
  here as an archived research subdir when the ACU is imported (its import is HELD this round).

This directory is intentionally a stub for now — it is seeded so the structure exists and so
chip-wide experiment notes have a home from day one.
