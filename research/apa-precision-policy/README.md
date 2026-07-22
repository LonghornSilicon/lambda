# Adaptive Precision Attention (APA) — precision-policy research

This is the **RL research project that birthed the ACU**. It is chip-wide research (not block RTL),
so it lives at the repo root under `research/`, per the monorepo block-major layout
(`docs/repo_reorg_plan.md`, decision #2).

## What it is
An evolutionary/RL search over per-tile attention precision policies. It started from an
entropy-based peakedness detector and **discovered the simplification that became the ACU's
precision controller**: the pre-softmax ratio test `max(|S|)·N > 10·Σ|S|` (entropy-equivalent, but
no softmax/exponentials/division). The hardware that this research produced now lives, verified and
signed off, in `acu/mate/`, `acu/vecu/`, and `acu/precision_controller/`.

## Contents
| Dir | What |
|---|---|
| `phase1_policy/` | RL policy evolution (multi-objective evaluator, initial policy, config) |
| `phase2_kernel/` | the evolved adaptive-attention kernel + its evaluator/tests |
| `common/` | shared prototypes: quantization, block stats, reference, workloads |
| `kv_cache/` | Python KV-cache + a software precision controller / quantized cache |
| `analysis/` | 48 files — benchmarks (vs FA-2), entropy approx, block-size + seq-length sweeps, fixed-point sim, real-LLM validation (Qwen2/Phi-2/Llama), RTL test-vector gen, the paper figures |
| `paper/` | the APA method paper (`adaptive_precision_attention.{tex,pdf}`) + figures + refs |
| `tests/` | pytest for the research modules (evaluator, kernel, kv-cache, quantization, workloads, ...) |
| `superpowers/` | FlashAttention-5 design plans/specs + phase-1 analysis dashboard plans |
| `findings/` | `phase1-entropy-finding.md`, `sparsity-controller-finding.md` |
| `run_phase1.sh`, `run_phase2.sh`, `requirements.txt` | the research runners + deps |

## Provenance
Imported with full git history from the `attention-compute-unit` repo (union of its `rtl`, `master`,
and `pdk-asap7` branches) during the 2026-07-22 ACU monorepo import. See
`docs/acu_import_completeness.md` for the file-by-file accounting.

> **Note:** the `attention-compute-unit` repo also has an unmerged `c11-end-to-end-perplexity`
> research branch (ChannelQuant/KVCE end-to-end perplexity campaign) that was **not** imported here —
> it overlaps the KVE/ChannelQuant blocks and is out of scope for this ACU import. Flagged in the
> completeness report so it isn't silently lost.
