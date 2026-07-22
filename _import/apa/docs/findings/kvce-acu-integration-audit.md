# KV Cache Engine x Precision Controller Integration Audit

> **Superseded** by [`kvce_acu_integration_audit.tex` / `.pdf`](kvce_acu_integration_audit.pdf)
> (`kvce-acu-audit-0.2`, 2026-05-22). This Markdown version is preserved
> as the working history; the LaTeX version is the canonical document.

**Branch:** `kvce-acu-integration-audit` (this repo)
**KVCE fix:** `kv-cache-engine@9b1163a` (closes kv-cache-engine#1)
**Status:** **Resolved.** KVCE reference model fixed; integration
test answers the original question.
**Date:** 2026-05-19 (initial), 2026-05-21 (re-run after fix).

**One-line summary:** After fixing a normalize shift bug and C++
floor-division parity issue in the KV Cache Engine reference model,
integrated median rMSE dropped from 17.0 to 0.36. The precision
controller's FP16 routing decision is 99.8% stable under lossy KV
inputs, but the FP16 path does not materially improve accuracy
because the KV cache quantization noise dominates.

---

## What we wanted to measure

Whether the precision controller's INT8/FP16 routing decision still
adds value when its inputs (S and V) come from KV-cache decompression
rather than the model's true FP16 tensors. Five paths per tile:

| Path | K source | V source | SV path |
|---|---|---|---|
| A -- REF (baseline)      | true K | true V    | FP16 dense |
| B -- PC alone (no KVCE)  | true K | true V    | PC-routed (INT8 or FP16) |
| C -- KVCE alone, FP16 SV | K_hat  | V_hat     | FP16 |
| D -- KVCE alone, INT8 SV | K_hat  | V_hat     | INT8 |
| E -- Integrated          | K_hat  | V_hat     | PC-routed on lossy S |

Relative MSE of B, C, D, E vs. A tells us whether the precision
controller's value-add survives the KVCE compression step.

## What we found (initial run, pre-fix)

The initial smoke pass (1 layer of Qwen2-0.5B, 392 tiles, seq_len=512)
revealed that the KVCE reference model was producing essentially
random K reconstructions (cosine sim 0.48). The repo's 184 tests
passed because their MSE threshold was `max_val^2` -- a "doesn't
crash" gate, not an accuracy gate.

| Path | Median rMSE vs REF (pre-fix) |
|---|---|
| B -- PC alone        | 0.0001 |
| C -- KVCE + FP16 SV  | **17.01** |
| D -- KVCE + INT8 SV  | **17.05** |
| E -- Integrated      | **17.05** |

## Root cause

Two bugs in the KVCE reference model (`kv_cache_engine_ref.py` and
`kv_cache_engine_ref.cpp`):

1. **Normalize shift bug:** `normalize()` used `coord_frac` (12)
   instead of `norm_frac` (8) for the left shift, causing a 16x
   overscale that destroyed quantization accuracy. K cosine similarity
   was 0.48 (essentially random); after fix, 0.978.

2. **C++ floor-division parity:** C++'s `/` operator truncates toward
   zero while Python's `//` floors toward negative infinity. For
   negative numerators this caused 1-unit differences accumulating
   across 64 vector elements. Added `floor_div()` helper.

Fix: `kv-cache-engine@9b1163a` (3 lines in Python, 8 lines in C++).
Quality gate added: `test_reconstruction_quality.py` asserts
cosine > 0.9 and rMSE < 0.1 on both K and V paths.

## Results after fix (2744 tiles, 7 layers)

Re-run on Qwen2-0.5B, seq_len=512, layers 0/4/8/12/16/20/23,
1 prompt (prose), 2744 off-diagonal tiles.

### Overall

| Path | Median rMSE | p90 | p99 |
|---|---|---|---|
| B -- PC alone (no KV cache) | 0.0002 | 0.0014 | 0.0250 |
| C -- KV alone, FP16 SV      | 0.3637 | 3.5709 | 13.538 |
| D -- KV alone, INT8 SV      | 0.3631 | 3.5662 | 13.447 |
| E -- Integrated (PC + KV)   | 0.3631 | 3.5662 | 13.447 |

### Decision stability

| Metric | Value |
|---|---|
| FP16% on clean S | 0.00% |
| FP16% on lossy S | 0.18% |
| Decisions agree | 99.82% |
| Flipped clean-FP16 to lossy-INT8 | 0 |
| Flipped clean-INT8 to lossy-FP16 | 5 |

### Per-layer breakdown

| Layer | n | Clean FP16% | Lossy FP16% | rMSE B | rMSE C | rMSE D | rMSE E |
|---|---|---|---|---|---|---|---|
| 0  | 392 | 0% | 0%    | 0.0001 | 2.567 | 2.576 | 2.576 |
| 4  | 392 | 0% | 0%    | 0.0002 | 0.534 | 0.534 | 0.534 |
| 8  | 392 | 0% | 0%    | 0.0002 | 0.254 | 0.254 | 0.254 |
| 12 | 392 | 0% | 0%    | 0.0003 | 0.159 | 0.160 | 0.160 |
| 16 | 392 | 0% | 1.28% | 0.0003 | 0.335 | 0.339 | 0.339 |
| 20 | 392 | 0% | 0%    | 0.0004 | 0.297 | 0.304 | 0.304 |
| 23 | 392 | 0% | 0%    | 0.0004 | 0.342 | 0.344 | 0.344 |

### Does FP16 routing help under lossy V?

Among all 2744 tiles (PC says INT8 on clean S for all of them):
- Median rMSE with FP16 SV (path C): 0.3637
- Median rMSE with INT8 SV (path D): 0.3631
- % tiles where FP16 beats INT8: 61.5%

The FP16/INT8 distinction (0.0006 rMSE difference) is invisible
against the KV cache quantization noise floor (0.36 rMSE).

## Interpretation

### Improvement from fix

| Metric | Pre-fix | Post-fix | Improvement |
|---|---|---|---|
| K cosine similarity | 0.48 | 0.978 | random -> accurate |
| Integrated median rMSE | 17.0 | 0.36 | 47x |
| KVCE rMSE / PC rMSE | 170,000x | 1,800x | 94x |

### Answers to the original questions

1. **Decision stability under lossy KV:** Excellent. 99.8% of
   precision controller decisions are unchanged when fed K_hat
   instead of true K. The 5 flipped tiles all went INT8->FP16
   (conservative direction). The decision function is robust to
   3-bit PolarQuant + 1-bit QJL noise.

2. **Does FP16 routing still help?** Not materially on this corpus.
   The KV cache quantization noise (median rMSE 0.36) is ~1800x
   larger than the precision controller's INT8 routing error (median
   rMSE 0.0002). The FP16 SV path provides <0.2% rMSE improvement
   over INT8 SV when both operate on lossy V_hat. For this model
   and prompt, the KV cache is the binding constraint, not the
   SV matmul precision.

3. **Integrated MSE vs standalone:** Path E (integrated) tracks
   path D (KV+INT8) almost exactly, because the PC routes nearly
   everything INT8. The precision controller adds no measurable
   harm -- it just can't add value either, since the KV cache
   noise floor is too high for the INT8/FP16 distinction to matter.

### Layer-level observations

- **Layer 0 is worst** (rMSE 2.57): early-layer activations have
  different distributions; the Lloyd-Max centroids (tuned for
  N(0, 1/sqrt(d))) are a poorer fit.
- **Layer 12 is best** (rMSE 0.16): mid-network activations most
  closely match the centroid design point.
- The 16x variation across layers suggests per-layer or adaptive
  centroid tables could significantly improve quality.

## What this means for each repo

### `attention-compute-unit` (this repo)

- The precision controller is validated: rMSE 0.0002 in isolation,
  decision stability 99.8% under lossy inputs. No code changes needed.
- The FP16 routing path is not decorative in general -- it remains
  valuable when V is not lossy-quantized (e.g., if the KVCE only
  compresses K, or on high-precision V paths). On this specific
  corpus with turbo4 (3-bit PQ + 1-bit QJL on both K and V), the
  KV noise dominates.
- Future work: test with higher-precision KVCE modes (turbo8, turbo16)
  where V_hat quality improves and the FP16 distinction may re-emerge.

### `LonghornSilicon/kv-cache-engine`

- Fix shipped: `kv-cache-engine@9b1163a` (closes #1).
- Quality gate added: `test_reconstruction_quality.py` with
  cosine > 0.9 and rMSE < 0.1 thresholds.
- Layer 0's rMSE 2.57 suggests the centroid table is a poor fit for
  early-layer activations. Per-layer calibration or adaptive centroids
  (turbo4-adaptive) could be a v0.2 feature.

### LonghornSilicon chip-level

- ACU integration is **unblocked**. The KV cache engine + precision
  controller can now operate in the loop with meaningful accuracy.
- The binding constraint for end-to-end attention accuracy is the KV
  cache compression quality (rMSE 0.36), not the SV matmul precision
  routing (rMSE 0.0002).

---

## Reproducing

```sh
# Confirm KVCE reconstruction quality (post-fix):
python3 -c "
import sys, numpy as np
sys.path.insert(0, '/home/shadeform/kv-cache-engine/sw/reference_model')
from kv_cache_engine_ref import KVCacheEngine, KVCacheEngineInfo
e = KVCacheEngine(KVCacheEngineInfo(vector_dim=64))
rng = np.random.default_rng(0)
v = rng.standard_normal(64) / np.sqrt(64)
q = (np.round(v * 4096).clip(-32768, 32767)).astype(np.int32).tolist()
kh = np.array(e.decompress_key(e.compress_key(q))) / 4096.0
print(f'K cosine sim: {float(v@kh/(np.linalg.norm(v)*np.linalg.norm(kh))):.3f}')
"
# Expect cosine ~ 0.975

# Full 7-layer integration test (~75s + model load):
cd /home/shadeform/attention-compute-unit
python3 analysis/integration_test_kv_pc.py \
    --seq-len 512 --layers 0,4,8,12,16,20,23 --prompts 1
# Expect: path-B rMSE ~ 2e-4, paths C/D/E rMSE ~ 0.36
```

## Files on this branch

```
analysis/integration_test_kv_pc.py          # five-path integration harness
analysis/integration_test_kv_pc_stats.json  # full results (2744 tiles, 7 layers)
docs/findings/kvce-acu-integration-audit.md # this document
```
