# KV Cache Engine × Precision Controller Integration Audit

**Branch:** `kvce-acu-integration-audit` (this repo) — also affects
`LonghornSilicon/kv-cache-engine` master at `d99fc00`.
**Status:** **Block — KVCE reference model must be fixed before ACU
integration is meaningful.**
**Date:** 2026-05-19.
**One-line summary:** While building the integration test that compares
precision-controller routing on lossy vs. clean V, we discovered the KV
Cache Engine's Python reference model is producing **essentially random
key reconstructions** — median cosine similarity **0.48** between input
and decompressed K vectors, on the most charitable input distribution
(unit-norm Gaussian, matching the Lloyd-Max centroid design point).
The repo's 184 tests pass because their MSE threshold is `max_val² ≈
10⁹` (a "doesn't crash" gate, not an accuracy gate). The ACU's
integration question can't be answered until the KVCE reconstruction
actually works.

---

## What we wanted to measure

Whether the precision controller's INT8/FP16 routing decision still
adds value when its inputs (S and V) come from KV-cache decompression
rather than the model's true FP16 tensors. Five paths per tile:

| Path | K source | V source | SV path |
|---|---|---|---|
| A — REF (baseline)      | true K | true V    | FP16 dense |
| B — PC alone (no KVCE)   | true K | true V    | PC-routed (INT8 or FP16) |
| C — KVCE alone, FP16 SV  | K_hat  | V_hat     | FP16 |
| D — KVCE alone, INT8 SV  | K_hat  | V_hat     | INT8 |
| E — Integrated           | K_hat  | V_hat     | PC-routed on lossy S |

Relative MSE of B, C, D, E vs. A would tell us whether the precision
controller's value-add survives the KVCE compression step.

## What we found instead

The smoke pass (1 layer of Qwen2-0.5B, 392 tiles, seq_len=512) returned:

| Path | Median rMSE vs REF |
|---|---|
| B — PC alone        | **0.0001** |
| C — KVCE + FP16 SV  | **17.01**  |
| D — KVCE + INT8 SV  | **17.05**  |
| E — Integrated      | **17.05**  |

Output error from the KVCE round-trip is **170,000× larger** than the
output error from the precision controller's INT8 routing. The FP16-vs-
INT8 distinction (0.04 rMSE difference) is invisible against the noise
floor introduced by KVCE decompression.

That's already disqualifying — but it doesn't tell us *where* the
problem is. Two follow-up probes isolate the cause to the KVCE
reference model itself, not the boundary conversion or my use of the
API.

### Probe 1 — KVCE on its own test vectors

Running `_random_vector(dim=64, ...)` (the generator the KVCE repo
uses in `test_round_trip_mse`), measured **relative** MSE on the
round-trip (the repo's own MSE measure is absolute INT-space, threshold
`max_val² ≈ 10⁹`):

| Path | Median rMSE | Mean rMSE | Max rMSE |
|---|---|---|---|
| K compress→decompress | **62×**  | 64×  | 83×  |
| V compress→decompress | **1.9×** | 1.9× | 2.1× |

"rMSE 62×" means the reconstruction error has 62× the variance of the
original signal. Their tests pass because the threshold is
`max_val² = (2¹⁵ − 1)² ≈ 10⁹` — which in float-space units corresponds
to per-element squared error of `(max_val/2¹²)² = 64`, ~60× the
variance of a unit-norm input. Their gate is "doesn't crash," not
"reconstructs."

### Probe 2 — Most-charitable input

Bypass any worry about test-vector distribution: feed exactly
`N(0, 1/√64)` Gaussians (which is what the Lloyd-Max centroids
[±0.2451σ, ±0.7560σ, ±1.3439σ, ±2.1520σ] are designed to approximate
optimally).

| Path | Median rMSE | Median cosine similarity |
|---|---|---|
| K | **515×** | **0.475** |
| V | **1.86×** | 0.824 |

**Cosine similarity 0.48 on K** is the diagnostic. The reconstructed
K vector points in essentially a random direction relative to the
original. For comparison: published KV-cache-quantization schemes
(GEAR, KIVI, KVQuant) target cosine > 0.99, rMSE < 0.05 on real LLM
activations. The KVCE reference model at its design point is two
orders of magnitude worse on both metrics.

## Why the precision controller wasn't the problem

Path B (precision controller alone, no KVCE, on real Qwen2 K and V)
came in at **rMSE 0.0001 median**. The ACU's INT8/FP16 routing
preserves the FP16 baseline to better than four decimal places on
this corpus. That number is consistent with the 91-97% INT8-safe
claim in the precision controller's paper. The ACU is fine.

## What this means for each repo

### `adaptive-precision-attention` (this repo)

- The precision controller's 1.7× INT8/FP16 win remains valid for the
  ACU **in isolation**, and any chip-level integration where K and V
  flow into the ACU without lossy compression in between.
- The original "Conflict #1" in the architectural conflict scan
  ("V is always lossy before the ACU sees it") is real but not the
  binding constraint — the binding constraint is "K is barely
  reconstructed at all, so the precision controller's *decision* and
  the *attention output* both collapse." The right framing of the
  conflict is upstream of the ACU, not at the ACU's input.
- No code changes needed here. The integration test harness
  (`analysis/integration_test_kv_pc.py`) is preserved on this branch
  for re-running once KVCE is fixed.

### `LonghornSilicon/kv-cache-engine` (separate repo)

This is the actionable side. Three concrete findings the KVCE team
should investigate, in order of severity:

1. **The Lloyd-Max + WHT + QJL reference's reconstruction quality is
   too poor to use, even on its design-point input distribution.** Path
   to confirm: add a `test_reconstruction_quality.py` that asserts
   relative MSE < 0.1 and cosine similarity > 0.9 on
   `_random_vector` inputs (or `N(0, 1/√d)` Gaussians). The current
   test threshold (`max_val²`) does not constrain accuracy.

2. **Possible suspects** (un-investigated, listed in order of "what I'd
   check first"):
   - The norm-rescale step in `decompress_key` (line 469-472 of
     `kv_cache_engine_ref.py`): `_fixed_mul(sv, ck.norm, coord_frac,
     norm_frac, coord_width, coord_frac)` with shift 12+8-12=8. If
     the sign/range handling here has an off-by-one, the entire
     output gets miscaled by 2× or 4×, which would match the
     "magnitude is way too big" observation in the round-trip.
   - The WHT rotation does *not* include the 1/√n normalization
     factor; the inverse WHT is `WHT(WHT(x)) = n·x`. If the rotate
     and inverse-rotate aren't matched on the normalization factor,
     the output gets scaled by 64×.
   - The QJL correction (`qjl_decompress`) reconstructs the residual
     using `√(π/2)` scaling — fixed-point representation of √(π/2)
     in Q4.12 has limited precision (1.2533… ≈ 5133 in int = 1.253
     ≈ correct), but the formula's correctness needs end-to-end check.

3. **The 184 passing tests are insufficient as a quality gate.** The
   repo's CI will accept any RTL implementation that bit-exactly
   matches the Python reference, regardless of whether the reference
   reconstructs anything close to the input. The shared block-CI
   workflow could optionally add an accuracy gate input
   (`min-cosine-similarity`, `max-relative-mse`) and the KVCE repo
   could opt into it.

### LonghornSilicon chip-level

Until KVCE is fixed, the integrated chip cannot run attention
correctly with KV cache in the loop, regardless of what the precision
controller does. This is build-blocking for chip integration but does
not affect either repo's standalone CI today.

The TurboQuant+ algorithm itself (PolarQuant + QJL) is sound in the
published literature; the issue is likely a bug in the reference
model's fixed-point arithmetic, not in the algorithmic design. So the
fix is probably a few-line correction in `kv_cache_engine_ref.py` +
matching RTL, not a re-design.

---

## Reproducing

```sh
# Confirm KVCE reconstruction quality on its own test vectors:
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
# Expect cosine < 0.6 — confirms the issue.

# Or run the integration smoke (~30s + ~7s model load):
python3 analysis/integration_test_kv_pc.py \
    --seq-len 512 --layers 0 --prompts 1
# Expect: path-B rMSE ~ 1e-4, paths C/D/E rMSE ~ 17
```

## Files added on this branch

```
analysis/integration_test_kv_pc.py          # five-path integration harness (preserved)
analysis/integration_test_kv_pc_stats.json  # smoke results showing the issue
docs/findings/kvce-acu-integration-audit.md # this document
```

## Next action

This branch should be **resurrected after** the KVCE team confirms
either:
  (a) reconstruction quality fix (cosine sim > 0.9 on `N(0, 1/√d)`
      Gaussians for both K and V paths), at which point we re-run the
      five-path integration test and answer the original question
      ("does FP16 routing still help under lossy V?") for real, or
  (b) an explicit decision that this is the intended quality, with a
      published end-to-end accuracy measurement on a real LLM showing
      the chip-level loss is acceptable — at which point the precision
      controller's FP16 path is decorative and can probably be
      simplified out.

Until then, ACU integration testing is blocked.
