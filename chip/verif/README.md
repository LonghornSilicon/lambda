# Cross-block RTL cosim — LonghornSilicon "Lambda"

First **cross-block** integration of the live block RTLs. Each block is verified
bit-exact in its own repo's `rtl` branch; this harness proves they **co-simulate on one
shared attention tile** and each still matches its reference in-context — including a real
**INT8 P·V MAC** so the attention output flows all the way through, not a straight V̂ copy.

> **Process node.** The vendored RTL is a **130nm Sky130 proxy**. Lambda targets **TSMC
> 16nm**; Sky130 is the best open PDK we have, used only to get gate-level area/latency
> *estimates* that we scale to 16nm. Timing/area numbers here are proxy figures, not the
> 16nm silicon.

## What runs (`make cosim`)

One real-Qwen attention tile (`vectors/qwen_val.hex` = V rows, `vectors/qwen_vhatwht.hex`
= their reference V̂) drives all blocks in chip order — KVE reconstructs the rotated values,
the MatE P·V MAC accumulates them under attention weights, `wht_inverse_out` unspins the
result, and the TIU/ACU controllers run on the same tile:

| Block | RTL | Checked |
|---|---|---|
| **KVE** (block 2) | `cq_value_path_wht` → rotated V̂ (CQ-3-rot, Path B) | reconstructed V̂ **bit-exact** vs the vendored reference, per token |
| **MatE** (P·V) | `mate_pv` (INT8 P·V, INT32 acc) → `wht_inverse_out` | **int32 bit-exact** vs `matmul_int8`; full KVE→P·V→inverse output vs `Σ A·Ghat` within tol |
| **TIU** (block 3) | `token_importance_unit` (H2O) | `tier_keep` = (mass ≥ threshold) **and** eviction victim = min-mass slot |
| **ACU** (block 1) | `precision_controller` | precision gate `max·N > 10·Σ` on a peaky score row → FP16 |

```
[KVE ] CQ-3-rot V̂ over 8 real-Qwen tokens: bit-exact vs reference
[MatE] INT8 P·V MAC (mate_pv), 8 tokens x D=128, INT32 acc: int32 bit-exact vs matmul_int8
[MatE] e2e KVE->P·V->inverse vs Sigma A*Ghat: max rel err 0.002563 (within tol, tol 0.06)
[TIU ] keep-tier (thr=128) + eviction victim: match reference (evict slot 3)
[ACU ] precision gate on a peaky score row: match reference (fp16=1)

CROSS-BLOCK COSIM (ACU + KVE + MatE P·V + TIU on one shared tile): ALL BLOCKS PASS
```

## The P·V MAC end-to-end check

The MatE `mate_pv` block (vendored from `adaptive-precision-attention`) is the INT8 P·V
tile: signed int8 × int8 → **INT32** accumulate, bit-exact to `mac_array_ref.matmul_int8`.
The cosim quantizes the KVE's rotated V̂ to int8 (shared tile scale) plus an int8 attention
row, streams them through `mate_pv`, and checks the int32 result **bit-exact** against a
testbench-computed `Σ A·V`. It then dequantizes, feeds `wht_inverse_out`, and compares the
reconstructed attention output to `Σ_t A[t]·Ghat[t]` — which, because the inverse WHT is
linear (`inverse(Σ A·V̂rot) = Σ A·V̂ = Σ A·Ghat`), is computable from the reference values
with **no model in the loop**. On rotated (flat) values the INT8 tile is near-lossless, so
the e2e error is ~0.26%.

## Scope / what this is *not*

- Only the **INT8** P·V tile is in RTL here; the FP16 tile is tolerance-only in the
  reference (`MAC_ARRAY_DESIGN.md`) and remains HLS. The full 16×16 systolic MatE array is
  the SystemC/Stratus HLS project (`architecture/src/blocks/mate`); `mate_pv` is its
  synthesizable token-reduction vector-MAC core for the Sky130 flow.
- The KVE check is capped to 8 tokens (the fp16 WHT butterfly is combinational; 8 tokens is
  enough to prove bit-exactness in-context — the full 348,160-element proof lives in the
  kv-cache-engine `rtl` branch).
- `blocks/` is **vendored** from each block repo's `rtl` branch (kv-cache-engine, ACU +
  `mate_pv`, token-importance-unit). The per-block authoritative proofs run in those repos' CI.

## Reproduce

```sh
mkdir -p build && make cosim
```
