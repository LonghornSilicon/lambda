"""Adaptive Precision Attention — Triton kernel implementation.

Implements FlashAttention-2 tile loop with per-tile precision selection:
  - Compute S = Q·Kᵀ in FP16 (same as standard FA)
  - Compute ratio = max(|S|) / mean(|S|)
  - ratio > RATIO_THRESHOLD → tile is "outlier" → keep FP16
  - ratio <= RATIO_THRESHOLD → tile is "safe" → quantize K,V to INT8 and back

The quantization simulates what hardware INT8 multipliers would do.
On ASIC: INT8 path uses integer multipliers (smaller, faster).
Here in software: both paths use FP16 dot, so no throughput gain —
the value is proving correctness and measuring decision overhead.
"""

import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _adaptive_attn_fwd(
    Q, K, V, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    RATIO_THRESHOLD: tl.constexpr,
):
    """
    Grid: (ceil(N/BLOCK_M),  Z*H)
    Each program handles one Q-block for one (batch, head).
    """
    start_m = tl.program_id(0)
    off_hz  = tl.program_id(1)
    off_z   = off_hz // H
    off_h   = off_hz % H

    # Base pointers for this (batch, head)
    Q_ptr = Q + off_z * stride_qz + off_h * stride_qh
    K_ptr = K + off_z * stride_kz + off_h * stride_kh
    V_ptr = V + off_z * stride_vz + off_h * stride_vh
    O_ptr = Out + off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q block [BLOCK_M, BLOCK_DMODEL]
    q = tl.load(
        Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < N, other=0.0,
    )

    # Online-softmax accumulators (FP32 for stability)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # For causal, only attend to tokens up to the current Q-block's last row
    hi = (start_m + 1) * BLOCK_M if CAUSAL else N

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < N

        # Load K [BLOCK_DMODEL, BLOCK_N] and V [BLOCK_N, BLOCK_DMODEL]
        k = tl.load(
            K_ptr + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd,
            mask=kv_mask[None, :], other=0.0,
        )
        v = tl.load(
            V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=kv_mask[:, None], other=0.0,
        )

        # ── Compute raw scores ────────────────────────────────────────────
        s = tl.dot(q, k) * sm_scale  # [BLOCK_M, BLOCK_N], fp32

        # ── Precision decision: ratio = max(|S|) / mean(|S|) ─────────────
        abs_s  = tl.abs(s)
        s_max  = tl.max(abs_s)
        s_mean = tl.sum(abs_s) / (BLOCK_M * BLOCK_N)
        ratio  = s_max / (s_mean + 1e-6)
        use_int8 = ratio < RATIO_THRESHOLD  # scalar bool

        # ── INT8 simulation: quantize K and V ─────────────────────────────
        # K: per-tensor symmetric INT8
        k_scale   = tl.max(tl.abs(k)) / 127.0 + 1e-6
        k_q       = tl.floor(k / k_scale + 0.5)            # round to nearest
        k_q       = tl.minimum(tl.maximum(k_q, -127.0), 127.0)
        k_dequant = (k_q * k_scale).to(tl.float16)

        # V: per-tensor symmetric INT8
        v_scale   = tl.max(tl.abs(v)) / 127.0 + 1e-6
        v_q       = tl.floor(v / v_scale + 0.5)
        v_q       = tl.minimum(tl.maximum(v_q, -127.0), 127.0)
        v_dequant = (v_q * v_scale).to(tl.float16)

        # Select K, V based on precision decision
        # use_int8 is a scalar — tl.where broadcasts it across the tile
        k_final = tl.where(use_int8, k_dequant, k.to(tl.float16))
        v_final = tl.where(use_int8, v_dequant, v.to(tl.float16))

        # Recompute scores with chosen K
        s = tl.dot(q, k_final) * sm_scale

        # Causal mask (compile-time branch via constexpr)
        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # ── Online softmax update (FA-2) ──────────────────────────────────
        m_ij  = tl.max(s, axis=1)
        p     = tl.exp(s - m_ij[:, None])
        l_ij  = tl.sum(p, axis=1)

        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i  - m_new)
        beta  = tl.exp(m_ij - m_new)

        l_i = alpha * l_i + beta * l_ij
        acc = (alpha[:, None] * acc
               + beta[:, None] * tl.dot(p.to(tl.float16), v_final))
        m_i = m_new

    # Normalise
    acc = acc / l_i[:, None]

    # Store output [BLOCK_M, BLOCK_DMODEL]
    tl.store(
        O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(tl.float16),
        mask=offs_m[:, None] < N,
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def adaptive_attention(Q, K, V, causal=False, ratio_threshold=10.0):
    """Adaptive precision attention forward pass.

    Args:
        Q, K, V: (B, H, N, d) float16 tensors on CUDA
        causal: apply causal mask
        ratio_threshold: max/mean ratio above which FP16 is used (default 10)

    Returns:
        O: (B, H, N, d) float16 output tensor
    """
    assert Q.dtype == torch.float16, "Q must be float16"
    assert Q.is_cuda, "tensors must be on CUDA"
    B, H, N, d = Q.shape
    assert d in (32, 64, 128), f"head_dim {d} not supported (use 32, 64, or 128)"

    sm_scale = 1.0 / math.sqrt(d)
    O = torch.empty_like(Q)

    # Shared memory budget: BLOCK_M*d + BLOCK_N*d + BLOCK_M*d (Q, K, V tiles)
    # × 2 bytes (fp16) × num_stages < 101376 bytes (A4000 per-SM limit)
    # d=64:  128×64×2×3×2 = 98304 ✓   d=128: 128×128×2×3×2 = 196608 ✗
    # For d=128 use 64×64 blocks: 64×128×2×3×2 = 98304 ✓
    if d == 128:
        BLOCK_M, BLOCK_N, num_warps = 64, 64, 4
    else:
        BLOCK_M, BLOCK_N, num_warps = 128, 128, 4

    grid = (triton.cdiv(N, BLOCK_M), B * H)

    _adaptive_attn_fwd[grid](
        Q, K, V, sm_scale, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        B, H, N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=d,
        CAUSAL=causal,
        RATIO_THRESHOLD=ratio_threshold,
        num_warps=num_warps,
        num_stages=2,
    )
    return O
