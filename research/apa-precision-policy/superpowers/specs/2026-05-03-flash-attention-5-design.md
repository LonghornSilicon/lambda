# FlashAttention-5: Adaptive Precision Attention via Evolutionary Search

## Problem Statement

FlashAttention-3 treats precision as a global choice — all blocks computed in FP16 or all in FP8 with uniform block quantization. In practice, different blocks of the attention matrix have wildly different precision requirements:

- Blocks near the causal diagonal carry most signal and need high precision
- Blocks with outlier scores (common in LLMs) need careful handling
- Blocks in low-entropy, low-magnitude regions can be heavily compressed with minimal accuracy loss
- As sequence length grows, the fraction of "compressible" blocks increases

FA-5 discovers the optimal per-block precision allocation policy through evolutionary search (OpenEvolve + Claude Code CLI), then co-optimizes a kernel implementation around that policy.

## Prior Work

- **FlashAttention-1/2** (Dao et al.): Tiled, fused attention that avoids materializing the N×N attention matrix. IO-aware algorithm design.
- **FlashAttention-3** (Shah et al.): Hopper-specific optimizations — warp specialization, GEMM-softmax pipelining, FP8 with block quantization and incoherent processing (Hadamard rotation). Reaches 740 TFLOPs/s FP16, ~1.2 PFLOPs/s FP8 on H100.
- **Adaptive KV-Cache Quantization** (Boroujeni et al., CVPR 2026): Per-token precision allocation {2,4,8,16}-bit based on importance signals (entropy, attention variance, frequency). 17.75% latency reduction, within 0.30 points of FP16 accuracy.
- **TurboQuant-Plus** (local prior work): Hadamard rotation experiments, value quantization, fused decode kernels for KV-cache compression.

## Design Overview

Two-phase evolutionary search using OpenEvolve with Claude Code CLI as the LLM backend.

### Phase 1: Precision Policy Evolution

Evolve a Python function that maps per-block statistics to a precision level. Pure PyTorch evaluation — no kernel optimization. Focus on accuracy vs. compression tradeoff.

### Phase 2: Kernel Evolution

Take the winning policy from Phase 1. Evolve a Triton attention kernel that implements tiled mixed-precision attention with that policy baked in. Focus on throughput.

## Project Structure

```
flash-attention-5/
├── common/
│   ├── workloads.py              # Standard attention workloads
│   ├── reference.py              # FP64 reference attention
│   ├── quantization.py           # Quantization primitives (FP16/FP8/FP4/INT8/INT4)
│   └── block_stats.py            # Per-block statistics extraction
├── phase1_policy/
│   ├── initial_policy.py         # Evolved by OpenEvolve: precision allocation function
│   ├── evaluator.py              # Multi-objective scorer
│   └── config.yaml               # OpenEvolve config (Claude Code backend)
├── phase2_kernel/
│   ├── initial_kernel.py         # Triton attention kernel with mixed-precision
│   ├── evaluator.py              # Throughput + accuracy scorer
│   └── config.yaml               # OpenEvolve config
├── run_phase1.sh                 # Launch Phase 1 evolution
├── run_phase2.sh                 # Launch Phase 2 evolution
└── requirements.txt
```

## Phase 1: Precision Policy — Detailed Spec

### Evolved Function Interface

```python
# EVOLVE-BLOCK-START
def precision_policy(block_stats: dict) -> str:
    """Map block-level statistics to a precision level.

    Args:
        block_stats: dict with keys:
            q_norm       (float): L2 norm of query block
            k_norm       (float): L2 norm of key block
            v_norm       (float): L2 norm of value block
            score_max    (float): max pre-softmax attention score in block
            score_mean   (float): mean pre-softmax attention score
            score_var    (float): variance of pre-softmax scores
            entropy      (float): softmax entropy (higher = more uniform)
            causal_dist  (int):   |query_pos - key_pos| average for block
            block_row    (int):   query block index
            block_col    (int):   key block index
            has_outlier  (bool):  True if max value > 6 * mean abs value
            seq_len      (int):   total sequence length
            head_dim     (int):   head dimension
            num_heads    (int):   number of attention heads

    Returns:
        One of: "fp16", "fp8_e4m3", "fp8_e5m2", "int8", "fp4", "int4"
    """
    return "fp16"
# EVOLVE-BLOCK-END
```

### Block Statistics Extraction

Before calling the policy, we compute block-level statistics using the same tiling structure as the attention algorithm itself. Each block's stats are computed independently — we never materialize the full N×N matrix. This is the "oracle" approach; in a production kernel, these statistics would be computed on-the-fly or approximated from the previous layer's attention pattern.

Block size for statistics: B_r = B_c = 128 (matching FA-3's typical block size).

**A4000 note:** Ampere GPUs lack native FP8/FP4 tensor cores. All sub-FP16 quantization is simulated via quantize-dequantize roundtrips (clamp to representable range, round, cast back to FP32). This is correct for measuring accuracy and compression — the numerical error introduced by the roundtrip is identical to real hardware quantization. Kernel throughput benefits of lower precision are only measurable on Hopper+ hardware (Phase 2 on H100).

Statistics computed per block (i, j):
1. `q_norm`: torch.norm(Q_block) / sqrt(numel)
2. `k_norm`: torch.norm(K_block) / sqrt(numel)
3. `v_norm`: torch.norm(V_block) / sqrt(numel)
4. `score_max`: max(S_block) where S = Q_block @ K_block.T / sqrt(d)
5. `score_mean`: mean(S_block)
6. `score_var`: var(S_block)
7. `entropy`: -sum(P_block * log(P_block + eps)) averaged over rows, where P = softmax(S)
8. `causal_dist`: abs(mean(row_indices) - mean(col_indices))
9. `block_row`, `block_col`: integer indices
10. `has_outlier`: any(abs(S_block) > 6 * mean(abs(S_block)))
11. `seq_len`, `head_dim`, `num_heads`: from input shapes

### Quantization Primitives

Each precision level is implemented as a quantize-then-dequantize roundtrip:

| Level | Format | Bits | Implementation |
|-------|--------|------|----------------|
| fp16 | IEEE FP16 | 16 | Native torch.float16 |
| fp8_e4m3 | E4M3 | 8 | torch.float8_e4m3fn (if available) or simulated clamp+round |
| fp8_e5m2 | E5M2 | 8 | torch.float8_e5m2 (if available) or simulated |
| int8 | Symmetric INT8 | 8 | Per-block scale, round to [-127, 127] |
| fp4 | E2M1 | 4 | Simulated: 4-bit mantissa/exponent roundtrip |
| int4 | Symmetric INT4 | 4 | Per-block scale, round to [-7, 7] |

All quantization uses per-block scaling (one scale factor per B_r × d or B_c × d block).

### Mixed-Precision Attention Algorithm

```
For each query block Q_i (i = 0..T_r-1):
    Initialize O_i = 0, m_i = -inf, l_i = 0
    For each key/value block K_j, V_j (j = 0..T_c-1):
        1. Compute block_stats for (Q_i, K_j, V_j)
        2. precision = precision_policy(block_stats)
        3. Q_q = quantize(Q_i, precision)  # quantize-dequantize roundtrip
        4. K_q = quantize(K_j, precision)
        5. S_ij = Q_q @ K_q.T / sqrt(d)   # scores at selected precision
        6. Apply causal mask if needed
        7. Online softmax update: m_i, l_i, O_i
        8. V_q = quantize(V_j, precision)
        9. O_i += P_ij @ V_q              # accumulate at FP32
    Final rescale O_i
```

Accumulation always happens in FP32 (matching FA-3's approach). Only the GEMM operands are quantized.

### Evaluator Scoring

Workload configurations (12 total):

| Seq Length | Head Dim | Causal | Outliers |
|-----------|---------|--------|----------|
| 512 | 64 | No | No |
| 512 | 128 | Yes | Yes |
| 2048 | 64 | No | Yes |
| 2048 | 128 | Yes | No |
| 8192 | 64 | Yes | No |
| 8192 | 128 | No | Yes |
| 2048 | 64 | Yes | Yes |
| 2048 | 128 | No | No |
| 4096 | 64 | No | No |
| 4096 | 128 | Yes | Yes |
| 8192 | 64 | No | No |
| 8192 | 128 | Yes | No |

For each workload:
- Batch size = 2, num_heads = 8
- Total tokens capped at 16k to keep eval under 60s total
- Input distribution: N(0,1) + N(0,100) * Bernoulli(0.001) for outlier workloads

Outlier distribution matches FA-3 paper's validation setup.

Metrics per workload:
- RMSE vs FP64 reference output
- Average bits per element (weighted by block precision choices)
- Whether output contains NaN/Inf (instant zero score)

Aggregated scoring:
```
accuracy_score    = mean(1.0 / (1.0 + rmse * 5000))     across workloads
compression_score = mean((16 - avg_bits) / 14)           across workloads
reliability_score = fraction of workloads without NaN/Inf

combined_score = 0.50 * accuracy_score
               + 0.30 * compression_score
               + 0.20 * reliability_score
```

### OpenEvolve Configuration

```yaml
max_iterations: 80
checkpoint_interval: 10

llm:
  provider: "claude_code"
  models:
    - name: "sonnet"
      weight: 0.8
      max_tokens: 16000
      timeout: 300
      max_budget_usd: 1.0
    - name: "haiku"
      weight: 0.2
      max_tokens: 8000
      timeout: 120
      max_budget_usd: 0.5
  retries: 3
  retry_delay: 5

prompt:
  system_message: >
    You are an expert in GPU attention mechanisms, quantization, and numerical
    optimization. Your task is to improve a precision allocation policy for
    mixed-precision attention. The policy maps per-block statistics (score
    magnitudes, entropy, outlier presence, causal distance) to precision levels
    (fp16, fp8_e4m3, fp8_e5m2, int8, fp4, int4). The goal is to maximize
    accuracy (minimize RMSE vs FP64 reference) while maximizing compression
    (using lower precision where possible). Blocks with high attention scores,
    outliers, or low entropy typically need higher precision. Blocks with low
    scores, high causal distance, or high entropy can often tolerate lower
    precision. Consider that accumulation happens in FP32, so quantization
    error in individual blocks is partially averaged out.

database:
  population_size: 50
  archive_size: 20
  num_islands: 3
  elite_selection_ratio: 0.2
  exploitation_ratio: 0.7
  similarity_threshold: 0.99

evaluator:
  timeout: 120
  cascade_thresholds: [0.3]
  parallel_evaluations: 2

diff_based_evolution: true
max_code_length: 10000
```

## Phase 2: Kernel Evolution — Detailed Spec

### Prerequisites

Phase 1 must complete first. The best policy from Phase 1 is saved as `phase2_kernel/best_policy.py`.

### Evolved Kernel Interface

```python
# EVOLVE-BLOCK-START
def flash_attention_5(Q, K, V, causal=False, precision_policy=None):
    """Mixed-precision fused attention.

    Args:
        Q: (batch, heads, seq_len, head_dim) float16
        K: (batch, heads, seq_len, head_dim) float16
        V: (batch, heads, seq_len, head_dim) float16
        causal: whether to apply causal mask
        precision_policy: callable from Phase 1

    Returns:
        O: (batch, heads, seq_len, head_dim) float16
    """
    # Implementation to be evolved — starting point is a tiled
    # PyTorch implementation, graduating to Triton
# EVOLVE-BLOCK-END
```

### Search Space

OpenEvolve can modify:
- Block sizes B_r, B_c (from 32 to 256, powers of 2)
- Whether to pre-compute block stats or approximate them cheaply
- Accumulator precision (FP32 vs FP16 for intermediate results)
- Whether to apply incoherent processing (Hadamard rotation) selectively
- Loop ordering (Q-outer vs K-outer)
- Whether to fuse quantization into the tiling loop or pre-quantize

### Evaluator Scoring

Same workloads as Phase 1, plus larger sequences if hardware permits:
- Additional workloads: seq_len 16384, 32768

Metrics:
```
throughput_score = min(measured_tflops / theoretical_peak, 1.0)
accuracy_score   = 1.0 / (1.0 + rmse * 5000)

combined_score = 0.60 * throughput_score + 0.40 * accuracy_score
```

Theoretical peak for A4000: ~150 TFLOPs/s FP16 Tensor Core.
For H100 (if hardware upgraded): ~990 TFLOPs/s FP16, ~1980 TFLOPs/s FP8.

## What Makes This "FlashAttention-5"

1. **Per-block adaptive precision** — subsumes FA-3's uniform FP8 as a special case. The policy can choose FP8 everywhere (recovering FA-3 behavior) or allocate precision non-uniformly for better accuracy/compression Pareto frontier.

2. **Evolved, not hand-designed** — the precision strategy is discovered through search over real workloads with outlier distributions matching production LLMs. As new precision formats emerge, add them to the action space and re-evolve.

3. **Selective incoherent processing** — FA-3 always applies Hadamard rotation before FP8 quantization. FA-5 can learn to apply it only for blocks with outliers, saving the O(d log d) cost when it's unnecessary.

4. **Architecture-portable strategy** — the policy is a pure function over block statistics. Any kernel (CUDA, Triton, Etched ASIC, Groq TSP) can call it. The kernel is hardware-specific; the precision strategy is universal.

5. **Context-length aware** — the policy receives seq_len and causal_dist as inputs, so it can learn that longer sequences have more compressible blocks (blocks far from the diagonal), automatically improving compression ratio as context grows.

## Hardware Requirements

- **Phase 1 (policy evolution):** Any CUDA GPU with 8+ GB VRAM. RTX A4000 (15 GB) is sufficient. ~80 iterations × ~60s eval = ~80 minutes.
- **Phase 2 (kernel evolution):** Same GPU for Triton kernels. H100 recommended for final benchmarking against FA-3. ~50 iterations × ~90s eval = ~75 minutes.

## Dependencies

```
torch>=2.1.0
triton>=2.2.0
numpy
```

Optional (for H100):
```
flash-attn>=2.5.0   # for baseline comparison
```

## Success Criteria

Phase 1:
- Discover a policy achieving < 6 average bits per element with RMSE within 2× of uniform FP16
- Combined score > 0.65 (baseline uniform FP16 scores ~0.35 due to zero compression)

Phase 2:
- Kernel achieves > 50% of theoretical peak TFLOPs/s on target GPU
- Accuracy within 1.5× of FP16 reference RMSE
