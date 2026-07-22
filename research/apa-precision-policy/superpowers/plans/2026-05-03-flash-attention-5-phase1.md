# FlashAttention-5 Phase 1: Precision Policy Evolution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OpenEvolve pipeline that evolves a per-block precision allocation policy for mixed-precision attention, discovering which attention blocks can be compressed to FP8/FP4/INT4 without meaningful accuracy loss.

**Architecture:** Two-layer project: (1) a `common/` library providing FP64 reference attention, simulated quantization primitives, block statistics extraction, and standardized workloads; (2) a `phase1_policy/` OpenEvolve example with an evolvable policy function, evaluator, and config using Claude Code CLI as the LLM backend. OpenEvolve is already installed at `/home/shadeform/openevolve` with Claude Code support on the `feat/claude-code-llm` branch.

**Tech Stack:** Python 3.10, PyTorch >=2.1.0, NumPy, OpenEvolve (local install), Claude Code CLI

**Spec:** `docs/superpowers/specs/2026-05-03-flash-attention-5-design.md`

---

## File Map

```
flash-attention-5/
├── requirements.txt                          # PyTorch, numpy
├── common/
│   ├── __init__.py
│   ├── reference.py                          # FP64 tiled reference attention (no N×N materialization)
│   ├── quantization.py                       # Simulated quantize-dequantize for 6 precision levels
│   ├── block_stats.py                        # Extract per-block statistics during tiled attention
│   └── workloads.py                          # Generate standard (Q,K,V) test inputs
├── phase1_policy/
│   ├── initial_policy.py                     # The file OpenEvolve evolves (EVOLVE-BLOCK markers)
│   ├── evaluator.py                          # Multi-objective scorer: accuracy × compression × reliability
│   └── config.yaml                           # OpenEvolve config (Claude Code CLI backend)
├── tests/
│   ├── test_reference.py                     # Reference attention correctness
│   ├── test_quantization.py                  # Quantization roundtrip tests
│   ├── test_block_stats.py                   # Block stats shape/value tests
│   ├── test_workloads.py                     # Workload generation tests
│   ├── test_evaluator.py                     # Evaluator integration test
│   └── test_mixed_precision_attention.py     # Mixed-precision attention end-to-end
├── run_phase1.sh                             # Launch script
└── docs/
    └── superpowers/
        ├── specs/
        │   └── 2026-05-03-flash-attention-5-design.md
        └── plans/
            └── 2026-05-03-flash-attention-5-phase1.md  (this file)
```

---

### Task 1: Project Setup and Dependencies

**Files:**
- Create: `flash-attention-5/requirements.txt`
- Create: `flash-attention-5/common/__init__.py`
- Create: `flash-attention-5/tests/__init__.py` (empty)

- [ ] **Step 1: Initialize git repo**

```bash
cd /home/shadeform/flash-attention-5
git init
```

- [ ] **Step 2: Create requirements.txt**

```
torch>=2.1.0
numpy>=1.24.0
```

- [ ] **Step 3: Install dependencies**

```bash
cd /home/shadeform/flash-attention-5
pip install -r requirements.txt
```

Expected: PyTorch installs with CUDA support (A4000).

- [ ] **Step 4: Create package init files**

`common/__init__.py`: empty file initially. Will be populated in Task 6 after all modules exist.

`tests/__init__.py`: empty file.

- [ ] **Step 5: Verify PyTorch + CUDA**

```bash
cd /home/shadeform/flash-attention-5
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
```

Expected: `PyTorch 2.x.x, CUDA 12.x, GPU: NVIDIA RTX A4000`

- [ ] **Step 6: Create .gitignore and commit**

`.gitignore`:
```
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
checkpoints/
*.pt
```

```bash
cd /home/shadeform/flash-attention-5
git add requirements.txt common/__init__.py tests/__init__.py .gitignore
git commit -m "feat: project setup with PyTorch dependency"
```

---

### Task 2: FP64 Reference Attention

**Files:**
- Create: `flash-attention-5/common/reference.py`
- Create: `flash-attention-5/tests/test_reference.py`

- [ ] **Step 1: Write failing test**

`tests/test_reference.py`:
```python
import torch
import math
import sys
sys.path.insert(0, "/home/shadeform/flash-attention-5")
from common.reference import reference_attention


def test_reference_matches_naive():
    """Reference tiled attention must match naive O(N^2) attention in FP64."""
    torch.manual_seed(42)
    B, H, N, d = 1, 2, 256, 64
    Q = torch.randn(B, H, N, d, dtype=torch.float64)
    K = torch.randn(B, H, N, d, dtype=torch.float64)
    V = torch.randn(B, H, N, d, dtype=torch.float64)

    # Naive attention
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.softmax(S, dim=-1)
    O_naive = torch.matmul(P, V)

    O_ref = reference_attention(Q, K, V, causal=False)
    assert torch.allclose(O_naive, O_ref, atol=1e-10), f"Max diff: {(O_naive - O_ref).abs().max()}"


def test_reference_causal():
    """Causal reference attention must zero out future positions."""
    torch.manual_seed(42)
    B, H, N, d = 1, 1, 128, 64
    Q = torch.randn(B, H, N, d, dtype=torch.float64)
    K = torch.randn(B, H, N, d, dtype=torch.float64)
    V = torch.randn(B, H, N, d, dtype=torch.float64)

    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    S.masked_fill_(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    O_naive = torch.matmul(P, V)

    O_ref = reference_attention(Q, K, V, causal=True)
    assert torch.allclose(O_naive, O_ref, atol=1e-10), f"Max diff: {(O_naive - O_ref).abs().max()}"


def test_reference_block_size_independent():
    """Result must be identical regardless of internal block size."""
    torch.manual_seed(42)
    B, H, N, d = 1, 1, 512, 64
    Q = torch.randn(B, H, N, d, dtype=torch.float64)
    K = torch.randn(B, H, N, d, dtype=torch.float64)
    V = torch.randn(B, H, N, d, dtype=torch.float64)

    O_64 = reference_attention(Q, K, V, causal=False, block_size=64)
    O_128 = reference_attention(Q, K, V, causal=False, block_size=128)
    assert torch.allclose(O_64, O_128, atol=1e-10)


if __name__ == "__main__":
    test_reference_matches_naive()
    test_reference_causal()
    test_reference_block_size_independent()
    print("All reference attention tests passed.")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/shadeform/flash-attention-5
python -m pytest tests/test_reference.py -v
```

Expected: `ModuleNotFoundError: No module named 'common.reference'` or `ImportError`

- [ ] **Step 3: Implement reference attention**

`common/reference.py`:
```python
import torch
import math


def reference_attention(Q, K, V, causal=False, block_size=128):
    """Tiled FP64 reference attention using online softmax.

    Computes exact attention without materializing the full N×N score matrix.
    Uses the same online softmax algorithm as FlashAttention.

    Args:
        Q: (B, H, N, d) float64
        K: (B, H, N, d) float64
        V: (B, H, N, d) float64
        causal: apply causal mask
        block_size: tile size for tiled computation

    Returns:
        O: (B, H, N, d) float64
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    Br = block_size
    Bc = block_size
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)

    O = torch.zeros_like(Q)
    L = torch.zeros(B, H, N, 1, dtype=Q.dtype, device=Q.device)
    M = torch.full((B, H, N, 1), float("-inf"), dtype=Q.dtype, device=Q.device)

    for i in range(Tr):
        q_start = i * Br
        q_end = min(q_start + Br, N)
        Q_block = Q[:, :, q_start:q_end, :]

        o_block = torch.zeros_like(Q_block)
        m_block = torch.full(
            (B, H, q_end - q_start, 1), float("-inf"),
            dtype=Q.dtype, device=Q.device,
        )
        l_block = torch.zeros(
            B, H, q_end - q_start, 1, dtype=Q.dtype, device=Q.device,
        )

        j_end = Tc if not causal else min(i + 1, Tc)
        for j in range(j_end):
            k_start = j * Bc
            k_end = min(k_start + Bc, N)
            K_block = K[:, :, k_start:k_end, :]
            V_block = V[:, :, k_start:k_end, :]

            S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale

            if causal:
                q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                causal_mask = q_idx < k_idx
                S_block.masked_fill_(causal_mask, float("-inf"))

            m_block_old = m_block.clone()
            m_new = torch.max(S_block, dim=-1, keepdim=True).values
            m_block = torch.maximum(m_block, m_new)

            exp_scores = torch.exp(S_block - m_block)
            exp_old = torch.exp(m_block_old - m_block)

            l_block = exp_old * l_block + exp_scores.sum(dim=-1, keepdim=True)
            o_block = exp_old * o_block + torch.matmul(exp_scores, V_block)

        o_block = o_block / l_block
        O[:, :, q_start:q_end, :] = o_block
        L[:, :, q_start:q_end, :] = l_block
        M[:, :, q_start:q_end, :] = m_block

    return O
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/shadeform/flash-attention-5
python -m pytest tests/test_reference.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/shadeform/flash-attention-5
git add common/reference.py tests/test_reference.py
git commit -m "feat: FP64 tiled reference attention with online softmax"
```

---

### Task 3: Quantization Primitives

**Files:**
- Create: `flash-attention-5/common/quantization.py`
- Create: `flash-attention-5/tests/test_quantization.py`

- [ ] **Step 1: Write failing test**

`tests/test_quantization.py`:
```python
import torch
import sys
sys.path.insert(0, "/home/shadeform/flash-attention-5")
from common.quantization import quantize_dequantize, PRECISION_BITS


def test_fp16_is_lossless_for_fp16_input():
    """FP16 roundtrip on FP16-range values should be near-lossless."""
    x = torch.randn(128, 64, dtype=torch.float32)
    x_rt = quantize_dequantize(x, "fp16")
    assert torch.allclose(x, x_rt, atol=1e-3)


def test_int8_roundtrip_preserves_sign():
    """INT8 quantization must preserve sign of all elements."""
    x = torch.randn(128, 64, dtype=torch.float32)
    x_rt = quantize_dequantize(x, "int8")
    assert (x.sign() == x_rt.sign()).all() or x_rt[x == 0].abs().max() < 1e-6


def test_int4_has_at_most_15_unique_values():
    """INT4 symmetric: values in {-7,...,0,...,7}, so at most 15 unique per block."""
    x = torch.randn(128, 64, dtype=torch.float32)
    x_rt = quantize_dequantize(x, "int4")
    unique_ratios = x_rt.unique().numel()
    assert unique_ratios <= 15


def test_fp8_e4m3_clamps_outliers():
    """FP8 e4m3 has max ~448. Large values must be clamped."""
    x = torch.tensor([0.1, 1.0, 100.0, 1000.0, 10000.0], dtype=torch.float32)
    x_rt = quantize_dequantize(x, "fp8_e4m3")
    assert x_rt.max() <= 500.0


def test_fp4_extreme_compression():
    """FP4 should still preserve rough magnitude ordering."""
    x = torch.tensor([0.01, 0.1, 1.0, 10.0], dtype=torch.float32)
    x_rt = quantize_dequantize(x, "fp4")
    assert (x_rt[1:] >= x_rt[:-1]).all(), "FP4 should preserve ordering"


def test_precision_bits_map():
    """PRECISION_BITS must map all supported precisions to their bit width."""
    assert PRECISION_BITS["fp16"] == 16
    assert PRECISION_BITS["fp8_e4m3"] == 8
    assert PRECISION_BITS["fp8_e5m2"] == 8
    assert PRECISION_BITS["int8"] == 8
    assert PRECISION_BITS["fp4"] == 4
    assert PRECISION_BITS["int4"] == 4


def test_quantize_dequantize_rejects_unknown():
    """Unknown precision string should raise ValueError."""
    x = torch.randn(16, 16, dtype=torch.float32)
    try:
        quantize_dequantize(x, "bfloat3")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_fp16_is_lossless_for_fp16_input()
    test_int8_roundtrip_preserves_sign()
    test_int4_has_at_most_15_unique_values()
    test_fp8_e4m3_clamps_outliers()
    test_fp4_extreme_compression()
    test_precision_bits_map()
    test_quantize_dequantize_rejects_unknown()
    print("All quantization tests passed.")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/shadeform/flash-attention-5
python -m pytest tests/test_quantization.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: Implement quantization primitives**

`common/quantization.py`:
```python
import torch

PRECISION_BITS = {
    "fp16": 16,
    "fp8_e4m3": 8,
    "fp8_e5m2": 8,
    "int8": 8,
    "fp4": 4,
    "int4": 4,
}

# FP8 E4M3: 1 sign + 4 exponent + 3 mantissa, max ≈ 448, min subnormal ≈ 2^-9
_FP8_E4M3_MAX = 448.0
_FP8_E4M3_MIN = 1.0 / 512.0

# FP8 E5M2: 1 sign + 5 exponent + 2 mantissa, max ≈ 57344, min subnormal ≈ 2^-16
_FP8_E5M2_MAX = 57344.0
_FP8_E5M2_MIN = 1.0 / 65536.0


def _simulate_fp_roundtrip(x, max_val, min_val, mantissa_bits):
    """Simulate floating-point quantization by rounding mantissa."""
    x_clamped = x.clamp(-max_val, max_val)
    sign = x_clamped.sign()
    abs_x = x_clamped.abs().clamp(min=min_val)
    log2_x = torch.log2(abs_x)
    exponent = torch.floor(log2_x)
    mantissa = abs_x / (2.0 ** exponent)
    quant_steps = 2 ** mantissa_bits
    mantissa_q = torch.round(mantissa * quant_steps) / quant_steps
    result = sign * mantissa_q * (2.0 ** exponent)
    result[x == 0] = 0.0
    return result


def _symmetric_int_roundtrip(x, num_levels):
    """Symmetric integer quantization with per-tensor scale."""
    max_val = x.abs().max()
    if max_val == 0:
        return x.clone()
    scale = max_val / num_levels
    x_q = torch.round(x / scale).clamp(-num_levels, num_levels)
    return x_q * scale


def quantize_dequantize(x, precision):
    """Quantize then dequantize tensor to simulate precision loss.

    Args:
        x: input tensor (float32)
        precision: one of "fp16", "fp8_e4m3", "fp8_e5m2", "int8", "fp4", "int4"

    Returns:
        tensor with same shape/dtype as x, but with quantization noise applied
    """
    if precision not in PRECISION_BITS:
        raise ValueError(f"Unknown precision: {precision}. Must be one of {list(PRECISION_BITS.keys())}")

    if precision == "fp16":
        return x.to(torch.float16).to(x.dtype)

    if precision == "fp8_e4m3":
        return _simulate_fp_roundtrip(x, _FP8_E4M3_MAX, _FP8_E4M3_MIN, mantissa_bits=3)

    if precision == "fp8_e5m2":
        return _simulate_fp_roundtrip(x, _FP8_E5M2_MAX, _FP8_E5M2_MIN, mantissa_bits=2)

    if precision == "int8":
        return _symmetric_int_roundtrip(x, 127)

    if precision == "fp4":
        return _simulate_fp_roundtrip(x, 6.0, 0.0625, mantissa_bits=1)

    if precision == "int4":
        return _symmetric_int_roundtrip(x, 7)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/shadeform/flash-attention-5
python -m pytest tests/test_quantization.py -v
```

Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/shadeform/flash-attention-5
git add common/quantization.py tests/test_quantization.py
git commit -m "feat: simulated quantization for 6 precision levels"
```

---

### Task 4: Block Statistics Extraction

**Files:**
- Create: `flash-attention-5/common/block_stats.py`
- Create: `flash-attention-5/tests/test_block_stats.py`

- [ ] **Step 1: Write failing test**

`tests/test_block_stats.py`:
```python
import torch
import math
import sys
sys.path.insert(0, "/home/shadeform/flash-attention-5")
from common.block_stats import compute_block_stats


def test_block_stats_keys():
    """Block stats dict must contain all required keys."""
    torch.manual_seed(42)
    B, H, N, d = 1, 1, 256, 64
    Q = torch.randn(B, H, N, d, dtype=torch.float32)
    K = torch.randn(B, H, N, d, dtype=torch.float32)
    V = torch.randn(B, H, N, d, dtype=torch.float32)

    stats_grid = compute_block_stats(Q, K, V, block_size=128)
    assert len(stats_grid) == 2  # T_r = 256/128 = 2
    assert len(stats_grid[0]) == 2  # T_c = 256/128 = 2

    s = stats_grid[0][0]
    required_keys = [
        "q_norm", "k_norm", "v_norm",
        "score_max", "score_mean", "score_var",
        "entropy", "causal_dist", "block_row", "block_col",
        "has_outlier", "seq_len", "head_dim", "num_heads",
    ]
    for key in required_keys:
        assert key in s, f"Missing key: {key}"


def test_block_stats_dimensions():
    """Grid dimensions must match ceil(N/block_size)."""
    torch.manual_seed(42)
    B, H, N, d = 2, 4, 384, 128
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)

    stats = compute_block_stats(Q, K, V, block_size=128)
    assert len(stats) == 3  # ceil(384/128) = 3
    assert len(stats[0]) == 3


def test_outlier_detection():
    """Injected outlier should be detected."""
    torch.manual_seed(42)
    B, H, N, d = 1, 1, 128, 64
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)
    Q[0, 0, 0, 0] = 500.0  # extreme outlier

    stats = compute_block_stats(Q, K, V, block_size=128)
    assert stats[0][0]["has_outlier"] is True


def test_entropy_is_positive():
    """Softmax entropy must be non-negative."""
    torch.manual_seed(42)
    B, H, N, d = 1, 1, 256, 64
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)

    stats = compute_block_stats(Q, K, V, block_size=128)
    for row in stats:
        for s in row:
            assert s["entropy"] >= 0.0


if __name__ == "__main__":
    test_block_stats_keys()
    test_block_stats_dimensions()
    test_outlier_detection()
    test_entropy_is_positive()
    print("All block stats tests passed.")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/shadeform/flash-attention-5
python -m pytest tests/test_block_stats.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: Implement block stats extraction**

`common/block_stats.py`:
```python
import torch
import math


def compute_block_stats(Q, K, V, block_size=128):
    """Extract per-block statistics for precision policy decisions.

    Computes statistics block-by-block without materializing the full N×N matrix.
    Averages across batch and head dimensions for each (block_row, block_col) position.

    Args:
        Q, K, V: (B, H, N, d) tensors
        block_size: tile size

    Returns:
        List of lists: stats_grid[i][j] is a dict of statistics for block (i, j)
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    Br = Bc = block_size
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)

    stats_grid = []

    for i in range(Tr):
        row_stats = []
        q_start = i * Br
        q_end = min(q_start + Br, N)
        Q_block = Q[:, :, q_start:q_end, :]

        q_norm = Q_block.float().norm() / math.sqrt(Q_block.numel())

        for j in range(Tc):
            k_start = j * Bc
            k_end = min(k_start + Bc, N)
            K_block = K[:, :, k_start:k_end, :]
            V_block = V[:, :, k_start:k_end, :]

            k_norm = K_block.float().norm() / math.sqrt(K_block.numel())
            v_norm = V_block.float().norm() / math.sqrt(V_block.numel())

            S_block = torch.matmul(Q_block.float(), K_block.float().transpose(-2, -1)) * scale

            score_max = S_block.max().item()
            score_mean = S_block.mean().item()
            score_var = S_block.var().item()

            abs_mean = S_block.abs().mean().item()
            has_outlier = bool(S_block.abs().max().item() > 6 * abs_mean) if abs_mean > 0 else False

            P_block = torch.softmax(S_block, dim=-1)
            log_P = torch.log(P_block + 1e-10)
            entropy_per_row = -(P_block * log_P).sum(dim=-1)
            entropy = entropy_per_row.mean().item()

            q_center = (q_start + q_end) / 2.0
            k_center = (k_start + k_end) / 2.0
            causal_dist = int(abs(q_center - k_center))

            row_stats.append({
                "q_norm": q_norm.item(),
                "k_norm": k_norm.item(),
                "v_norm": v_norm.item(),
                "score_max": score_max,
                "score_mean": score_mean,
                "score_var": score_var,
                "entropy": entropy,
                "causal_dist": causal_dist,
                "block_row": i,
                "block_col": j,
                "has_outlier": has_outlier,
                "seq_len": N,
                "head_dim": d,
                "num_heads": H,
            })

        stats_grid.append(row_stats)

    return stats_grid
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/shadeform/flash-attention-5
python -m pytest tests/test_block_stats.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/shadeform/flash-attention-5
git add common/block_stats.py tests/test_block_stats.py
git commit -m "feat: per-block statistics extraction for precision policy"
```

---

### Task 5: Workload Generator

**Files:**
- Create: `flash-attention-5/common/workloads.py`
- Create: `flash-attention-5/tests/test_workloads.py`

- [ ] **Step 1: Write failing test**

`tests/test_workloads.py`:
```python
import torch
import sys
sys.path.insert(0, "/home/shadeform/flash-attention-5")
from common.workloads import generate_workload, ALL_WORKLOADS


def test_all_workloads_count():
    """Should have 12 workload configurations."""
    assert len(ALL_WORKLOADS) == 12


def test_generate_workload_shapes():
    """Generated tensors must have correct shapes."""
    cfg = ALL_WORKLOADS[0]
    Q, K, V = generate_workload(cfg, seed=42)
    assert Q.shape == (cfg["batch"], cfg["num_heads"], cfg["seq_len"], cfg["head_dim"])
    assert Q.dtype == torch.float32


def test_outlier_workload_has_large_values():
    """Outlier workloads should contain values >> 10."""
    outlier_cfgs = [c for c in ALL_WORKLOADS if c["outliers"]]
    assert len(outlier_cfgs) > 0
    Q, K, V = generate_workload(outlier_cfgs[0], seed=42)
    assert Q.abs().max() > 10.0


def test_no_outlier_workload_is_standard_normal():
    """Non-outlier workloads should have max values roughly in normal range."""
    clean_cfgs = [c for c in ALL_WORKLOADS if not c["outliers"]]
    Q, K, V = generate_workload(clean_cfgs[0], seed=42)
    assert Q.abs().max() < 10.0


def test_deterministic_with_seed():
    """Same seed should produce identical workloads."""
    cfg = ALL_WORKLOADS[0]
    Q1, K1, V1 = generate_workload(cfg, seed=123)
    Q2, K2, V2 = generate_workload(cfg, seed=123)
    assert torch.equal(Q1, Q2)


if __name__ == "__main__":
    test_all_workloads_count()
    test_generate_workload_shapes()
    test_outlier_workload_has_large_values()
    test_no_outlier_workload_is_standard_normal()
    test_deterministic_with_seed()
    print("All workload tests passed.")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/shadeform/flash-attention-5
python -m pytest tests/test_workloads.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement workload generator**

`common/workloads.py`:
```python
import torch

ALL_WORKLOADS = [
    {"seq_len": 512,  "head_dim": 64,  "causal": False, "outliers": False, "batch": 2, "num_heads": 8},
    {"seq_len": 512,  "head_dim": 128, "causal": True,  "outliers": True,  "batch": 2, "num_heads": 8},
    {"seq_len": 2048, "head_dim": 64,  "causal": False, "outliers": True,  "batch": 2, "num_heads": 8},
    {"seq_len": 2048, "head_dim": 128, "causal": True,  "outliers": False, "batch": 2, "num_heads": 8},
    {"seq_len": 8192, "head_dim": 64,  "causal": True,  "outliers": False, "batch": 1, "num_heads": 4},
    {"seq_len": 8192, "head_dim": 128, "causal": False, "outliers": True,  "batch": 1, "num_heads": 4},
    {"seq_len": 2048, "head_dim": 64,  "causal": True,  "outliers": True,  "batch": 2, "num_heads": 8},
    {"seq_len": 2048, "head_dim": 128, "causal": False, "outliers": False, "batch": 2, "num_heads": 8},
    {"seq_len": 4096, "head_dim": 64,  "causal": False, "outliers": False, "batch": 2, "num_heads": 8},
    {"seq_len": 4096, "head_dim": 128, "causal": True,  "outliers": True,  "batch": 1, "num_heads": 8},
    {"seq_len": 8192, "head_dim": 64,  "causal": False, "outliers": False, "batch": 1, "num_heads": 4},
    {"seq_len": 8192, "head_dim": 128, "causal": True,  "outliers": False, "batch": 1, "num_heads": 4},
]


def generate_workload(config, seed=42):
    """Generate Q, K, V tensors for an attention workload.

    Args:
        config: dict with keys seq_len, head_dim, batch, num_heads, outliers
        seed: random seed for reproducibility

    Returns:
        (Q, K, V) tuple of float32 tensors with shape (batch, num_heads, seq_len, head_dim)
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    B = config["batch"]
    H = config["num_heads"]
    N = config["seq_len"]
    d = config["head_dim"]
    shape = (B, H, N, d)

    Q = torch.randn(shape, generator=gen, dtype=torch.float32)
    K = torch.randn(shape, generator=gen, dtype=torch.float32)
    V = torch.randn(shape, generator=gen, dtype=torch.float32)

    if config["outliers"]:
        # 0.1% of entries drawn from N(0, 100) — matches FA-3 paper setup
        for tensor in [Q, K, V]:
            mask = torch.bernoulli(torch.full(shape, 0.001), generator=gen).bool()
            outlier_vals = torch.randn(shape, generator=gen, dtype=torch.float32) * 100.0
            tensor[mask] = outlier_vals[mask]

    return Q, K, V
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/shadeform/flash-attention-5
python -m pytest tests/test_workloads.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/shadeform/flash-attention-5
git add common/workloads.py tests/test_workloads.py
git commit -m "feat: 12 standard attention workloads with outlier injection"
```

---

### Task 6: Mixed-Precision Attention + End-to-End Test

**Files:**
- Create: `flash-attention-5/tests/test_mixed_precision_attention.py`

This test validates the full pipeline: reference attention → block stats → policy → quantized attention. No new source files — it exercises the components built in Tasks 2-5 together.

- [ ] **Step 1: Write test**

`tests/test_mixed_precision_attention.py`:
```python
import torch
import math
import sys
sys.path.insert(0, "/home/shadeform/flash-attention-5")
from common.reference import reference_attention
from common.quantization import quantize_dequantize, PRECISION_BITS
from common.block_stats import compute_block_stats
from common.workloads import generate_workload, ALL_WORKLOADS


def mixed_precision_attention(Q, K, V, policy_fn, causal=False, block_size=128):
    """Attention with per-block precision selection."""
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    Br = Bc = block_size
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)

    stats_grid = compute_block_stats(Q, K, V, block_size=block_size)
    total_bits = 0
    total_elements = 0

    O = torch.zeros(B, H, N, d, dtype=torch.float32, device=Q.device)
    M = torch.full((B, H, N, 1), float("-inf"), dtype=torch.float32, device=Q.device)
    L = torch.zeros(B, H, N, 1, dtype=torch.float32, device=Q.device)

    for i in range(Tr):
        q_s = i * Br
        q_e = min(q_s + Br, N)
        Q_block = Q[:, :, q_s:q_e, :].float()

        j_end = Tc if not causal else min(i + 1, Tc)
        for j in range(j_end):
            k_s = j * Bc
            k_e = min(k_s + Bc, N)
            K_block = K[:, :, k_s:k_e, :].float()
            V_block = V[:, :, k_s:k_e, :].float()

            prec = policy_fn(stats_grid[i][j])
            Q_q = quantize_dequantize(Q_block, prec)
            K_q = quantize_dequantize(K_block, prec)
            V_q = quantize_dequantize(V_block, prec)

            S = torch.matmul(Q_q, K_q.transpose(-2, -1)) * scale
            if causal:
                q_idx = torch.arange(q_s, q_e, device=Q.device).unsqueeze(1)
                k_idx = torch.arange(k_s, k_e, device=Q.device).unsqueeze(0)
                S.masked_fill_(q_idx < k_idx, float("-inf"))

            m_old = M[:, :, q_s:q_e, :]
            m_new = torch.max(S, dim=-1, keepdim=True).values
            m_cur = torch.maximum(m_old, m_new)

            exp_s = torch.exp(S - m_cur)
            exp_old = torch.exp(m_old - m_cur)

            L[:, :, q_s:q_e, :] = exp_old * L[:, :, q_s:q_e, :] + exp_s.sum(dim=-1, keepdim=True)
            O[:, :, q_s:q_e, :] = exp_old * O[:, :, q_s:q_e, :] + torch.matmul(exp_s, V_q)
            M[:, :, q_s:q_e, :] = m_cur

            bits = PRECISION_BITS[prec]
            total_bits += bits * (q_e - q_s) * (k_e - k_s)
            total_elements += (q_e - q_s) * (k_e - k_s)

    O = O / L
    avg_bits = total_bits / max(total_elements, 1)
    return O, avg_bits


def test_all_fp16_matches_reference():
    """All-FP16 policy should closely match reference."""
    cfg = ALL_WORKLOADS[0]
    Q, K, V = generate_workload(cfg, seed=42)
    O_ref = reference_attention(Q.double(), K.double(), V.double(), causal=cfg["causal"])
    O_mp, avg_bits = mixed_precision_attention(Q, K, V, lambda s: "fp16", causal=cfg["causal"])
    assert avg_bits == 16.0
    rmse = (O_mp.double() - O_ref).pow(2).mean().sqrt().item()
    assert rmse < 1e-3, f"FP16 RMSE too high: {rmse}"


def test_all_int4_has_high_compression():
    """All-INT4 policy should achieve 4 bits average."""
    cfg = ALL_WORKLOADS[0]
    Q, K, V = generate_workload(cfg, seed=42)
    O_mp, avg_bits = mixed_precision_attention(Q, K, V, lambda s: "int4", causal=cfg["causal"])
    assert avg_bits == 4.0
    assert not torch.isnan(O_mp).any()


def test_mixed_policy_between_extremes():
    """A mixed policy should have avg_bits between 4 and 16."""
    def mixed_policy(stats):
        if stats["has_outlier"]:
            return "fp16"
        if stats["causal_dist"] > 512:
            return "int4"
        return "fp8_e4m3"

    cfg = ALL_WORKLOADS[2]  # 2048, outliers=True
    Q, K, V = generate_workload(cfg, seed=42)
    O_mp, avg_bits = mixed_precision_attention(Q, K, V, mixed_policy, causal=cfg["causal"])
    assert 4.0 < avg_bits < 16.0, f"avg_bits={avg_bits} not mixed"
    assert not torch.isnan(O_mp).any()


if __name__ == "__main__":
    test_all_fp16_matches_reference()
    test_all_int4_has_high_compression()
    test_mixed_policy_between_extremes()
    print("All mixed-precision attention tests passed.")
```

- [ ] **Step 2: Run tests**

```bash
cd /home/shadeform/flash-attention-5
python -m pytest tests/test_mixed_precision_attention.py -v
```

Expected: 3 tests PASS (these import existing modules and define `mixed_precision_attention` inline)

- [ ] **Step 3: Add mixed_precision_attention to common/reference.py and update common/__init__.py**

Append `mixed_precision_attention` to `common/reference.py` so the evaluator can import it. Also populate `common/__init__.py` now that all modules exist.

`common/__init__.py`:
```python
from common.reference import reference_attention, mixed_precision_attention
from common.quantization import quantize_dequantize, PRECISION_BITS
from common.block_stats import compute_block_stats
from common.workloads import generate_workload, ALL_WORKLOADS
```

Append to the end of `common/reference.py`:

```python
def mixed_precision_attention(Q, K, V, policy_fn, causal=False, block_size=128):
    """Attention with per-block precision selection.

    Args:
        Q, K, V: (B, H, N, d) float32 tensors
        policy_fn: callable(block_stats_dict) -> precision string
        causal: apply causal mask
        block_size: tile size

    Returns:
        (O, avg_bits): output tensor and average bits per element used
    """
    from common.quantization import quantize_dequantize, PRECISION_BITS
    from common.block_stats import compute_block_stats

    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    Br = Bc = block_size
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)

    stats_grid = compute_block_stats(Q, K, V, block_size=block_size)
    total_bits = 0
    total_elements = 0

    O = torch.zeros(B, H, N, d, dtype=torch.float32, device=Q.device)
    M = torch.full((B, H, N, 1), float("-inf"), dtype=torch.float32, device=Q.device)
    L = torch.zeros(B, H, N, 1, dtype=torch.float32, device=Q.device)

    for i in range(Tr):
        q_s = i * Br
        q_e = min(q_s + Br, N)
        Q_block = Q[:, :, q_s:q_e, :].float()

        j_end = Tc if not causal else min(i + 1, Tc)
        for j in range(j_end):
            k_s = j * Bc
            k_e = min(k_s + Bc, N)
            K_block = K[:, :, k_s:k_e, :].float()
            V_block = V[:, :, k_s:k_e, :].float()

            prec = policy_fn(stats_grid[i][j])
            Q_q = quantize_dequantize(Q_block, prec)
            K_q = quantize_dequantize(K_block, prec)
            V_q = quantize_dequantize(V_block, prec)

            S = torch.matmul(Q_q, K_q.transpose(-2, -1)) * scale
            if causal:
                q_idx = torch.arange(q_s, q_e, device=Q.device).unsqueeze(1)
                k_idx = torch.arange(k_s, k_e, device=Q.device).unsqueeze(0)
                S.masked_fill_(q_idx < k_idx, float("-inf"))

            m_old = M[:, :, q_s:q_e, :]
            m_new = torch.max(S, dim=-1, keepdim=True).values
            m_cur = torch.maximum(m_old, m_new)

            exp_s = torch.exp(S - m_cur)
            exp_old = torch.exp(m_old - m_cur)

            L[:, :, q_s:q_e, :] = exp_old * L[:, :, q_s:q_e, :] + exp_s.sum(dim=-1, keepdim=True)
            O[:, :, q_s:q_e, :] = exp_old * O[:, :, q_s:q_e, :] + torch.matmul(exp_s, V_q)
            M[:, :, q_s:q_e, :] = m_cur

            bits = PRECISION_BITS[prec]
            total_bits += bits * (q_e - q_s) * (k_e - k_s)
            total_elements += (q_e - q_s) * (k_e - k_s)

    O = O / L
    avg_bits = total_bits / max(total_elements, 1)
    return O, avg_bits
```

- [ ] **Step 4: Commit**

```bash
cd /home/shadeform/flash-attention-5
git add common/reference.py common/__init__.py tests/test_mixed_precision_attention.py
git commit -m "feat: mixed-precision attention pipeline with end-to-end test"
```

---

### Task 7: Initial Policy (Evolvable File)

**Files:**
- Create: `flash-attention-5/phase1_policy/initial_policy.py`

- [ ] **Step 1: Create the evolvable policy file**

`phase1_policy/initial_policy.py`:
```python
# EVOLVE-BLOCK-START
def precision_policy(block_stats):
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


def get_policy():
    """Entry point used by the evaluator to retrieve the current policy function."""
    return precision_policy
```

- [ ] **Step 2: Verify it loads**

```bash
cd /home/shadeform/flash-attention-5
python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('policy', 'phase1_policy/initial_policy.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = mod.get_policy()
result = fn({'q_norm': 1.0, 'k_norm': 1.0, 'v_norm': 1.0, 'score_max': 5.0, 'score_mean': 0.0, 'score_var': 1.0, 'entropy': 3.5, 'causal_dist': 100, 'block_row': 0, 'block_col': 1, 'has_outlier': False, 'seq_len': 2048, 'head_dim': 128, 'num_heads': 8})
assert result == 'fp16', f'Expected fp16, got {result}'
print('Initial policy loads and returns fp16 — OK')
"
```

Expected: `Initial policy loads and returns fp16 — OK`

- [ ] **Step 3: Commit**

```bash
cd /home/shadeform/flash-attention-5
git add phase1_policy/initial_policy.py
git commit -m "feat: initial precision policy (uniform FP16 baseline)"
```

---

### Task 8: Phase 1 Evaluator

**Files:**
- Create: `flash-attention-5/phase1_policy/evaluator.py`
- Create: `flash-attention-5/tests/test_evaluator.py`

- [ ] **Step 1: Write failing test**

`tests/test_evaluator.py`:
```python
import sys
import os
sys.path.insert(0, "/home/shadeform/flash-attention-5")
sys.path.insert(0, "/home/shadeform/openevolve")
os.chdir("/home/shadeform/flash-attention-5")


def test_evaluator_on_initial_policy():
    """Evaluator should return valid scores for the baseline FP16 policy."""
    from phase1_policy.evaluator import evaluate
    result = evaluate("phase1_policy/initial_policy.py")
    m = result.metrics

    assert "combined_score" in m
    assert "accuracy_score" in m
    assert "compression_score" in m
    assert "reliability_score" in m

    # FP16 baseline: high accuracy, zero compression
    assert m["accuracy_score"] > 0.5, f"Accuracy too low: {m['accuracy_score']}"
    assert m["compression_score"] < 0.05, f"FP16 should have near-zero compression: {m['compression_score']}"
    assert m["reliability_score"] == 1.0, f"FP16 should never produce NaN"
    assert 0.0 < m["combined_score"] < 1.0


def test_evaluator_on_broken_policy():
    """Evaluator should handle a policy that returns invalid precision gracefully."""
    import tempfile
    code = '''
def precision_policy(block_stats):
    return "bfloat3"

def get_policy():
    return precision_policy
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(code)
        path = f.name

    from phase1_policy.evaluator import evaluate
    result = evaluate(path)
    assert result.metrics["reliability_score"] == 0.0
    os.unlink(path)


if __name__ == "__main__":
    test_evaluator_on_initial_policy()
    test_evaluator_on_broken_policy()
    print("All evaluator tests passed.")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/shadeform/flash-attention-5
PYTHONPATH="/home/shadeform/flash-attention-5:/home/shadeform/openevolve:$PYTHONPATH" \
python -m pytest tests/test_evaluator.py::test_evaluator_on_initial_policy -v --timeout=120
```

Expected: FAIL — module not found

- [ ] **Step 3: Implement evaluator**

`phase1_policy/evaluator.py`:
```python
"""
Evaluator for FlashAttention-5 precision policy evolution.

Scores a candidate policy on accuracy (RMSE vs FP64), compression (avg bits),
and reliability (no NaN/Inf outputs).
"""

import importlib.util
import sys
import os
import traceback
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.reference import reference_attention, mixed_precision_attention
from common.workloads import ALL_WORKLOADS, generate_workload
from openevolve.evaluation_result import EvaluationResult

EVAL_WORKLOADS = [
    ALL_WORKLOADS[0],   # 512, d64, no causal, no outliers
    ALL_WORKLOADS[1],   # 512, d128, causal, outliers
    ALL_WORKLOADS[2],   # 2048, d64, no causal, outliers
    ALL_WORKLOADS[3],   # 2048, d128, causal, no outliers
    ALL_WORKLOADS[6],   # 2048, d64, causal, outliers
    ALL_WORKLOADS[8],   # 4096, d64, no causal, no outliers
]


def evaluate(program_path):
    """Evaluate a precision policy program.

    Args:
        program_path: path to Python file containing get_policy() function

    Returns:
        EvaluationResult with metrics: accuracy_score, compression_score,
        reliability_score, combined_score
    """
    try:
        spec = importlib.util.spec_from_file_location("policy_module", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "get_policy"):
            return _error_result("Missing get_policy() function")

        policy_fn = module.get_policy()
    except Exception as e:
        return _error_result(f"Failed to load policy: {e}")

    accuracy_scores = []
    compression_scores = []
    successes = 0
    total = len(EVAL_WORKLOADS)
    details = []

    for idx, cfg in enumerate(EVAL_WORKLOADS):
        try:
            Q, K, V = generate_workload(cfg, seed=42 + idx)

            if torch.cuda.is_available():
                Q, K, V = Q.cuda(), K.cuda(), V.cuda()

            with torch.no_grad():
                O_ref = reference_attention(
                    Q.double(), K.double(), V.double(), causal=cfg["causal"]
                )
                O_mp, avg_bits = mixed_precision_attention(
                    Q, K, V, policy_fn, causal=cfg["causal"]
                )

            if torch.isnan(O_mp).any() or torch.isinf(O_mp).any():
                details.append(f"Workload {idx}: NaN/Inf in output")
                continue

            rmse = (O_mp.double() - O_ref).pow(2).mean().sqrt().item()
            acc = 1.0 / (1.0 + rmse * 5000)
            comp = (16.0 - avg_bits) / 14.0

            accuracy_scores.append(acc)
            compression_scores.append(comp)
            successes += 1
            details.append(
                f"Workload {idx} (N={cfg['seq_len']},d={cfg['head_dim']},"
                f"causal={cfg['causal']},outliers={cfg['outliers']}): "
                f"RMSE={rmse:.6f}, bits={avg_bits:.1f}, acc={acc:.4f}, comp={comp:.4f}"
            )

        except Exception as e:
            details.append(f"Workload {idx}: ERROR {e}")
            traceback.print_exc()
            continue

    if successes == 0:
        return _error_result("All workloads failed", details=details)

    accuracy_score = sum(accuracy_scores) / len(accuracy_scores)
    compression_score = sum(compression_scores) / len(compression_scores)
    reliability_score = successes / total

    combined_score = (
        0.50 * accuracy_score
        + 0.30 * compression_score
        + 0.20 * reliability_score
    )

    return EvaluationResult(
        metrics={
            "accuracy_score": round(accuracy_score, 6),
            "compression_score": round(compression_score, 6),
            "reliability_score": round(reliability_score, 6),
            "combined_score": round(combined_score, 6),
        },
        artifacts={
            "workload_details": "\n".join(details),
            "num_workloads_passed": str(successes),
        },
    )


def _error_result(msg, details=None):
    artifacts = {"error": msg}
    if details:
        artifacts["details"] = "\n".join(details)
    return EvaluationResult(
        metrics={
            "accuracy_score": 0.0,
            "compression_score": 0.0,
            "reliability_score": 0.0,
            "combined_score": 0.0,
            "error": 1.0,
        },
        artifacts=artifacts,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/shadeform/flash-attention-5
PYTHONPATH="/home/shadeform/flash-attention-5:/home/shadeform/openevolve:$PYTHONPATH" \
python -m pytest tests/test_evaluator.py -v --timeout=180
```

Expected: 2 tests PASS. The initial FP16 policy should score high on accuracy, zero on compression.

- [ ] **Step 5: Commit**

```bash
cd /home/shadeform/flash-attention-5
git add phase1_policy/evaluator.py tests/test_evaluator.py
git commit -m "feat: multi-objective evaluator for precision policy evolution"
```

---

### Task 9: OpenEvolve Config and Launch Script

**Files:**
- Create: `flash-attention-5/phase1_policy/config.yaml`
- Create: `flash-attention-5/run_phase1.sh`

- [ ] **Step 1: Create OpenEvolve config**

`phase1_policy/config.yaml`:
```yaml
max_iterations: 80
checkpoint_interval: 10
log_level: "INFO"

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
    optimization. You are evolving a precision_policy function that decides
    what precision level to use for each block of an attention computation.

    Available precisions: "fp16" (16 bits), "fp8_e4m3" (8 bits), "fp8_e5m2" (8 bits),
    "int8" (8 bits), "fp4" (4 bits), "int4" (4 bits).

    The policy receives per-block statistics including score magnitudes, entropy,
    outlier presence, and causal distance. The goal is to maximize a combined score:
    50% accuracy (RMSE vs FP64 reference), 30% compression (fewer bits = better),
    20% reliability (no NaN outputs).

    Key insights:
    - Blocks with high attention scores or outliers need high precision (fp16)
    - Blocks with high entropy (uniform attention) can be compressed (fp4/int4)
    - Blocks far from the causal diagonal carry less signal
    - The accumulator is always FP32, so per-block quantization errors partially average out
    - The baseline (all fp16) scores ~0.35 combined because compression is 0
    - A good policy targets combined_score > 0.65

    Be creative with the policy logic. Consider thresholds, multi-level decisions,
    and combinations of features. The function must be pure Python with no imports.

database:
  population_size: 50
  archive_size: 20
  num_islands: 3
  elite_selection_ratio: 0.2
  exploitation_ratio: 0.7
  similarity_threshold: 0.99

evaluator:
  timeout: 180
  cascade_thresholds: [0.3]
  parallel_evaluations: 1

diff_based_evolution: true
max_code_length: 10000
```

- [ ] **Step 2: Create launch script**

`run_phase1.sh`:
```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure OpenEvolve is importable
export PYTHONPATH="$SCRIPT_DIR:/home/shadeform/openevolve:$PYTHONPATH"

echo "=== FlashAttention-5 Phase 1: Precision Policy Evolution ==="
echo "Using OpenEvolve with Claude Code CLI backend"
echo "Workdir: $SCRIPT_DIR"
echo ""

python /home/shadeform/openevolve/openevolve-run.py \
    phase1_policy/initial_policy.py \
    phase1_policy/evaluator.py \
    --config phase1_policy/config.yaml \
    --iterations 80 \
    "$@"

echo ""
echo "=== Phase 1 complete ==="
echo "Best policy saved in checkpoints/ directory"
```

- [ ] **Step 3: Make launch script executable**

```bash
chmod +x /home/shadeform/flash-attention-5/run_phase1.sh
```

- [ ] **Step 4: Dry-run validation (load config, don't run evolution)**

```bash
cd /home/shadeform/flash-attention-5
PYTHONPATH="/home/shadeform/flash-attention-5:/home/shadeform/openevolve:$PYTHONPATH" \
python -c "
import yaml
with open('phase1_policy/config.yaml') as f:
    cfg = yaml.safe_load(f)
print('Config loaded OK')
print(f'  Provider: {cfg[\"llm\"][\"provider\"]}')
print(f'  Models: {[m[\"name\"] for m in cfg[\"llm\"][\"models\"]]}')
print(f'  Max iterations: {cfg[\"max_iterations\"]}')

from phase1_policy.evaluator import evaluate
result = evaluate('phase1_policy/initial_policy.py')
print(f'  Baseline combined_score: {result.metrics[\"combined_score\"]:.4f}')
print(f'  Baseline accuracy_score: {result.metrics[\"accuracy_score\"]:.4f}')
print(f'  Baseline compression_score: {result.metrics[\"compression_score\"]:.4f}')
"
```

Expected: Config loads, baseline scores print (accuracy ~0.7+, compression ~0.0, combined ~0.35-0.55).

- [ ] **Step 5: Commit**

```bash
cd /home/shadeform/flash-attention-5
git add phase1_policy/config.yaml run_phase1.sh
git commit -m "feat: OpenEvolve config and launch script for Phase 1 policy evolution"
```

---

### Task 10: Full Integration Test and Run

**Files:** No new files. This task validates the complete pipeline works end-to-end.

- [ ] **Step 1: Run all tests**

```bash
cd /home/shadeform/flash-attention-5
PYTHONPATH="/home/shadeform/flash-attention-5:/home/shadeform/openevolve:$PYTHONPATH" \
python -m pytest tests/ -v --timeout=180
```

Expected: All tests pass (reference, quantization, block_stats, workloads, mixed-precision, evaluator).

- [ ] **Step 2: Test with a hand-crafted improved policy**

Create a temporary test to verify the evaluator can distinguish good from bad policies:

```bash
cd /home/shadeform/flash-attention-5
PYTHONPATH="/home/shadeform/flash-attention-5:/home/shadeform/openevolve:$PYTHONPATH" \
python -c "
import tempfile, os
from phase1_policy.evaluator import evaluate

# Policy that uses FP8 for most blocks, FP16 for outliers
code = '''
def precision_policy(block_stats):
    if block_stats[\"has_outlier\"]:
        return \"fp16\"
    if block_stats[\"score_var\"] < 0.5 and block_stats[\"causal_dist\"] > 256:
        return \"int4\"
    if block_stats[\"entropy\"] > 4.0:
        return \"fp4\"
    if block_stats[\"score_max\"] > 5.0:
        return \"fp16\"
    return \"fp8_e4m3\"

def get_policy():
    return precision_policy
'''
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
    f.write(code)
    path = f.name

result_baseline = evaluate('phase1_policy/initial_policy.py')
result_improved = evaluate(path)
os.unlink(path)

print(f'Baseline combined: {result_baseline.metrics[\"combined_score\"]:.4f}')
print(f'Improved combined: {result_improved.metrics[\"combined_score\"]:.4f}')
print(f'Baseline compression: {result_baseline.metrics[\"compression_score\"]:.4f}')
print(f'Improved compression: {result_improved.metrics[\"compression_score\"]:.4f}')

assert result_improved.metrics['compression_score'] > result_baseline.metrics['compression_score'], \
    'Improved policy must compress more than baseline'
print('Integration test PASSED: improved policy has better compression')
"
```

Expected: Improved policy shows higher compression and potentially higher combined score than baseline.

- [ ] **Step 3: Final commit with updated __init__.py**

```bash
cd /home/shadeform/flash-attention-5
git add -A
git commit -m "feat: complete Phase 1 pipeline — ready for OpenEvolve evolution"
```

- [ ] **Step 4: Launch Phase 1 evolution**

```bash
cd /home/shadeform/flash-attention-5
./run_phase1.sh --iterations 10
```

Start with 10 iterations to verify the loop works end-to-end. If it succeeds, run the full 80:

```bash
./run_phase1.sh
```

---
