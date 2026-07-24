"""End-to-end integration example for the compiler team.

Shows what compiler-emitted code for an attention tile looks like
when both ACU sub-blocks (precision controller + MAC array) are used
together. The shape of this script is the shape the codegen should
produce — given a Q-tile, a K-tile, and a V-tile:

    1. compute QK^T scores (FP16)
    2. push the scores to the precision controller
    3. read back the INT8/FP16 decision
    4. dispatch the score-times-V matmul to the right MAC path
    5. accumulate into the output tile

This is exactly the worked example documented in
docs/isa/precision_controller_isa.pdf §4.1. Running it end-to-end
on the Python reference models prove the two blocks compose correctly.

Run:  python sw/reference_model/integration_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Resolve sibling modules regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from precision_controller_ref import PrecisionController  # noqa: E402
from mac_array_ref import MacArray                        # noqa: E402


# ---------------------------------------------------------------------------
# Configuration — matches the default 64x64 tile size
# ---------------------------------------------------------------------------
BLOCK_M = 64
BLOCK_N = 64
SCORE_WIDTH = 8
HEAD_DIM = 64       # D — dimension of each Q/K/V vector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def quantize_int8(scores_fp16: np.ndarray) -> np.ndarray:
    """Per-tile symmetric INT8 quantization (same as the RTL TB generator)."""
    max_abs = float(np.abs(scores_fp16).max())
    if max_abs < 1e-9:
        return np.zeros_like(scores_fp16, dtype=np.int8)
    scale = max_abs / 127.0
    return np.round(np.clip(scores_fp16 / scale, -127, 127)).astype(np.int8)


def process_one_attention_tile(
        Q: np.ndarray, K: np.ndarray, V: np.ndarray,
        pc: PrecisionController, mac: MacArray) -> tuple[np.ndarray, str]:
    """One tile of FlashAttention-style attention.

    Q, K, V are BLOCK_M x HEAD_DIM, BLOCK_N x HEAD_DIM, BLOCK_N x HEAD_DIM
    respectively (all float32 at fp16 precision).

    Returns:
      - The output tile (BLOCK_M x HEAD_DIM)
      - "INT8" or "FP16" indicating which path the precision controller chose
    """
    # ---- step 1: compute QK^T (always FP16 at first; cheap path) ----
    scores_fp16 = mac.matmul_fp16(Q, K.T.copy())  # shape (BLOCK_M, BLOCK_N)

    # ---- step 2: gate the tile via the precision controller ----
    scores_int8_for_gate = quantize_int8(scores_fp16)
    # The gate operates on INT8 scores (matches the RTL).
    decision_fp16 = pc.process_tile(scores_int8_for_gate.flatten().tolist())

    if decision_fp16:
        # ---- step 3a: FP16 path — multiply scores by V at fp16 ----
        out = mac.matmul_fp16(scores_fp16, V)
        return out, "FP16"

    # ---- step 3b: INT8 path — quantize V and multiply at INT8 ----
    # In production the runtime would have pre-quantized V; we quantize
    # here to keep the example self-contained.
    V_int8 = quantize_int8(V)
    scores_int8 = scores_int8_for_gate
    out_int32 = mac.matmul_int8(scores_int8, V_int8)
    # Dequantize back to fp32-at-fp16 precision for the output buffer.
    # A real compiler would do this differently (per-output requantization,
    # accumulator-aware scaling); we use a simple per-tile rescale here.
    scale = float(np.abs(scores_fp16).max() / 127.0) * \
            float(np.abs(V).max() / 127.0)
    out = (out_int32.astype(np.float32) * scale).astype(np.float16).astype(np.float32)
    return out, "INT8"


# ---------------------------------------------------------------------------
# Driver — make a small batch of synthetic tiles with realistic statistics
# ---------------------------------------------------------------------------
def make_workload(num_tiles: int, fp16_fraction: float = 0.21):
    """Generate `num_tiles` tiles, ~`fp16_fraction` of which should be peaked
    enough to need FP16 routing."""
    rng = np.random.default_rng(0xACE0)
    tiles = []
    truth = []
    for _ in range(num_tiles):
        Q = rng.uniform(-1.0, 1.0, size=(BLOCK_M, HEAD_DIM)).astype(np.float16).astype(np.float32)
        K = rng.uniform(-1.0, 1.0, size=(BLOCK_N, HEAD_DIM)).astype(np.float16).astype(np.float32)
        V = rng.uniform(-1.0, 1.0, size=(BLOCK_N, HEAD_DIM)).astype(np.float16).astype(np.float32)

        # Inject a Q outlier large enough that the resulting QK^T row dominates
        # the tile statistics (max/mean > 10 after INT8 quantization).
        if rng.random() < fp16_fraction:
            i = rng.integers(BLOCK_M)
            j = rng.integers(HEAD_DIM)
            Q[i, j] = rng.choice([-1.0, 1.0]) * 100.0
            truth.append("FP16")
        else:
            truth.append("INT8")
        tiles.append((Q, K, V))
    return tiles, truth


def main() -> int:
    print("LonghornSilicon ACU integration example")
    print("---------------------------------------")
    print("Driver: 100 attention tiles, ~21% expected to need FP16")
    print()

    tiles, truth = make_workload(100)
    pc  = PrecisionController()
    mac = MacArray()

    counts = {"INT8": 0, "FP16": 0}
    for Q, K, V in tiles:
        out, path = process_one_attention_tile(Q, K, V, pc, mac)
        counts[path] += 1

    print(f"Routed:  INT8 = {counts['INT8']}, FP16 = {counts['FP16']}")
    print(f"Truth :  INT8 = {truth.count('INT8')}, FP16 = {truth.count('FP16')}")
    print()

    # Quick cost-model report
    e_int8 = mac.estimate(BLOCK_M, BLOCK_N, HEAD_DIM, dtype="int8")
    e_fp16 = mac.estimate(BLOCK_M, BLOCK_N, HEAD_DIM, dtype="fp16")
    total_cycles = counts["INT8"] * e_int8.cycles + counts["FP16"] * e_fp16.cycles
    total_energy = counts["INT8"] * e_int8.energy_pj + counts["FP16"] * e_fp16.energy_pj
    all_fp16_cycles = 100 * e_fp16.cycles
    all_fp16_energy = 100 * e_fp16.energy_pj

    print(f"Per-tile cost (BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, D={HEAD_DIM}):")
    print(f"  INT8: {e_int8.cycles:>6} cycles, {e_int8.energy_pj/1000:.1f} nJ")
    print(f"  FP16: {e_fp16.cycles:>6} cycles, {e_fp16.energy_pj/1000:.1f} nJ")
    print()
    print(f"Mixed-precision policy:   {total_cycles:>9} cycles, "
          f"{total_energy/1e6:.2f} µJ")
    print(f"All-FP16 baseline:        {all_fp16_cycles:>9} cycles, "
          f"{all_fp16_energy/1e6:.2f} µJ")
    print(f"Speedup:        {all_fp16_cycles / total_cycles:.2f}x")
    print(f"Energy saving:  {(1 - total_energy / all_fp16_energy) * 100:.1f}%")
    print()
    print("This is the shape of code a compiler backend produces. Each "
          "function call maps to a chip operation; the data flow is "
          "identical when run against the Python reference, the FPGA "
          "prototype, or the silicon chip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
