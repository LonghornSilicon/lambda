"""End-to-end example of using the precision-controller reference model
from a silicon-agnostic compiler backend's perspective.

This script imagines what compiler-emitted code looks like at three levels of
sophistication, all hitting the same Python model that's bit-exact with the
RTL (and eventually the chip):

    Level A — naive: process every tile through the gate, take the decision
    Level B — batched: do many tiles, accumulate INT8/FP16 statistics
    Level C — calibration: use the model to pick the right precision policy
              for a workload offline, then bake it into the kernel selection

If you're a compiler engineer evaluating an integration, this is the
fastest way to see exactly what surface area you're targeting.

Run:    python3 sw/reference_model/example_compiler_use.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from time import perf_counter

# Resolve the model regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from precision_controller_ref import (  # noqa: E402
    PrecisionController,
    PrecisionControllerInfo,
)


# ---------------------------------------------------------------------------
# Synthetic workload: pretend a compiler is processing attention tiles from
# a real model. We mix uniform tiles (should decide INT8) with spiky tiles
# (should decide FP16) at a realistic ratio.
# ---------------------------------------------------------------------------

def make_uniform_tile(rng: random.Random, info: PrecisionControllerInfo) -> list:
    """Background activity, no outliers — the typical attention tile."""
    return [rng.randint(-20, 20) for _ in range(info.n)]


def make_spiky_tile(rng: random.Random, info: PrecisionControllerInfo) -> list:
    """One token dominates — the kind of tile FP16 is needed for."""
    tile = [rng.randint(-3, 3) for _ in range(info.n)]
    # Inject one or two outliers.
    for _ in range(rng.randint(1, 2)):
        idx = rng.randint(0, info.n - 1)
        tile[idx] = rng.choice([-1, 1]) * rng.randint(60, 127)
    return tile


def make_realistic_workload(num_tiles: int, fp16_fraction: float = 0.21):
    """Return (tiles, ground_truth_should_be_fp16).

    fp16_fraction=0.21 mirrors what we measured on Qwen2/Phi-2 — about 21%
    of tiles are peaked enough to need FP16, 79% are uniform-enough for INT8.
    """
    rng = random.Random(42)
    info = PrecisionControllerInfo()
    tiles = []
    truth = []
    for _ in range(num_tiles):
        if rng.random() < fp16_fraction:
            tiles.append(make_spiky_tile(rng, info))
            truth.append(True)
        else:
            tiles.append(make_uniform_tile(rng, info))
            truth.append(False)
    return tiles, truth


# ---------------------------------------------------------------------------
# Level A — naive: one tile, one decision, one kernel choice
# ---------------------------------------------------------------------------

def level_a_naive(tiles):
    """The simplest possible integration. A compiler-emitted attention kernel
    calls into the precision-controller model for each tile, then dispatches
    to either the INT8 or FP16 inner-loop kernel based on the answer.
    """
    pc = PrecisionController()
    int8_count = 0
    fp16_count = 0
    for tile in tiles:
        decision = pc.process_tile(tile)
        if decision:
            # ... compiler emits FP16 attention path here ...
            fp16_count += 1
        else:
            # ... compiler emits INT8 attention path here ...
            int8_count += 1
    return int8_count, fp16_count


# ---------------------------------------------------------------------------
# Level B — batched: amortize the per-tile overhead in the compiler runtime
# ---------------------------------------------------------------------------

def level_b_batched(tiles):
    """Compiler emits a batched dispatch: collect decisions for a window of
    tiles, then issue all INT8 tiles to one kernel launch and all FP16 tiles
    to another. The model's `process_tiles` returns the decision list in one
    call, mirroring how a real driver would coalesce decision-FIFO pops.
    """
    pc = PrecisionController()
    decisions = pc.process_tiles(tiles)
    int8_indices = [i for i, d in enumerate(decisions) if not d]
    fp16_indices = [i for i, d in enumerate(decisions) if d]
    # ... compiler now batches the tiles by precision and dispatches ...
    return int8_indices, fp16_indices


# ---------------------------------------------------------------------------
# Level C — calibration: use the model offline to inform compile-time
# precision policy without running the gate at every tile in the hot path
# ---------------------------------------------------------------------------

def level_c_calibration(calibration_tiles, runtime_tiles):
    """If the compiler is told a workload's statistics are stable (e.g., a
    fixed model running on representative data), it can run the gate over a
    calibration set offline and emit a workload-specific policy: e.g.
    "always INT8 for layers 1..N-2, always FP16 for layer 0", skipping the
    runtime gate entirely.

    This is exactly the simplification path the paper documents (the
    per-layer threshold register approach captures ~97% of the gate's
    benefit at ~0% of the silicon cost).
    """
    pc = PrecisionController()
    cal_decisions = pc.process_tiles(calibration_tiles)
    fp16_rate = sum(cal_decisions) / len(cal_decisions)
    # Compiler now picks a precision allocation based on the rate.
    # Toy policy: if a workload is overwhelmingly INT8-safe, skip the
    # runtime gate altogether.
    if fp16_rate < 0.05:
        return "ALWAYS_INT8", fp16_rate
    if fp16_rate > 0.95:
        return "ALWAYS_FP16", fp16_rate
    return "USE_RUNTIME_GATE", fp16_rate


# ---------------------------------------------------------------------------
# Run all three levels and print a comparison
# ---------------------------------------------------------------------------

def main() -> int:
    NUM_TILES = 500
    tiles, truth = make_realistic_workload(NUM_TILES, fp16_fraction=0.21)
    true_fp16 = sum(truth)
    true_int8 = NUM_TILES - true_fp16

    print(f"Workload: {NUM_TILES} tiles "
          f"(~{true_int8} expected INT8, ~{true_fp16} expected FP16)\n")

    # Level A
    t0 = perf_counter()
    a_int8, a_fp16 = level_a_naive(tiles)
    a_time = (perf_counter() - t0) * 1000
    print(f"Level A (naive per-tile dispatch):")
    print(f"  INT8: {a_int8}, FP16: {a_fp16}  ({a_time:.1f} ms)")

    # Level B
    t0 = perf_counter()
    b_int8, b_fp16 = level_b_batched(tiles)
    b_time = (perf_counter() - t0) * 1000
    print(f"\nLevel B (batched index lists):")
    print(f"  INT8 tiles: {len(b_int8)}, FP16 tiles: {len(b_fp16)}  "
          f"({b_time:.1f} ms)")
    print(f"  First 8 INT8 indices: {b_int8[:8]}")
    print(f"  First 8 FP16 indices: {b_fp16[:8]}")

    # Level C
    cal_tiles = tiles[:50]
    runtime_tiles = tiles[50:]
    t0 = perf_counter()
    policy, rate = level_c_calibration(cal_tiles, runtime_tiles)
    c_time = (perf_counter() - t0) * 1000
    print(f"\nLevel C (offline calibration):")
    print(f"  Measured FP16 rate over 50 tiles: {rate:.2%}")
    print(f"  Compiler emits policy: {policy}  ({c_time:.1f} ms)")

    # Consistency check: A and B must agree on every decision.
    if a_int8 != len(b_int8) or a_fp16 != len(b_fp16):
        print("\nERROR: Level A and Level B disagreed!")
        return 1
    print("\nLevels A and B agree on every decision (as required).")

    # Bonus: throughput check. The model runs the gate on a small CPU at
    # millions of scores per second, so it's never the bottleneck during
    # compiler-side simulation.
    total_scores = NUM_TILES * PrecisionControllerInfo().n
    throughput = total_scores / (a_time / 1000) / 1e6
    print(f"\nModel throughput: {throughput:.1f} M scores/sec on this CPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
