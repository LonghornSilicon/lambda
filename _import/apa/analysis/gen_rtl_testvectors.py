"""Generate INT8-quantized score tiles + expected decisions for the RTL replay TB.

Outputs (read by rtl/tb/tb_realdata.sv via $readmemh):
  rtl/tb/testvectors/scores.hex     — NUM_TILES * 4096 lines, one byte each (int8 two's-complement)
  rtl/tb/testvectors/expected.hex   — NUM_TILES lines, '0' = INT8, '1' = FP16
  rtl/tb/testvectors/labels.txt     — human-readable summary (tile index, name, max, sum, decision)

Reference decision is computed from the *quantized* int8 tile using the exact
RTL integer formula (no float):

    max(|s|) * N  >  THRESHOLD * sum(|s|)   →  FP16
    otherwise                               →  INT8

Boundary cases at the (N=4096, T=10) decision edge:
  bg=1, spike=10  → INT8  (LHS = 10*4096 = 40960  <  RHS = 10*(4095+10) = 41050)
  bg=1, spike=11  → FP16  (LHS = 11*4096 = 45056  >  RHS = 10*(4095+11) = 41060)
"""

import os
import sys
import numpy as np
from pathlib import Path

BLOCK_M = 64
BLOCK_N = 64
N_TILE = BLOCK_M * BLOCK_N        # 4096
THRESHOLD = 10
INT8_MAX = 127                    # symmetric int8 range [-127, +127]

OUT_DIR = Path(__file__).resolve().parent.parent / "rtl" / "tb" / "testvectors"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)


def reference_decision(tile_i8: np.ndarray) -> int:
    """Bit-exact RTL reference: 1 = FP16, 0 = INT8."""
    abs_s = np.abs(tile_i8.astype(np.int64))
    max_v = int(abs_s.max())
    sum_v = int(abs_s.sum())
    return 1 if max_v * N_TILE > THRESHOLD * sum_v else 0


def quantize_int8(s_float: np.ndarray) -> np.ndarray:
    max_abs = float(np.abs(s_float).max())
    if max_abs < 1e-9:
        return np.zeros(s_float.shape, dtype=np.int8)
    scale = max_abs / INT8_MAX
    q = np.round(np.clip(s_float / scale, -INT8_MAX, INT8_MAX))
    return q.astype(np.int8)


def make_random_int8_tile() -> np.ndarray:
    s = rng.uniform(-3.0, 3.0, size=(BLOCK_M, BLOCK_N)).astype(np.float32)
    return quantize_int8(s)


def make_random_fp16_tile() -> np.ndarray:
    s = rng.uniform(-1.0, 1.0, size=(BLOCK_M, BLOCK_N)).astype(np.float32)
    n_spikes = int(rng.integers(1, 5))
    spike_val = float(rng.uniform(15.0, 60.0))
    for _ in range(n_spikes):
        i = int(rng.integers(0, BLOCK_M))
        j = int(rng.integers(0, BLOCK_N))
        s[i, j] = spike_val * float(rng.choice([-1.0, 1.0]))
    return quantize_int8(s)


def make_bg1_spike_tile(spike: int) -> np.ndarray:
    """Background of all +1 with a single spike of value `spike` at the middle."""
    flat = np.ones(N_TILE, dtype=np.int8)
    flat[N_TILE // 2] = np.int8(spike)
    return flat.reshape(BLOCK_M, BLOCK_N)


def append_tile_lines(tile_i8: np.ndarray, lines: list) -> None:
    flat = tile_i8.flatten()
    for v in flat:
        lines.append(f"{int(v) & 0xFF:02x}")


def main() -> int:
    boundary_tiles: list = []

    # The two canonical edge cases the user called out.
    t10 = make_bg1_spike_tile(10)   # LHS=40960 < RHS=41050 → INT8
    t11 = make_bg1_spike_tile(11)   # LHS=45056 > RHS=41060 → FP16
    assert reference_decision(t10) == 0, "spike=10 must be INT8"
    assert reference_decision(t11) == 1, "spike=11 must be FP16"
    boundary_tiles += [("bg1_spike10_INT8_edge", t10), ("bg1_spike11_FP16_edge", t11)]

    # Sweep around the edge from both signs (bg=+1, spike negative & positive).
    for sp in [-127, -50, -20, -12, -11, -10, -5, -1, 0, 5, 9, 12, 30, 50, 100, 127]:
        boundary_tiles.append((f"bg1_spike{sp:+d}", make_bg1_spike_tile(sp)))

    # Pathological constants
    z = np.zeros((BLOCK_M, BLOCK_N), dtype=np.int8)
    boundary_tiles.append(("all_zero_INT8", z))
    f = np.full((BLOCK_M, BLOCK_N), 127, dtype=np.int8)
    boundary_tiles.append(("all_p127_INT8", f))
    g = np.full((BLOCK_M, BLOCK_N), -127, dtype=np.int8)
    boundary_tiles.append(("all_n127_INT8", g))

    # Single-spike-in-zero-bg (definite FP16: anything nonzero with all else zero)
    s1 = np.zeros((BLOCK_M, BLOCK_N), dtype=np.int8)
    s1[0, 0] = 1
    boundary_tiles.append(("zero_bg_spike1_FP16", s1))
    s2 = np.zeros((BLOCK_M, BLOCK_N), dtype=np.int8)
    s2[BLOCK_M - 1, BLOCK_N - 1] = -1
    boundary_tiles.append(("zero_bg_spike_n1_last_FP16", s2))

    random_int8 = [("rnd_int8", make_random_int8_tile()) for _ in range(70)]
    random_fp16 = [("rnd_fp16", make_random_fp16_tile()) for _ in range(50)]
    random_pool = random_int8 + random_fp16
    rng.shuffle(random_pool)

    all_tiles = boundary_tiles + random_pool

    score_lines: list = []
    expected_lines: list = []
    label_lines: list = []
    n_fp16 = 0

    for idx, (name, tile) in enumerate(all_tiles):
        decision = reference_decision(tile)
        if decision == 1:
            n_fp16 += 1
        append_tile_lines(tile, score_lines)
        expected_lines.append(f"{decision:01x}")
        abs_t = np.abs(tile.astype(np.int64))
        label_lines.append(
            f"{idx:3d}  {name:30s}  max={int(abs_t.max()):3d}  "
            f"sum={int(abs_t.sum()):7d}  → {'FP16' if decision else 'INT8'}"
        )

    (OUT_DIR / "scores.hex").write_text("\n".join(score_lines) + "\n")
    (OUT_DIR / "expected.hex").write_text("\n".join(expected_lines) + "\n")
    (OUT_DIR / "labels.txt").write_text("\n".join(label_lines) + "\n")

    print(f"Wrote {len(all_tiles)} tiles ({len(score_lines)} score lines) → {OUT_DIR}")
    print(f"  FP16 expected: {n_fp16}")
    print(f"  INT8 expected: {len(all_tiles) - n_fp16}")
    print(f"  Boundary tiles (first {len(boundary_tiles)}):")
    for line in label_lines[:len(boundary_tiles)]:
        print(f"    {line}")
    return len(all_tiles)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
