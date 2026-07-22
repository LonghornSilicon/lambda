"""Sparsity-controller pilot: cross-tabulate XAttention-style antidiagonal
sparsity scores against the precision controller's INT8/FP16 decisions on
the existing 143 replay tiles.

This is a methodology pilot, not a deployment-grade study:

  - Input is `rtl/tb/testvectors/scores.hex` — the same int8 tiles the
    precision controller already runs against. Mix of synthetic
    random tiles and boundary edge cases (see analysis/gen_rtl_testvectors.py).
  - We measure how an antidiagonal-sum proxy would rank tiles for
    skipping, then sweep the skip threshold τ.
  - For each τ we report: skip rate, the conditional INT8-vs-FP16 split
    among survivors, and how often a tile flagged "skip" looks like
    it actually carries low pre-softmax mass.

To validate the proxy itself you want a real-LLM trace pass (Qwen2 / Phi-2,
seq_len ≥ 512) where ground-truth tile mass is the post-softmax attention
weight sum. Layer that on top of this same harness once the trace
capture lands.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

# Allow running from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sw" / "reference_model"))

from precision_controller_ref import PrecisionController   # noqa: E402
from sparsity_controller_ref import (                       # noqa: E402
    SparsityController,
    SparsityControllerInfo,
)


# ---------------------------------------------------------------------------
# Constants (match the existing testvector generator)
# ---------------------------------------------------------------------------
BLOCK_M = 64
BLOCK_N = 64
N_TILE  = BLOCK_M * BLOCK_N
STRIDES = [1, 2, 4, 8, 16]
TAU_SWEEP = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

TESTVECTOR_DIR = REPO_ROOT / "rtl" / "tb" / "testvectors"
SCORES_HEX     = TESTVECTOR_DIR / "scores.hex"
EXPECTED_HEX   = TESTVECTOR_DIR / "expected.hex"
LABELS_TXT     = TESTVECTOR_DIR / "labels.txt"


# ---------------------------------------------------------------------------
# Testvector I/O
# ---------------------------------------------------------------------------
def load_tiles() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (tiles[T, N_TILE] int8, expected_fp16[T] bool, labels[T] str)."""
    if not SCORES_HEX.exists():
        raise FileNotFoundError(
            f"{SCORES_HEX} not found — run "
            f"`python analysis/gen_rtl_testvectors.py` first."
        )

    raw = SCORES_HEX.read_text().split()
    if len(raw) % N_TILE:
        raise ValueError(
            f"scores.hex has {len(raw)} bytes, not a multiple of {N_TILE}"
        )
    flat = np.array([int(x, 16) for x in raw], dtype=np.uint8)
    signed = flat.astype(np.int16)
    signed[signed >= 128] -= 256
    tiles = signed.astype(np.int8).reshape(-1, N_TILE)

    expected_fp16 = np.array(
        [int(x, 16) for x in EXPECTED_HEX.read_text().split()],
        dtype=np.int8,
    ).astype(bool)
    assert expected_fp16.shape[0] == tiles.shape[0]

    labels = [ln.strip() for ln in LABELS_TXT.read_text().splitlines() if ln.strip()]
    return tiles, expected_fp16, labels


# ---------------------------------------------------------------------------
# Per-tile statistics
# ---------------------------------------------------------------------------
def antidiag_indices(stride: int) -> np.ndarray:
    """Flat indices into a BLOCK_M x BLOCK_N tile that lie on a stride-S
    antidiagonal mask: ((i + j) & (stride-1)) == 0."""
    i, j = np.indices((BLOCK_M, BLOCK_N))
    mask = ((i + j) & (stride - 1)) == 0
    return np.flatnonzero(mask.flatten())


def tile_stats(tile: np.ndarray) -> dict:
    """Compute every per-tile statistic the analysis needs in one pass."""
    abs_s_2d = np.abs(tile.astype(np.int32))
    abs_s = abs_s_2d.flatten()
    out = {
        "max_abs":  int(abs_s.max()),
        "sum_abs":  int(abs_s.sum()),
        "mean_abs": float(abs_s.mean()),
    }
    for s in STRIDES:
        idx = antidiag_indices(s)
        sampled = abs_s[idx]
        out[f"antidiag_sum_s{s}"]  = int(sampled.sum())
        out[f"antidiag_mean_s{s}"] = float(sampled.mean())
        out[f"antidiag_samples_s{s}"] = int(idx.size)
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def precision_decision(tile: np.ndarray) -> bool:
    """Run the bit-exact precision controller and return d_fp16."""
    return PrecisionController().process_tile(tile.flatten().tolist())


def sweep_threshold(stats: list[dict], stride: int, tau: float,
                    precision_fp16: np.ndarray) -> dict:
    """For a given (stride, tau) sweep point report skip / route counts.

    The threshold τ is expressed as a multiple of the *mean* antidiag sum
    across the corpus. The hardware will store an absolute integer in
    `THRESHOLD_REG`; we use a relative τ here so the pilot is independent
    of the score-scale.
    """
    key = f"antidiag_sum_s{stride}"
    vals = np.array([s[key] for s in stats], dtype=np.float64)
    mean_v = float(vals.mean())
    thr = tau * mean_v
    skip = vals < thr

    n = len(stats)
    n_skip = int(skip.sum())
    # Among survivors, what does the precision controller route?
    survivors = ~skip
    surv_fp16 = int((survivors & precision_fp16).sum())
    surv_int8 = int((survivors & ~precision_fp16).sum())

    # Tile-mass leakage: total sum_abs that we throw away vs total in the corpus.
    sum_abs = np.array([s["sum_abs"] for s in stats], dtype=np.float64)
    mass_kept = float(sum_abs[survivors].sum())
    mass_total = float(sum_abs.sum())

    return {
        "stride":      stride,
        "tau":         tau,
        "threshold":   thr,
        "skip_rate":   n_skip / n,
        "n_skip":      n_skip,
        "n_compute":   n - n_skip,
        "survivors_int8": surv_int8,
        "survivors_fp16": surv_fp16,
        "mass_kept_frac": mass_kept / mass_total if mass_total > 0 else 1.0,
    }


def main() -> int:
    print("Sparsity-controller pilot")
    print("=========================")
    tiles, precision_fp16_label, labels = load_tiles()
    n_tiles = tiles.shape[0]
    print(f"Loaded {n_tiles} tiles from {SCORES_HEX.relative_to(REPO_ROOT)}")
    print()

    # ---- per-tile stats ----
    stats = [tile_stats(t.reshape(BLOCK_M, BLOCK_N)) for t in tiles]

    # ---- precision controller decisions (matches expected.hex) ----
    pc_fp16 = np.array(
        [precision_decision(t.reshape(BLOCK_M, BLOCK_N)) for t in tiles],
        dtype=bool,
    )
    agree = int((pc_fp16 == precision_fp16_label).sum())
    print(f"Precision controller agrees with expected.hex on {agree}/{n_tiles} tiles "
          f"(should be {n_tiles}/{n_tiles}).")
    print(f"  precision says FP16: {int(pc_fp16.sum())}  INT8: {int((~pc_fp16).sum())}")
    print()

    # ---- antidiag stride coverage check ----
    print("Antidiagonal coverage per stride (samples per 64x64 tile):")
    for s in STRIDES:
        n_samples = int(antidiag_indices(s).size)
        print(f"  stride={s:>2}  samples={n_samples:>5}   "
              f"({100.0 * n_samples / N_TILE:5.2f}% of tile)")
    print()

    # ---- correlation: antidiag sum vs full sum_abs (sanity) ----
    sum_all     = np.array([s["sum_abs"] for s in stats], dtype=np.float64)
    print("Correlation of antidiag sum with full sum(|S|):")
    for s in STRIDES:
        vec = np.array([st[f"antidiag_sum_s{s}"] for st in stats], dtype=np.float64)
        if vec.std() > 0 and sum_all.std() > 0:
            r = float(np.corrcoef(vec, sum_all)[0, 1])
        else:
            r = float("nan")
        print(f"  stride={s:>2}   pearson_r = {r:+.4f}")
    print()

    # ---- threshold sweep at stride=8 (XAttention default) ----
    print("Threshold sweep at stride=8 (skip if antidiag_sum < τ · mean):")
    print(f"  {'τ':>5}  {'skip%':>6}  {'#skip':>5}  {'#INT8':>5}  {'#FP16':>5}  "
          f"{'mass kept':>9}")
    sweep_results = []
    for tau in TAU_SWEEP:
        r = sweep_threshold(stats, stride=8, tau=tau, precision_fp16=pc_fp16)
        sweep_results.append(r)
        print(f"  {tau:>5.2f}  {100*r['skip_rate']:>6.1f}  {r['n_skip']:>5d}  "
              f"{r['survivors_int8']:>5d}  {r['survivors_fp16']:>5d}  "
              f"{100*r['mass_kept_frac']:>8.2f}%")
    print()

    # ---- where do the FP16-flagged tiles land in the antidiag distribution? ----
    print("FP16-flagged tiles vs INT8-flagged tiles, antidiag_sum_s8 distribution:")
    a8 = np.array([s["antidiag_sum_s8"] for s in stats], dtype=np.float64)
    for tag, mask in [("INT8 tiles", ~pc_fp16), ("FP16 tiles", pc_fp16)]:
        if mask.sum() == 0:
            print(f"  {tag}: (none)")
            continue
        vals = a8[mask]
        print(f"  {tag:>11}  n={int(mask.sum()):>3}   "
              f"min={vals.min():>7.0f}  "
              f"p25={np.percentile(vals,25):>7.0f}  "
              f"median={np.median(vals):>7.0f}  "
              f"p75={np.percentile(vals,75):>7.0f}  "
              f"max={vals.max():>7.0f}")
    print()

    # ---- caveats summary ----
    print("Caveats")
    print("-------")
    print("- These 143 tiles are 110 boundary + 33 random synthetic tiles, NOT a")
    print("  real attention trace. Skip rates here reflect tile-generator design")
    print("  choices, not real LLM sparsity. The next step is hooking this same")
    print("  harness to validate_real_llm_v2.py-style Qwen2/Phi-2 score capture.")
    print("- 'mass kept' is sum(|S|), which is a softmax-mass proxy. Real ground")
    print("  truth is sum of post-softmax attention weights per tile against the")
    print("  global online-softmax denominator (FlashAttention's m, ℓ state).")
    print("- The antidiagonal mask `(i+j) & (STRIDE-1) == 0` is one of several")
    print("  valid samplings. XAttention's paper uses a strided sampling of a")
    print("  single antidiagonal in a 128-block; the version here samples all")
    print("  diagonals where (i+j) is a STRIDE multiple in a 64-block. Either")
    print("  is a cheap proxy; the real choice should be picked by ablation on")
    print("  the real-LLM trace pass once that lands.")

    # ---- persist for downstream reporting ----
    out_path = REPO_ROOT / "analysis" / "sparsity_pilot_results.json"
    out = {
        "n_tiles": n_tiles,
        "stride_sweep": [
            {
                "stride": s,
                "samples_per_tile": int(antidiag_indices(s).size),
                "correlation_with_full_sum": float(np.corrcoef(
                    [st[f"antidiag_sum_s{s}"] for st in stats], sum_all
                )[0, 1]) if sum_all.std() > 0 else float("nan"),
            }
            for s in STRIDES
        ],
        "tau_sweep_stride8": sweep_results,
        "antidiag_s8_per_label": {
            "int8_tiles": a8[~pc_fp16].tolist(),
            "fp16_tiles": a8[pc_fp16].tolist(),
        },
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
