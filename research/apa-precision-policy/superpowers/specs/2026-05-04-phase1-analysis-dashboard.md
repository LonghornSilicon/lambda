# Phase 1 Analysis Dashboard: "Most Attention Blocks Don't Need FP16"

## Goal

Package the Phase 1 evolved precision policy as a visual, demonstrable research artifact. A single Python script runs the analysis across all 12 workloads, generates figures, builds a static HTML dashboard, and serves it locally.

## Output

- `analysis/run_analysis.py` — single entry point: compute, plot, build HTML, serve
- `analysis/figures/` — generated PNGs
- `analysis/dashboard.html` — self-contained HTML with embedded base64 images

## What It Computes

Runs mixed-precision attention pipeline (from `common/`) across all 12 standard workloads with three strategies:

| Strategy | Description | Expected avg bits |
|----------|-------------|-------------------|
| Uniform FP16 | All blocks FP16 | 16 |
| Uniform INT8 | All blocks INT8 | 8 |
| Evolved policy | Phase 1 best: INT8 default, FP16 for outlier+peaked (entropy < 2.0) | ~12 |

For each strategy × workload, records:
- Per-block precision decisions (for heatmaps)
- RMSE vs FP64 reference output
- Average bits per element
- Per-block entropy values

## Dashboard Sections

### 1. Precision Heatmaps (Hero)

3×4 grid of heatmaps (one per workload). Each shows block_row × block_col, colored blue (INT8) or red (FP16). Title: workload params (seq_len, head_dim, causal, outliers).

### 2. Compression Summary

Horizontal bar chart: fraction of blocks at INT8 vs FP16 per workload. Callout number at top: "X% of blocks safely compressed to INT8" (average across all workloads).

### 3. Accuracy vs Baselines

Grouped bar chart: RMSE (vs FP64) for each strategy across workloads. Shows evolved policy ≈ FP16 accuracy, far better than uniform INT8 on outlier workloads.

### 4. Sequence Length Scaling

Line chart: x = seq_len, y = % blocks assigned INT8. One line per workload family (outlier vs non-outlier). Hypothesis: longer sequences → more compressible blocks.

### 5. Why It Works: Entropy Distribution

Two panels:
- Left: Histogram of block entropies across all workloads, vertical line at threshold 2.0.
- Right: Scatter of entropy vs per-block RMSE contribution. Shows low-entropy blocks would cause most error if downgraded.

## Serving

Starts `http.server` on port 3000 (configurable via `--port`). Prints URL. Ctrl+C to stop.

## Dependencies

- `matplotlib` (available via torch ecosystem)
- Python stdlib only (`http.server`, `base64`, `json`)
- No new dependencies

## Existing Code Used

- `common/workloads.py` — 12 standard attention workloads
- `common/reference.py` — FP64 tiled reference attention
- `common/quantization.py` — quantization primitives (FP16, INT8, etc.)
- `common/block_stats.py` — per-block statistics extraction
- `phase1_policy/openevolve_output/best/best_program.py` — evolved precision policy

## Runtime

Estimated ~2-3 minutes on A4000 (12 workloads × 3 strategies × ~5s each).
