# Lambda — Paper

The seminal paper for the Lambda chip. IEEEtran conference format. Single author: Alan Schwartz (UT Austin / Longhorn Silicon).

## Files

- `lambda.tex` — main paper source (~9 pages compiled)
- `lambda.bib` — BibTeX bibliography (real citations only; arXiv-traceable)
- `Makefile` — `make` produces `lambda.pdf`

## Build

### Local (requires TeX Live / MacTeX / MiKTeX)

```bash
make            # build lambda.pdf
make view       # open it
make clean      # remove build artifacts
```

If you don't have `pdflatex` installed locally, on macOS:

```bash
brew install --cask mactex-no-gui
```

…or install [BasicTeX](https://www.tug.org/mactex/morepackages.html) (~100 MB).

### Overleaf (no local install)

1. Zip this `paper/` directory.
2. Upload to [Overleaf](https://www.overleaf.com/) as a new project ("Upload Project").
3. Set the compiler to `pdfLaTeX` (the default).
4. Compile.

## Structure

The paper is organized as IEEEtran conference format, 6–9 pages double-column:

| Section | Contents |
|---|---|
| Abstract | What Lambda is, three contributions, key numbers |
| 1. Introduction | The small-model (≤1.5 B, validated on Qwen2-1.5B) on-device deployment regime; the open-source gap; contributions |
| 2. Background | FlashAttention, PagedAttention, KV compression (KIVI/KVQuant), attention-weight eviction |
| 3. Architecture | Block-by-block: ACU (MatE/VecU/KVE ChannelQuant), MSC, LSU, **TIU**, HIF |
| 4. Dataflow & Quantization | Token-through-chip walk; adaptive precision — the per-tile FP16 P·V path in MatE |
| 5. Implementation | Process, area accounting, power, pre-RTL audit |
| 6. Performance | Decode tok/s vs model size; LPDDR5X-vs-LPDDR4X PHY tradeoff |
| 7. Discussion | What we excluded and why; Etched-patent IP clearance; risks |
| 8. Conclusion | |

## Figures (all TikZ, no external image files)

1. **Top-level architecture** (Fig. 1) — block diagram showing ACU/MSC/LSU/TIU/HIF + SRAM + LPDDR PHY
2. **Area accounting** (Fig. 2) — stacked-bar comparison of gross vs shrunk area budget
3. **Decode throughput vs model size** (Fig. 3) — LPDDR5X x16 (12 GB/s) vs LPDDR4X x16 (6 GB/s)

## Tables

1. **Per-block area, verification surface, risk** — Table I
2. **Lambda vs Apple ANE / Hexagon NPU / academic FPGA accelerators** — Table II

No bloat. Three figures, two tables. Each earns its slot in the narrative.

## Target venues

In rough order of fit:

1. **IEEE Micro** — long-form architecture deep dives; ideal for this kind of academic standalone-chip paper. 6–9 page sweet spot.
2. **MICRO** (IEEE/ACM International Symposium on Microarchitecture) — top-tier; would want a longer version with deeper microarchitecture detail.
3. **ISCA** — likewise top-tier; would need more emphasis on perf/area/power tradeoffs and a tape-out result.
4. **DAC** (Design Automation Conference) — student-track friendly; mini@sic-shuttle academic projects fit well here.
5. **HotChips** — student session; presentation-oriented.

## Citation correctness

Every reference in `lambda.bib` is a real, arXiv-traceable or proceedings-published source. The KV codec of record is **ChannelQuant**, whose load-bearing algorithmic citations are KIVI (Liu et al., ICML 2024) and KVQuant (Hooper et al., NeurIPS 2024); TurboQuant (arXiv 2504.19874) is retained as cited prior work only. The load-bearing online-softmax citation is the running-max/running-sum recurrence of Milakov & Gimelshein (arXiv 2018/1805.02867) — the algorithm VecU actually implements, identical across FlashAttention-1/2/3; the FlashAttention line (arXiv 2205.14135 / 2307.08691 / 2407.08608) is cited for the IO-aware tiling that popularized it, not as the origin of the recurrence, and FA-3's distinctive features (Hopper warp-specialized async, FP8) are explicitly *not* implemented here. vLLM PagedAttention (arXiv 2309.06180) is the load-bearing KV-management citation. The adaptive-precision-KV paper (arXiv 2604.04722) grounds the TIU design.

> Note: the KIVI bib entry's author list / arXiv ID and the KVQuant venue (NeurIPS 2024) use conservative published values; verify exact fields before camera-ready. Physical PPA numbers for the KVE (KV Cache Engine) at N16FFC are marked pending re-measurement for ChannelQuant in the paper.

## Companion repository

The paper cites and links to <https://github.com/LonghornSilicon/architecture>, which holds the canonical machine-readable spec (`arch.yml`), the audit log (`STATUS.md`), the literature deep-dive log (`docs/literature_audit.md`), the per-block HLS scaffolding (`src/`), and the visual floorplan (`floorplan.html`).
