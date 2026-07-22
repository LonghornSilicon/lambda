# Adaptive Precision Attention

> ⚙️ **You are on the `rtl` branch — the RTL home.** All SystemVerilog, testbenches,
> OpenLane, golden vectors, and the reference-model parity gate live here (not on `main`).
> **The RTL and its sign-off are 130nm (SkyWater Sky130)** — the best open-PDK proxy, used
> for 16nm *estimates* and realizability, not the 16nm design itself. TSMC 16nm is future.


> **Building a compiler / integrating this block?** Start with the chip-level
> [Compiler Programming Guide](https://github.com/LonghornSilicon/architecture/blob/main/docs/compiler_programming_guide.md)
> and the [documentation standard](https://github.com/LonghornSilicon/architecture/blob/main/docs/documentation_standard.md)
> in the `architecture` repo. This block's own interface spec is [`docs/isa/precision_controller_isa.md`](docs/isa/precision_controller_isa.md).

A hardware-verified extension of [FlashAttention](https://arxiv.org/abs/2205.14135)
that routes each attention tile to **INT8** or **FP16** at runtime based on
a single pre-softmax ratio check. ~79% of tiles end up on the cheaper
INT8 path with zero accuracy loss versus FlashAttention-2, giving a
theoretical **~1.7× KV-cache bandwidth improvement** for ~30 flip-flops
of additional silicon.
<!-- Derivations: 79%/21% is a conservative design blend across Qwen2+Phi-2
(sw/reference_model/example_compiler_use.py:59); measured per-model INT8-safe
rates are 91.7/97.1/96.6% (Qwen2) and 58.9% (Phi-2). Bandwidth: 0.79*0.5B +
0.21*1.0B = 0.605x avg -> 1/0.605 = 1.65x, rounded to ~1.7x. The 30 FFs are the
real Sky130 sign-off count (openlane/precision_controller/results/sky130_80MHz_signoff_metrics.json). -->

This is the **ACU (Attention Compute Unit)** block of the LonghornSilicon
LLM inference accelerator — one of four blocks targeting TSMC 16nm FinFET (N16FFC) tape-out.

📄 **Paper**: [`paper/adaptive_precision_attention.pdf`](paper/adaptive_precision_attention.pdf)
🧱 **Sky130 GDS** (signed off): [`openlane/precision_controller/results/precision_controller.gds`](openlane/precision_controller/results/precision_controller.gds)
🎨 **Layout**: [`paper/sky130_layout.png`](paper/sky130_layout.png)

---

## TL;DR

| | |
|---|---|
| **What** | A streaming precision controller that gates attention tiles INT8/FP16 |
| **Why** | Cuts KV-cache bandwidth ~1.7× with no accuracy loss vs FlashAttention-2 |
| **How** | Pre-softmax ratio test: `max(|S|)·N > 10·sum(|S|)` → no division, no transcendentals |
| **Hardware cost** | 30 flip-flops (real Sky130 sign-off; 3,438 µm² Sky130 stdcell). ~150 µm² in TSMC 16FFC — *16nm est.*, ASAP7-derived (`analysis/tsmc16_fit_report.md`); ~0.2% of an 8×8 INT8 MAC array (also a 16nm est.) |
| **Verified** | 253/253 testbench cases, signed off on Sky130 (DRC/LVS/antenna/IR-drop clean) |
| **Status** | Signed off on Sky130 (130nm flow). Full-chip tape-out target Summer 2027 (Lambda 16nm track) via TSMC University Program |

---

## How this improves on FlashAttention

### What FlashAttention already solved

FlashAttention reframed attention as a tiled, IO-aware operation: rather
than materialize the full `N×N` softmax matrix, it computes attention in
blocks small enough to fit in SRAM. FA-2 added better parallelism;
FA-3 added asynchronous execution and FP8 — but FP8 needs Hopper-class
Tensor Cores (H100).

### Where FA still leaves cycles on the table

Inside each tile, FA-2 keeps everything in FP16. But in practice the
attention score distribution within a tile is bimodal:

- **Uniform tiles** — attention spread evenly across many keys.
  Numerically tame. INT8 quantization preserves the result.
- **Peaked tiles** — one or two keys dominate. INT8 quantization sets the
  scale to the dominant value, rounding the small-but-nonzero values
  to zero and destroying the softmax.

The naive instinct is to compute entropy to detect peaked tiles. Entropy
needs softmax. Softmax needs exponentials. Exponentials are expensive
in hardware. We don't want any of that.

### Our contribution: an entropy-equivalent ratio test on raw scores

Empirically, peakedness is captured by a single ratio on the raw
(pre-softmax) scores:

```
            max(|S|)
   ratio = ─────────
            mean(|S|)
```

- Uniform tile: max ≈ mean → ratio ≈ 1
- Peaked tile (one outlier): max ≫ mean → ratio in the 100s–1000s

A threshold of 10 perfectly separates the two populations across 19,488
real attention tiles. The check rearranges to:

```
   max(|S|) · N  >  10 · sum(|S|)
```

— a left shift, a multiply-by-10 (= two shifts + an add), and one
comparator. Zero floating-point. Zero division. Decision in one cycle
after the last score of the tile streams in.

### Why this matters for the hardware budget

In TSMC 16FFC the controller is **~30 flip-flops, ~150 µm²** (*16nm est.*,
ASAP7-derived — `analysis/tsmc16_fit_report.md`; the real signed-off Sky130
stdcell area is 3,438 µm²), less than **0.2%** of a single 8×8 INT8 MAC array
(16FFC MAC ≈ 50k–100k µm², also estimated). The cost of *deciding* whether
to use INT8 is rounding error against the savings from *actually using*
INT8 on ~79% of tiles (conservative blend across Qwen2+Phi-2; measured
per-model 91.7/97.1/96.6% Qwen2, 58.9% Phi-2).

| | FlashAttention-2 | This work |
|---|---|---|
| Precision inside a tile | FP16 always | INT8 (79% of tiles) or FP16 (21%) |
| Decision logic | none | ratio gate (~30 FFs, 1-cycle latency) |
| KV-cache bandwidth per tile | 1.0× | 0.605× average |
| Effective KV-cache bandwidth | 1.0× | **~1.7×** |
| Hopper hardware required | no | no |
| Accuracy vs FP16 baseline | matched | matched |

### Why we can't show this speedup in software (yet)

Software simulation of "INT8 attention" on a GPU still uses the GPU's
FP16 Tensor Cores — the integer multiply is faked by quantize → cast →
FP16 multiply → dequantize. The actual speedup requires **physical
integer multipliers in silicon**, which is what we're building.

The current Triton kernel (`kernels/`) proves *correctness* across
real Qwen2 and Phi-2 traces. Throughput numbers come from the ASIC
or — sooner — the ZCU102/104 FPGA prototype.

---

## How this fits in LonghornSilicon

The precision controller is one of four blocks in the
LonghornSilicon LLM inference accelerator. The rest of the architecture
is in flight:

```
┌──────────────────────────────────────────────────────────────────────┐
│              LonghornSilicon LLM Inference Accelerator (16FFC)        │
│                                                                       │
│   ┌──────────────────┐         ┌──────────────────────────────┐       │
│   │  ACU             │ scores  │  Token Importance Unit       │       │
│   │  (this repo)     │────────▶│                              │       │
│   │                  │         │  Per-token attention weight  │       │
│   │  precision_      │         │  accumulator + comparator    │       │
│   │  controller      │         │  → keep / demote / evict     │       │
│   │  ────────────    │         └──────────┬───────────────────┘       │
│   │  INT8 vs FP16    │                    │ tier signals               │
│   │  gate per tile   │                    ▼                            │
│   │                  │         ┌──────────────────────────────┐       │
│   │  + INT8/FP16     │  K, V   │  KV Cache Engine             │       │
│   │  MAC array       │◀───────▶│                              │       │
│   │                  │         │  ChannelQuant compress       │       │
│   │                  │         │  (K per-channel INT4 /       │       │
│   └──────────────────┘         │  V per-token INT4 + FP16     │       │
│                                │  outlier lane) on writes,    │       │
│                                │  decompress on reads         │       │
│                                └──────────┬───────────────────┘       │
│                                           │                            │
│                            ┌──────────────┴──────────────┐             │
│                            ▼                             ▼             │
│              ┌─────────────────────────┐   ┌─────────────────────┐    │
│              │  Memory Hierarchy Ctrl. │   │  Off-chip LPDDR5X   │    │
│              │  (block 4)              │◀─▶│  (cold KV + model   │    │
│              │  0.8 MB on-die SRAM     │   │   weights)          │    │
│              │  + off-chip LPDDR5X     │   └─────────────────────┘    │
│              │  direct, no eDRAM tier  │                                │
│              └─────────────────────────┘                                │
└──────────────────────────────────────────────────────────────────────┘
```

| Block | This repo? | Role |
|---|---|---|
| **ACU (Attention Compute Unit)** | ✅ this repo | Decides INT8 vs FP16 per tile, runs the MAC array |
| **KV Cache Engine** | not yet | ChannelQuant compress on write, decompress on read (per-channel INT4 keys / per-token INT4 values + FP16 outlier lane) |
| **Token Importance Unit** | not yet | Tracks attention weight per cached token → mixed-precision retention (hot tokens stay high precision, cold tokens demoted or evicted) |
| **Memory Hierarchy Controller** | not yet | Routes between 0.8 MB on-die SRAM and off-chip LPDDR5X, direct (no separate eDRAM tier) |

The precision controller is the **first** to get verified through to
GDSII because it's the smallest, has the clearest spec, and unblocks
the area/timing budgets of every block that uses it.

---

## What's in this repo

```
attention-compute-unit/
├── analysis/                  # Python: evolutionary search, fixed-point sim, RTL test-vector gen
├── kernels/                   # Triton GPU kernel (correctness proof on real LLM traces)
├── rtl/
│   ├── precision_controller.sv     # The DUT (streaming, 1-cycle decision latency)
│   ├── tb/                          # Self-checking + replay testbenches (253 cases)
│   ├── constraints/timing.sdc       # 16FFC-targeted timing constraints
│   ├── genus.tcl, innovus.tcl       # Cadence flow (waiting on TSMC PDK)
│   └── Makefile                     # Targets: sim, sim_realdata, testvectors, synth
├── openlane/
│   └── precision_controller/        # LibreLane / OpenROAD flow targeting Sky130A
│       ├── config.json              # Signed-off config (80 MHz, all corners clean)
│       └── results/                 # GDS + layout PNG + signoff metrics
├── sw/                                       # Compiler co-design: software references
│   ├── README.md                             # Top-level orientation for the compiler team
│   └── reference_model/
│       ├── precision_controller_ref.{hpp,cpp,py}   # Block 1a, bit-exact 143/143 vs RTL
│       ├── mac_array_ref.{hpp,cpp,py}              # Block 1b, INT8 + FP16 matmul
│       ├── test_*.{cpp,py}                          # C++ and Python test suites
│       ├── integration_example.py                  # End-to-end attention tile flow
│       └── Makefile                                # test, test-all, shared, static, integration
├── paper/
│   ├── adaptive_precision_attention.tex/.pdf   # Full writeup
│   └── *.png                                    # Figures
├── docs/
│   ├── isa/precision_controller_isa.{tex,pdf}   # ISA spec for block 1 (pc-isa-0.1)
│   ├── mac_array_design.{tex,pdf}               # Design rationale for block 2 (mac-array-v0.1)
│   ├── sw_overview.{tex,pdf}                    # Compiler-team orientation PDF
│   ├── reference_model_api.{tex,pdf}            # Formal C++ / C / Python API reference
│   ├── meeting_handout.{tex,pdf}                # 2-page brief for compiler-team meetings
│   ├── chamber_setup.md                          # End-to-end walkthrough for Cadence chamber sign-off
│   ├── ci_setup.md                               # GitHub Actions self-hosted runner setup
│   ├── ci_overview.md                            # CI pipeline reference
│   └── new_block_blueprint.md                    # Template for future LonghornSilicon block repos
├── .github/workflows/ci.yml      # CI: RTL verif → synth → Sky130 signoff → ref tests → paper build
└── README.md (this file)
```

---

## Results

### Accuracy (algorithmic)

- **253/253 testbench cases pass** bit-exactly against a Python integer
  reference (110 directed = 10 hand-written + 100 LCG-random in
  `rtl/tb/tb_precision_controller.sv`; 143 replay from real attention-score
  traces in `rtl/tb/tb_realdata.sv`, NUM_TILES=143).
- **Matches FlashAttention-2 accuracy** on Qwen2-0.5B/1.5B/3B and Phi-2.
  INT8-safe tile fraction (`analysis/deep_layer_combined_stats.json`,
  `analysis/phi2_validation_stats.json`): **91.7% / 97.1% / 96.6%** on the
  three Qwen2 models (the "91–97%" band; INT8 fraction *improves* with scale)
  and **58.9%** on Phi-2 (MHA, lower INT8 coverage but accuracy still matched).
- Threshold-of-10 (`analysis/fixed_point_sim.py`, RATIO_THRESHOLD=10.0)
  robust to its exact value: the gap between uniform and peaked tiles is
  1.5 < ratio < 3.5 (nearly empty in the 19,488-tile eval corpus — see the
  bimodality figure in the paper).

### Hardware (post-PnR Sky130, signed off)

<!-- All values below are the REAL, measured Sky130 sign-off from
openlane/precision_controller/results/sky130_80MHz_signoff_metrics.json:
80 MHz = CLOCK_PERIOD 12.5 ns (config.json); FFs=30 (sequential_cell);
stdcell area=3438.3; die bbox 87.755×98.475, util 0.591; power__total=330.5 µW
(nom_tt); drc/lvs/antenna/power-grid violations all 0. -->

| Metric | Value |
|---|---|
| Frequency (signed off at SS / 100 °C / 1.60 V) | **80 MHz** |
| Setup WNS (FF / TT / SS) | +8.34 / +6.10 / **+0.072** ns |
| Hold WS (FF / TT / SS) | +0.18 / +0.36 / +0.86 ns |
| DRC / LVS / antenna / IR-drop violations | **0 / 0 / 0 / 0** |
| Flip-flops | 30 (matches closed-form `2·SCORE_WIDTH + log₂N + 2`) |
| Stdcell area | 3,438 µm² (Sky130A) |
| Die size | 87.8 × 98.5 µm² (59% utilization) |
| Total power (TT corner) | 330.5 µW |
| End-to-end flow runtime | ~3 min per pass |

Independently scaling the *measured* Sky130 sign-off (3,438 µm², 80 MHz,
330.5 µW) to TSMC 16FFC with published node ratios (~5× area, ~2× speed,
~10× dynamic power) gives **~700 µm², ~30 µW** (*16nm est.*). These are
projections, not measured 16nm results. NOTE: the ASAP7-derived projection
(`analysis/tsmc16_fit_report.md`) instead gives **~150 µm² / 800 MHz** — the
two area estimates differ ~5× and the frequency estimates (Sky130-scaled
~160 MHz vs ASAP7 ~800 MHz) do not reconcile; treat the 16nm speed/area as a
range pending the Cadence 16FFC run, not a single number.

---

## Reproduce

### Functional verification (~30 sec)

```sh
cd rtl
make testvectors   # generate 143 INT8-quantized replay tiles
make sim           # 110 directed cases — should print 110/110 pass
make sim_realdata  # 143 replay cases — should print 143/143 pass
```

### Yosys synthesis (~10 sec)

```sh
cd rtl
yosys -p "read_verilog -sv precision_controller.sv; \
          synth -flatten -top precision_controller; stat"
# Expect 30 FFs (29 SDFFE + 1 SDFF)
```

### Sky130 OpenLane signoff (~3 min)

Requires Docker (~25 GB free) and `pip install librelane`:

```sh
cd openlane/precision_controller
librelane --docker-no-tty --dockerized config.json
# Final metrics at runs/<latest>/final/metrics.json
```

### Cadence flow (TSMC 16FFC, when PDK is provisioned)

```sh
cd rtl
genus -files genus.tcl -log reports/genus.log
innovus -files innovus.tcl -log reports/innovus.log
```

PDK paths are at the top of each Tcl file. The end-to-end walkthrough
for fresh-chamber sign-off is in [`docs/chamber_setup.md`](docs/chamber_setup.md).

### Build the paper (~1 min)

```sh
cd paper
pdflatex adaptive_precision_attention.tex
bibtex   adaptive_precision_attention
pdflatex adaptive_precision_attention.tex
pdflatex adaptive_precision_attention.tex
```

---

## CI / CD

The pipeline at [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
runs on every push and PR:

| Job | Runner | What it does |
|---|---|---|
| `rtl-functional-verification` | GitHub Ubuntu (free) | 253 iverilog test cases |
| `rtl-synthesis` | GitHub Ubuntu (free) | Yosys synth + assert FF count = 30 |
| `reference-model-tests` | GitHub Ubuntu (free) | Builds Python + C++ refs, runs 143/143 vs RTL + MAC array tests |
| `openlane-sky130` | **self-hosted (cloud box)** | Full Sky130 signoff; **fails CI if any violation** |
| `paper-build` | GitHub Ubuntu (free) | pdflatex + bibtex + undefined-ref check |

The OpenLane job runs on a self-hosted runner because the LibreLane
Docker image is 6 GB and the PDK cache benefits from staying warm
between runs.

### Reusing this CI in other LonghornSilicon repos

When you start the KV Cache Engine, Token Importance Unit, or Memory
Hierarchy Controller repos, you have three options for sharing the CI
setup, ordered by effort:

**Option A — Copy the workflow file** (easiest, 1 minute per repo)

1. Copy `.github/workflows/ci.yml` into the new repo.
2. Adjust the paths (`rtl/`, `openlane/<block>/`, `paper/`) to the new
   repo's layout.
3. Update the FF-count assertion in `rtl-synthesis` to whatever the new
   block's closed-form predicts.

This is sufficient for now. Each repo gets its own CI but they all
share the same self-hosted runner (next item).

**Option B — Share the self-hosted runner across the org** (recommended)

Currently the runner is registered to *this repo* only. Re-register
it at the **organization level** so every LonghornSilicon repo can use
it without re-installing:

1. On https://github.com/organizations/LonghornSilicon/settings/actions/runners,
   click **New runner** → Linux → x64 to get a new token.
2. On the always-on cloud box, stop the existing runner:
   ```sh
   cd ~/actions-runner && sudo ./svc.sh stop && sudo ./svc.sh uninstall
   ./config.sh remove --token <TOKEN-FROM-OLD-REPO-RUNNER-PAGE>
   ```
3. Re-register with the **org-level** token and URL:
   ```sh
   ./config.sh \
     --url https://github.com/LonghornSilicon \
     --token <ORG-TOKEN> \
     --labels self-hosted,linux,x64 \
     --unattended
   sudo ./svc.sh install && sudo ./svc.sh start
   ```
4. The same `runs-on: [self-hosted, linux, x64]` in any LonghornSilicon
   repo's workflow will now pick it up.

**Option C — Reusable workflow** (DRY for future-you, ~30 min one-time)

Create a `LonghornSilicon/.github` repo at the org level with a
`workflow-templates/` directory. Each component repo then references the
shared template:

```yaml
# in any component repo's .github/workflows/ci.yml:
jobs:
  signoff:
    uses: LonghornSilicon/.github/.github/workflows/openlane-signoff.yml@main
    with:
      design-dir: openlane/kv_cache_engine
      expected-ff-count: 1248
```

This is overkill for two blocks but pays off by the time you have all
four. Worth doing when the second block lands.

Detailed instructions, including troubleshooting, are in
[`docs/ci_setup.md`](docs/ci_setup.md) and
[`docs/runner_setup_prompt.md`](docs/runner_setup_prompt.md).

---

## Status & roadmap

- [x] Evolutionary policy discovery (entropy → ratio simplification)
- [x] Fixed-point hardware simulation (zero misses across 8,000 tiles)
- [x] Real-LLM validation on Qwen2-0.5B/1.5B/3B and Phi-2
- [x] SystemVerilog RTL + self-checking testbenches (253/253 pass)
- [x] Yosys synth + parameter sweep (log-area scaling confirmed)
- [x] ASAP7 open synth + 16FFC projection
- [x] Sky130 OpenLane signoff (all corners clean)
- [x] Paper + CI pipeline + chamber walkthrough docs
- [ ] **TSMC 16FFC sign-off on Cadence (waiting on PDK access)**
- [ ] **ZCU102/104 FPGA prototype (Vivado, when board arrives)**
- [ ] Integration with KV Cache Engine, Token Importance Unit, Memory Hierarchy Controller
- [ ] Full-chip tape-out via TSMC University Program shuttle (Lambda 16nm track, target Summer 2027)

---

## Citation

```bibtex
@misc{adaptive_precision_attention_2026,
  title  = {Adaptive Precision Attention: Evolutionary Discovery to Hardware-Verified Ratio Gates},
  author = {LonghornSilicon},
  year   = {2026},
  url    = {https://github.com/LonghornSilicon/attention-compute-unit}
}
```

## Acknowledgments

This work builds directly on FlashAttention
(Dao et al., 2022), FlashAttention-2 (Dao, 2023), and FlashAttention-3
(Shah et al., 2024). The open hardware flow uses
[Yosys](https://github.com/YosysHQ/yosys),
[OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD),
[LibreLane](https://github.com/librelane/librelane),
[ASAP7](https://github.com/The-OpenROAD-Project/asap7), and the
[SkyWater Sky130 PDK](https://github.com/google/skywater-pdk).
