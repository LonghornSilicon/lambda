# ACU Import — Completeness Report

**Date:** 2026-07-22 · **Branch:** `acu-import` · **Author:** Chaithu Talasila

This report proves the ACU import into the `lambda` monorepo is **1:1 complete** — every tracked
file in the two source repos is accounted for, either **imported** (with git history) or
**intentionally excluded** for a stated, allowed reason. The only allowed exclusions are (a)
gitignored `runs/`/`build/` EDA trees and (b) duplicate RTL copies in `chipathon-lambda-acu` whose
canonical source lives in a block dir.

## Tally (the 1:1 proof)

| Source repo (scope) | Tracked files | Imported | Excluded (allowed) | Unaccounted |
|---|---|---|---|---|
| `attention-compute-unit` (`rtl` ∪ `master` ∪ `pdk-asap7`) | **224** | 224 | 0 | **0** |
| `chipathon-lambda-acu` (`rtl`; `main` ⊆ `rtl`) | **54** | 29 | 25 | **0** |
| **Total** | **278** | **253** | **25** | **0** |

- **278 source files → 253 imported + 25 intentionally-excluded, 0 unaccounted.**
- Of the 224 APA files: 223 moved to new paths (history-preserving `git mv`); 1 (`.gitignore`) had
  its patterns merged into the monorepo root `.gitignore` (content preserved) — 224 accounted.
- Of the 54 chipathon files: 27 moved to new paths, `README.md` replaced the `chip/pdk/gf180/`
  placeholder (imported), `.gitignore` merged into root (29 imported); 25 excluded (23 duplicate RTL
  copies byte-identical to canonical + 2 duplicate cosim vector `.hex` files) — 54 accounted.

## How the source scope was determined

`attention-compute-unit` (local clone dir `adaptive-precision-attention`; remote
`LonghornSilicon/attention-compute-unit`) is branch-split, same as kve/tiu:
- **`master`** — docs + APA RL research; it *deleted* the RTL (commit `0ea58fc`). `master ⊂ rtl`
  except one file (`.github/workflows/paper.yml`).
- **`rtl`** — authoritative hardware: all 5 logic tiles + Sky130 sign-offs (tip `a1af871`).
- **`pdk-asap7`** — forked from `rtl` at `e501d90`; adds only ASAP7 ORFS sign-offs +
  `docs/pdk_bracket_asap7.md` (24 files). (It is NOT a superset of `rtl`; they diverged.)

Full block = union **`rtl ∪ master ∪ pdk-asap7`**, content-favoring `rtl` (authoritative; also
carries the RTL `master` deleted — the trap kve/tiu hit). Built with real merge commits
(`git merge -s ours` + selective checkout, history preserved), `git subtree add`-ed into the
monorepo, then `git mv`-restructured. Union = 224 (`rtl` 199 + `master`-only `paper.yml` 1 +
`pdk-asap7`-only 24).

`chipathon-lambda-acu` (local, no remote): `main ⊆ rtl`, so only `rtl` imported. Its GF180 hardened
outputs are gitignored (`runs/`, `build/`) — **no** tracked results exist to import; only the
LibreLane configs, padring/integration RTL, GLS tb, real-SRAM macro, and docs.

> ⚠️ **Out-of-scope branch flagged (nothing silently dropped):** `attention-compute-unit` also has an
> unmerged remote branch **`c11-end-to-end-perplexity`** (ChannelQuant + KVCE end-to-end perplexity
> campaign, notebooks, per-layer ablations). It is NOT part of `rtl ∪ master`, overlaps the
> KVE/ChannelQuant blocks, and was NOT imported here. It remains on the source remote; import it with
> the KVE/ChannelQuant work if desired.

## History preservation — verified
`git log --follow` on moved files shows true upstream history:
- `acu/mate/rtl/mate_qkt.sv` → `93e9960 mate_qkt: synthesizable Q.Kt decode-scoring RTL`.
- `research/apa-precision-policy/phase1_policy/evaluator.py` → back through the RL-project commits.
- `acu/mate/pdk/asap7/orfs/asap7/mate_pv/config.mk` → `25a5124 pdk(asap7): predictive 7nm bracket`.
- `acu/.github/workflows/paper.yml` → `65a8769` (master-side paper gate).

## A. `attention-compute-unit` — full file map (224 → 223 moved + 1 merged)

`.gitignore` → **merged into monorepo root `.gitignore`** (openlane/orfs run trees, latex/py/C++
build artifacts, generated testvectors — re-scoped to the block-major `**/` globs). All other 223:

| source (attention-compute-unit, `rtl∪master∪pdk-asap7`) | monorepo destination |
|---|---|
| `.github/workflows/ci.yml` | `acu/.github/workflows/ci.yml` |
| `.github/workflows/paper.yml` | `acu/.github/workflows/paper.yml` |
| `README.md` | `acu/docs/acu_overview.md` |
| `analysis/benchmark_entropy_approx.py` | `research/apa-precision-policy/analysis/benchmark_entropy_approx.py` |
| `analysis/benchmark_results.json` | `research/apa-precision-policy/analysis/benchmark_results.json` |
| `analysis/benchmark_vs_fa2.py` | `research/apa-precision-policy/analysis/benchmark_vs_fa2.py` |
| `analysis/block_size_sweep.py` | `research/apa-precision-policy/analysis/block_size_sweep.py` |
| `analysis/block_size_sweep_results.json` | `research/apa-precision-policy/analysis/block_size_sweep_results.json` |
| `analysis/cq_apa_e2e.py` | `research/apa-precision-policy/analysis/cq_apa_e2e.py` |
| `analysis/cq_apa_qwen05b_n1000.json` | `research/apa-precision-policy/analysis/cq_apa_qwen05b_n1000.json` |
| `analysis/cq_apa_qwen15b_n1000.json` | `research/apa-precision-policy/analysis/cq_apa_qwen15b_n1000.json` |
| `analysis/deep_layer_combined_stats.json` | `research/apa-precision-policy/analysis/deep_layer_combined_stats.json` |
| `analysis/deep_layer_stats_Qwen2-1.5B.json` | `research/apa-precision-policy/analysis/deep_layer_stats_Qwen2-1.5B.json` |
| `analysis/deep_layer_stats_Qwen2.5-3B.json` | `research/apa-precision-policy/analysis/deep_layer_stats_Qwen2.5-3B.json` |
| `analysis/figures/accuracy_vs_baselines.png` | `research/apa-precision-policy/analysis/figures/accuracy_vs_baselines.png` |
| `analysis/figures/block_size_sweep.png` | `research/apa-precision-policy/analysis/figures/block_size_sweep.png` |
| `analysis/figures/compression_summary.png` | `research/apa-precision-policy/analysis/figures/compression_summary.png` |
| `analysis/figures/deep_layer_comparison.png` | `research/apa-precision-policy/analysis/figures/deep_layer_comparison.png` |
| `analysis/figures/entropy_distribution.png` | `research/apa-precision-policy/analysis/figures/entropy_distribution.png` |
| `analysis/figures/fixed_point_sim.png` | `research/apa-precision-policy/analysis/figures/fixed_point_sim.png` |
| `analysis/figures/phi2_validation.png` | `research/apa-precision-policy/analysis/figures/phi2_validation.png` |
| `analysis/figures/precision_heatmaps.png` | `research/apa-precision-policy/analysis/figures/precision_heatmaps.png` |
| `analysis/figures/real_llm_v2.png` | `research/apa-precision-policy/analysis/figures/real_llm_v2.png` |
| `analysis/figures/real_llm_validation.png` | `research/apa-precision-policy/analysis/figures/real_llm_validation.png` |
| `analysis/figures/rtl_area_vs_bitwidth.png` | `research/apa-precision-policy/analysis/figures/rtl_area_vs_bitwidth.png` |
| `analysis/figures/rtl_area_vs_tile.png` | `research/apa-precision-policy/analysis/figures/rtl_area_vs_tile.png` |
| `analysis/figures/seq_length_scaling.png` | `research/apa-precision-policy/analysis/figures/seq_length_scaling.png` |
| `analysis/fixed_point_sim.py` | `research/apa-precision-policy/analysis/fixed_point_sim.py` |
| `analysis/fixed_point_sim_results.json` | `research/apa-precision-policy/analysis/fixed_point_sim_results.json` |
| `analysis/gen_rtl_testvectors.py` | `research/apa-precision-policy/analysis/gen_rtl_testvectors.py` |
| `analysis/integration_test_kv_pc.py` | `research/apa-precision-policy/analysis/integration_test_kv_pc.py` |
| `analysis/integration_test_kv_pc_stats.json` | `research/apa-precision-policy/analysis/integration_test_kv_pc_stats.json` |
| `analysis/phi2_validation_stats.json` | `research/apa-precision-policy/analysis/phi2_validation_stats.json` |
| `analysis/pv_accumulator_width.json` | `research/apa-precision-policy/analysis/pv_accumulator_width.json` |
| `analysis/pv_accumulator_width.py` | `research/apa-precision-policy/analysis/pv_accumulator_width.py` |
| `analysis/qwen_ci_gate.py` | `research/apa-precision-policy/analysis/qwen_ci_gate.py` |
| `analysis/real_llm_stats.json` | `research/apa-precision-policy/analysis/real_llm_stats.json` |
| `analysis/real_llm_v2_stats.json` | `research/apa-precision-policy/analysis/real_llm_v2_stats.json` |
| `analysis/rtl_sweep_notes.md` | `research/apa-precision-policy/analysis/rtl_sweep_notes.md` |
| `analysis/rtl_sweep_results.json` | `research/apa-precision-policy/analysis/rtl_sweep_results.json` |
| `analysis/run_analysis.py` | `research/apa-precision-policy/analysis/run_analysis.py` |
| `analysis/sparsity_pilot.py` | `research/apa-precision-policy/analysis/sparsity_pilot.py` |
| `analysis/sparsity_pilot_results.json` | `research/apa-precision-policy/analysis/sparsity_pilot_results.json` |
| `analysis/sparsity_real_llm_4k_stats.json` | `research/apa-precision-policy/analysis/sparsity_real_llm_4k_stats.json` |
| `analysis/sparsity_real_llm_capture.py` | `research/apa-precision-policy/analysis/sparsity_real_llm_capture.py` |
| `analysis/sparsity_real_llm_stats.json` | `research/apa-precision-policy/analysis/sparsity_real_llm_stats.json` |
| `analysis/tsmc16_fit_report.md` | `research/apa-precision-policy/analysis/tsmc16_fit_report.md` |
| `analysis/validate_deep_layers.py` | `research/apa-precision-policy/analysis/validate_deep_layers.py` |
| `analysis/validate_llama.py` | `research/apa-precision-policy/analysis/validate_llama.py` |
| `analysis/validate_real_llm.py` | `research/apa-precision-policy/analysis/validate_real_llm.py` |
| `analysis/validate_real_llm_v2.py` | `research/apa-precision-policy/analysis/validate_real_llm_v2.py` |
| `common/__init__.py` | `research/apa-precision-policy/common/__init__.py` |
| `common/block_stats.py` | `research/apa-precision-policy/common/block_stats.py` |
| `common/quantization.py` | `research/apa-precision-policy/common/quantization.py` |
| `common/reference.py` | `research/apa-precision-policy/common/reference.py` |
| `common/workloads.py` | `research/apa-precision-policy/common/workloads.py` |
| `docs/chamber_setup.md` | `acu/docs/chamber_setup.md` |
| `docs/ci_overview.md` | `acu/docs/ci_overview.md` |
| `docs/ci_setup.md` | `acu/docs/ci_setup.md` |
| `docs/findings/kvce-acu-integration-audit.md` | `acu/docs/findings/kvce-acu-integration-audit.md` |
| `docs/findings/kvce_acu_architectural_conflicts.pdf` | `acu/docs/findings/kvce_acu_architectural_conflicts.pdf` |
| `docs/findings/kvce_acu_architectural_conflicts.tex` | `acu/docs/findings/kvce_acu_architectural_conflicts.tex` |
| `docs/findings/kvce_acu_integration_audit.pdf` | `acu/docs/findings/kvce_acu_integration_audit.pdf` |
| `docs/findings/kvce_acu_integration_audit.tex` | `acu/docs/findings/kvce_acu_integration_audit.tex` |
| `docs/findings/phase1-entropy-finding.md` | `research/apa-precision-policy/findings/phase1-entropy-finding.md` |
| `docs/findings/sparsity-controller-finding.md` | `research/apa-precision-policy/findings/sparsity-controller-finding.md` |
| `docs/isa/precision_controller_isa.md` | `acu/precision_controller/docs/isa/precision_controller_isa.md` |
| `docs/isa/precision_controller_isa.pdf` | `acu/precision_controller/docs/isa/precision_controller_isa.pdf` |
| `docs/isa/precision_controller_isa.tex` | `acu/precision_controller/docs/isa/precision_controller_isa.tex` |
| `docs/library_proposal.pdf` | `acu/docs/library_proposal.pdf` |
| `docs/library_proposal.tex` | `acu/docs/library_proposal.tex` |
| `docs/mac_array_design.pdf` | `acu/mate/docs/mac_array_design.pdf` |
| `docs/mac_array_design.tex` | `acu/mate/docs/mac_array_design.tex` |
| `docs/mate_pv_fp16_rtl.md` | `acu/mate/docs/mate_pv_fp16_rtl.md` |
| `docs/mate_pv_rtl.md` | `acu/mate/docs/mate_pv_rtl.md` |
| `docs/mate_qkt_rtl.md` | `acu/mate/docs/mate_qkt_rtl.md` |
| `docs/meeting_handout.md` | `acu/docs/meeting_handout.md` |
| `docs/meeting_handout.pdf` | `acu/docs/meeting_handout.pdf` |
| `docs/meeting_handout.tex` | `acu/docs/meeting_handout.tex` |
| `docs/new_block_blueprint.md` | `acu/docs/new_block_blueprint.md` |
| `docs/pdk_bracket_asap7.md` | `acu/docs/pdk_bracket_asap7.md` |
| `docs/reference_model_api.pdf` | `acu/docs/reference_model_api.pdf` |
| `docs/reference_model_api.tex` | `acu/docs/reference_model_api.tex` |
| `docs/superpowers/plans/2026-05-03-flash-attention-5-phase1.md` | `research/apa-precision-policy/superpowers/plans/2026-05-03-flash-attention-5-phase1.md` |
| `docs/superpowers/plans/2026-05-04-phase1-analysis-dashboard.md` | `research/apa-precision-policy/superpowers/plans/2026-05-04-phase1-analysis-dashboard.md` |
| `docs/superpowers/specs/2026-05-03-flash-attention-5-design.md` | `research/apa-precision-policy/superpowers/specs/2026-05-03-flash-attention-5-design.md` |
| `docs/superpowers/specs/2026-05-04-phase1-analysis-dashboard.md` | `research/apa-precision-policy/superpowers/specs/2026-05-04-phase1-analysis-dashboard.md` |
| `docs/sw_overview.pdf` | `acu/docs/sw_overview.pdf` |
| `docs/sw_overview.tex` | `acu/docs/sw_overview.tex` |
| `docs/vecu_softmax_rtl.md` | `acu/vecu/docs/vecu_softmax_rtl.md` |
| `kv_cache/__init__.py` | `research/apa-precision-policy/kv_cache/__init__.py` |
| `kv_cache/precision_controller.py` | `research/apa-precision-policy/kv_cache/precision_controller.py` |
| `kv_cache/quantized_cache.py` | `research/apa-precision-policy/kv_cache/quantized_cache.py` |
| `openlane/mate_pv/README.md` | `acu/mate/pdk/sky130/openlane/mate_pv/README.md` |
| `openlane/mate_pv/config.json` | `acu/mate/pdk/sky130/openlane/mate_pv/config.json` |
| `openlane/mate_pv/results/mate_pv.gds` | `acu/mate/pdk/sky130/openlane/mate_pv/results/mate_pv.gds` |
| `openlane/mate_pv/results/mate_pv.png` | `acu/mate/pdk/sky130/openlane/mate_pv/results/mate_pv.png` |
| `openlane/mate_pv/results/sky130_71MHz_signoff_metrics.json` | `acu/mate/pdk/sky130/openlane/mate_pv/results/sky130_71MHz_signoff_metrics.json` |
| `openlane/mate_pv/src/mate_pv.sv` | `acu/mate/pdk/sky130/openlane/mate_pv/src/mate_pv.sv` |
| `openlane/mate_pv_fp16/README.md` | `acu/mate/pdk/sky130/openlane/mate_pv_fp16/README.md` |
| `openlane/mate_pv_fp16/config.json` | `acu/mate/pdk/sky130/openlane/mate_pv_fp16/config.json` |
| `openlane/mate_pv_fp16/results/mate_pv_fp16.gds` | `acu/mate/pdk/sky130/openlane/mate_pv_fp16/results/mate_pv_fp16.gds` |
| `openlane/mate_pv_fp16/results/mate_pv_fp16.png` | `acu/mate/pdk/sky130/openlane/mate_pv_fp16/results/mate_pv_fp16.png` |
| `openlane/mate_pv_fp16/results/sky130_12MHz_signoff_metrics.json` | `acu/mate/pdk/sky130/openlane/mate_pv_fp16/results/sky130_12MHz_signoff_metrics.json` |
| `openlane/mate_pv_fp16/src/mate_pv_fp16.sv` | `acu/mate/pdk/sky130/openlane/mate_pv_fp16/src/mate_pv_fp16.sv` |
| `openlane/mate_qkt/README.md` | `acu/mate/pdk/sky130/openlane/mate_qkt/README.md` |
| `openlane/mate_qkt/config.json` | `acu/mate/pdk/sky130/openlane/mate_qkt/config.json` |
| `openlane/mate_qkt/results/mate_qkt.gds` | `acu/mate/pdk/sky130/openlane/mate_qkt/results/mate_qkt.gds` |
| `openlane/mate_qkt/results/mate_qkt.png` | `acu/mate/pdk/sky130/openlane/mate_qkt/results/mate_qkt.png` |
| `openlane/mate_qkt/results/sky130_signoff_metrics.json` | `acu/mate/pdk/sky130/openlane/mate_qkt/results/sky130_signoff_metrics.json` |
| `openlane/mate_qkt/src/mate_qkt.sv` | `acu/mate/pdk/sky130/openlane/mate_qkt/src/mate_qkt.sv` |
| `openlane/precision_controller/README.md` | `acu/precision_controller/pdk/sky130/openlane/precision_controller/README.md` |
| `openlane/precision_controller/config.json` | `acu/precision_controller/pdk/sky130/openlane/precision_controller/config.json` |
| `openlane/precision_controller/results/precision_controller.gds` | `acu/precision_controller/pdk/sky130/openlane/precision_controller/results/precision_controller.gds` |
| `openlane/precision_controller/results/precision_controller.png` | `acu/precision_controller/pdk/sky130/openlane/precision_controller/results/precision_controller.png` |
| `openlane/precision_controller/results/sky130_80MHz_signoff_metrics.json` | `acu/precision_controller/pdk/sky130/openlane/precision_controller/results/sky130_80MHz_signoff_metrics.json` |
| `openlane/precision_controller/src/precision_controller.v` | `acu/precision_controller/pdk/sky130/openlane/precision_controller/src/precision_controller.v` |
| `openlane/vecu_softmax/README.md` | `acu/vecu/pdk/sky130/openlane/vecu_softmax/README.md` |
| `openlane/vecu_softmax/config.json` | `acu/vecu/pdk/sky130/openlane/vecu_softmax/config.json` |
| `openlane/vecu_softmax/results/sky130_signoff_metrics.json` | `acu/vecu/pdk/sky130/openlane/vecu_softmax/results/sky130_signoff_metrics.json` |
| `openlane/vecu_softmax/results/vecu_softmax.gds.gz` | `acu/vecu/pdk/sky130/openlane/vecu_softmax/results/vecu_softmax.gds.gz` |
| `openlane/vecu_softmax/results/vecu_softmax.png` | `acu/vecu/pdk/sky130/openlane/vecu_softmax/results/vecu_softmax.png` |
| `openlane/vecu_softmax/src/vecu_softmax.sv` | `acu/vecu/pdk/sky130/openlane/vecu_softmax/src/vecu_softmax.sv` |
| `orfs/asap7/README.md` | `acu/mate/pdk/asap7/orfs/asap7/README.md` |
| `orfs/asap7/mate_pv/config.mk` | `acu/mate/pdk/asap7/orfs/asap7/mate_pv/config.mk` |
| `orfs/asap7/mate_pv/constraint.sdc` | `acu/mate/pdk/asap7/orfs/asap7/mate_pv/constraint.sdc` |
| `orfs/asap7/mate_pv/results_asap7/6_finish.rpt` | `acu/mate/pdk/asap7/orfs/asap7/mate_pv/results_asap7/6_finish.rpt` |
| `orfs/asap7/mate_pv/results_asap7/asap7_2GHz_metrics.json` | `acu/mate/pdk/asap7/orfs/asap7/mate_pv/results_asap7/asap7_2GHz_metrics.json` |
| `orfs/asap7/mate_pv/results_asap7/mate_pv_asap7.gds` | `acu/mate/pdk/asap7/orfs/asap7/mate_pv/results_asap7/mate_pv_asap7.gds` |
| `orfs/asap7/mate_pv/results_asap7/mate_pv_asap7_layout.webp` | `acu/mate/pdk/asap7/orfs/asap7/mate_pv/results_asap7/mate_pv_asap7_layout.webp` |
| `orfs/asap7/mate_pv/results_asap7/synth_stat.txt` | `acu/mate/pdk/asap7/orfs/asap7/mate_pv/results_asap7/synth_stat.txt` |
| `orfs/asap7/mate_pv_fp16/config.mk` | `acu/mate/pdk/asap7/orfs/asap7/mate_pv_fp16/config.mk` |
| `orfs/asap7/mate_pv_fp16/constraint.sdc` | `acu/mate/pdk/asap7/orfs/asap7/mate_pv_fp16/constraint.sdc` |
| `orfs/asap7/mate_pv_fp16/results_asap7/6_finish.rpt` | `acu/mate/pdk/asap7/orfs/asap7/mate_pv_fp16/results_asap7/6_finish.rpt` |
| `orfs/asap7/mate_pv_fp16/results_asap7/asap7_286MHz_metrics.json` | `acu/mate/pdk/asap7/orfs/asap7/mate_pv_fp16/results_asap7/asap7_286MHz_metrics.json` |
| `orfs/asap7/mate_pv_fp16/results_asap7/mate_pv_fp16_asap7.gds` | `acu/mate/pdk/asap7/orfs/asap7/mate_pv_fp16/results_asap7/mate_pv_fp16_asap7.gds` |
| `orfs/asap7/mate_pv_fp16/results_asap7/mate_pv_fp16_asap7_layout.webp` | `acu/mate/pdk/asap7/orfs/asap7/mate_pv_fp16/results_asap7/mate_pv_fp16_asap7_layout.webp` |
| `orfs/asap7/mate_pv_fp16/results_asap7/synth_stat.txt` | `acu/mate/pdk/asap7/orfs/asap7/mate_pv_fp16/results_asap7/synth_stat.txt` |
| `orfs/asap7/precision_controller/config.mk` | `acu/precision_controller/pdk/asap7/orfs/asap7/precision_controller/config.mk` |
| `orfs/asap7/precision_controller/constraint.sdc` | `acu/precision_controller/pdk/asap7/orfs/asap7/precision_controller/constraint.sdc` |
| `orfs/asap7/precision_controller/results_asap7/6_finish.rpt` | `acu/precision_controller/pdk/asap7/orfs/asap7/precision_controller/results_asap7/6_finish.rpt` |
| `orfs/asap7/precision_controller/results_asap7/asap7_1176MHz_metrics.json` | `acu/precision_controller/pdk/asap7/orfs/asap7/precision_controller/results_asap7/asap7_1176MHz_metrics.json` |
| `orfs/asap7/precision_controller/results_asap7/precision_controller_asap7.gds` | `acu/precision_controller/pdk/asap7/orfs/asap7/precision_controller/results_asap7/precision_controller_asap7.gds` |
| `orfs/asap7/precision_controller/results_asap7/precision_controller_asap7_layout.webp` | `acu/precision_controller/pdk/asap7/orfs/asap7/precision_controller/results_asap7/precision_controller_asap7_layout.webp` |
| `orfs/asap7/precision_controller/results_asap7/synth_stat.txt` | `acu/precision_controller/pdk/asap7/orfs/asap7/precision_controller/results_asap7/synth_stat.txt` |
| `orfs/asap7/run_asap7.sh` | `acu/mate/pdk/asap7/orfs/asap7/run_asap7.sh` |
| `paper/accuracy_vs_baselines.png` | `research/apa-precision-policy/paper/accuracy_vs_baselines.png` |
| `paper/adaptive_precision_attention.md` | `research/apa-precision-policy/paper/adaptive_precision_attention.md` |
| `paper/adaptive_precision_attention.pdf` | `research/apa-precision-policy/paper/adaptive_precision_attention.pdf` |
| `paper/adaptive_precision_attention.tex` | `research/apa-precision-policy/paper/adaptive_precision_attention.tex` |
| `paper/compression_summary.png` | `research/apa-precision-policy/paper/compression_summary.png` |
| `paper/entropy_distribution.png` | `research/apa-precision-policy/paper/entropy_distribution.png` |
| `paper/precision_heatmaps.png` | `research/apa-precision-policy/paper/precision_heatmaps.png` |
| `paper/references.bib` | `research/apa-precision-policy/paper/references.bib` |
| `paper/seq_length_scaling.png` | `research/apa-precision-policy/paper/seq_length_scaling.png` |
| `paper/sky130_layout.png` | `research/apa-precision-policy/paper/sky130_layout.png` |
| `phase1_policy/__init__.py` | `research/apa-precision-policy/phase1_policy/__init__.py` |
| `phase1_policy/config.yaml` | `research/apa-precision-policy/phase1_policy/config.yaml` |
| `phase1_policy/evaluator.py` | `research/apa-precision-policy/phase1_policy/evaluator.py` |
| `phase1_policy/initial_policy.py` | `research/apa-precision-policy/phase1_policy/initial_policy.py` |
| `phase2_kernel/adaptive_attention.py` | `research/apa-precision-policy/phase2_kernel/adaptive_attention.py` |
| `phase2_kernel/best_policy.py` | `research/apa-precision-policy/phase2_kernel/best_policy.py` |
| `phase2_kernel/config.yaml` | `research/apa-precision-policy/phase2_kernel/config.yaml` |
| `phase2_kernel/evaluator.py` | `research/apa-precision-policy/phase2_kernel/evaluator.py` |
| `phase2_kernel/initial_kernel.py` | `research/apa-precision-policy/phase2_kernel/initial_kernel.py` |
| `phase2_kernel/test_adaptive_kernel.py` | `research/apa-precision-policy/phase2_kernel/test_adaptive_kernel.py` |
| `requirements.txt` | `research/apa-precision-policy/requirements.txt` |
| `rtl/Makefile` | `acu/mate/rtl/Makefile` |
| `rtl/constraints/timing.sdc` | `acu/mate/rtl/constraints/timing.sdc` |
| `rtl/constraints/timing_gsclib045.sdc` | `acu/mate/rtl/constraints/timing_gsclib045.sdc` |
| `rtl/constraints/zcu102.xdc` | `acu/mate/rtl/constraints/zcu102.xdc` |
| `rtl/genus.tcl` | `acu/mate/rtl/eda/genus.tcl` |
| `rtl/genus_gsclib045.tcl` | `acu/mate/rtl/eda/genus_gsclib045.tcl` |
| `rtl/innovus.tcl` | `acu/mate/rtl/eda/innovus.tcl` |
| `rtl/mate_pv.sv` | `acu/mate/rtl/mate_pv.sv` |
| `rtl/mate_pv_fp16.sv` | `acu/mate/rtl/mate_pv_fp16.sv` |
| `rtl/mate_qkt.sv` | `acu/mate/rtl/mate_qkt.sv` |
| `rtl/mmmc.tcl` | `acu/mate/rtl/eda/mmmc.tcl` |
| `rtl/precision_controller.sv` | `acu/precision_controller/rtl/precision_controller.sv` |
| `rtl/sweep_synth.py` | `acu/mate/rtl/sweep_synth.py` |
| `rtl/synth.ys` | `acu/mate/rtl/synth.ys` |
| `rtl/tb/gen_mate_pv_fp16_vectors.py` | `acu/mate/rtl/tb/gen_mate_pv_fp16_vectors.py` |
| `rtl/tb/gen_mate_pv_vectors.py` | `acu/mate/rtl/tb/gen_mate_pv_vectors.py` |
| `rtl/tb/gen_mate_qkt_vectors.py` | `acu/mate/rtl/tb/gen_mate_qkt_vectors.py` |
| `rtl/tb/gen_vecu_softmax_vectors.py` | `acu/vecu/rtl/tb/gen_vecu_softmax_vectors.py` |
| `rtl/tb/tb_mate_pv.sv` | `acu/mate/rtl/tb/tb_mate_pv.sv` |
| `rtl/tb/tb_mate_pv_fp16.sv` | `acu/mate/rtl/tb/tb_mate_pv_fp16.sv` |
| `rtl/tb/tb_mate_qkt.sv` | `acu/mate/rtl/tb/tb_mate_qkt.sv` |
| `rtl/tb/tb_precision_controller.sv` | `acu/precision_controller/rtl/tb/tb_precision_controller.sv` |
| `rtl/tb/tb_realdata.sv` | `acu/mate/rtl/tb/tb_realdata.sv` |
| `rtl/tb/tb_vecu_softmax.sv` | `acu/vecu/rtl/tb/tb_vecu_softmax.sv` |
| `rtl/tb/testvectors/labels.txt` | `acu/mate/rtl/tb/testvectors/labels.txt` |
| `rtl/vecu_softmax.sv` | `acu/vecu/rtl/vecu_softmax.sv` |
| `run_phase1.sh` | `research/apa-precision-policy/run_phase1.sh` |
| `run_phase2.sh` | `research/apa-precision-policy/run_phase2.sh` |
| `sw/README.md` | `acu/mate/sw/README.md` |
| `sw/reference_model/MAC_ARRAY_DESIGN.md` | `acu/mate/sw/reference_model/MAC_ARRAY_DESIGN.md` |
| `sw/reference_model/Makefile` | `acu/mate/sw/reference_model/Makefile` |
| `sw/reference_model/README.md` | `acu/mate/sw/reference_model/README.md` |
| `sw/reference_model/example_compiler_use.py` | `acu/mate/sw/reference_model/example_compiler_use.py` |
| `sw/reference_model/integration_example.py` | `acu/mate/sw/reference_model/integration_example.py` |
| `sw/reference_model/mac_array_ref.cpp` | `acu/mate/sw/reference_model/mac_array_ref.cpp` |
| `sw/reference_model/mac_array_ref.hpp` | `acu/mate/sw/reference_model/mac_array_ref.hpp` |
| `sw/reference_model/mac_array_ref.py` | `acu/mate/sw/reference_model/mac_array_ref.py` |
| `sw/reference_model/precision_controller_ref.cpp` | `acu/precision_controller/sw/reference_model/precision_controller_ref.cpp` |
| `sw/reference_model/precision_controller_ref.hpp` | `acu/precision_controller/sw/reference_model/precision_controller_ref.hpp` |
| `sw/reference_model/precision_controller_ref.py` | `acu/precision_controller/sw/reference_model/precision_controller_ref.py` |
| `sw/reference_model/sparsity_controller_ref.py` | `acu/mate/sw/reference_model/sparsity_controller_ref.py` |
| `sw/reference_model/test_mac_array_ref.cpp` | `acu/mate/sw/reference_model/test_mac_array_ref.cpp` |
| `sw/reference_model/test_mac_array_ref.py` | `acu/mate/sw/reference_model/test_mac_array_ref.py` |
| `sw/reference_model/test_precision_controller_ref.cpp` | `acu/precision_controller/sw/reference_model/test_precision_controller_ref.cpp` |
| `sw/reference_model/test_precision_controller_ref.py` | `acu/precision_controller/sw/reference_model/test_precision_controller_ref.py` |
| `sw/reference_model/test_sparsity_controller_ref.py` | `acu/mate/sw/reference_model/test_sparsity_controller_ref.py` |
| `sw/reference_model/vecu_softmax_ref.py` | `acu/vecu/sw/reference_model/vecu_softmax_ref.py` |
| `tests/__init__.py` | `research/apa-precision-policy/tests/__init__.py` |
| `tests/test_block_stats.py` | `research/apa-precision-policy/tests/test_block_stats.py` |
| `tests/test_evaluator.py` | `research/apa-precision-policy/tests/test_evaluator.py` |
| `tests/test_kernel_evaluator.py` | `research/apa-precision-policy/tests/test_kernel_evaluator.py` |
| `tests/test_kv_cache.py` | `research/apa-precision-policy/tests/test_kv_cache.py` |
| `tests/test_mixed_precision_attention.py` | `research/apa-precision-policy/tests/test_mixed_precision_attention.py` |
| `tests/test_quantization.py` | `research/apa-precision-policy/tests/test_quantization.py` |
| `tests/test_reference.py` | `research/apa-precision-policy/tests/test_reference.py` |
| `tests/test_workloads.py` | `research/apa-precision-policy/tests/test_workloads.py` |

## B. `chipathon-lambda-acu` — full file map (54 → 29 imported + 25 excluded)

### B.1 Imported (29)
`README.md` → `chip/pdk/gf180/README.md` (replaced placeholder); `.gitignore` → merged into root.
The other 27:

| source (chipathon-lambda-acu, `rtl`) | monorepo destination |
|---|---|
| `docs/architecture.md` | `chip/pdk/gf180/docs/architecture.md` |
| `docs/build.md` | `chip/pdk/gf180/docs/build.md` |
| `docs/gf180_gls_report.md` | `chip/pdk/gf180/docs/gf180_gls_report.md` |
| `librelane/kve.yaml` | `kve/pdk/gf180/librelane/kve.yaml` |
| `librelane/kve_store_gf180.yaml` | `kve/pdk/gf180/librelane/kve_store_gf180.yaml` |
| `librelane/mate_pv.yaml` | `acu/mate/pdk/gf180/librelane/mate_pv.yaml` |
| `librelane/mate_pv_fp16.yaml` | `acu/mate/pdk/gf180/librelane/mate_pv_fp16.yaml` |
| `librelane/mate_qkt.yaml` | `acu/mate/pdk/gf180/librelane/mate_qkt.yaml` |
| `librelane/pdn_cfg_sram.tcl` | `kve/pdk/gf180/librelane/pdn_cfg_sram.tcl` |
| `librelane/precision_controller.yaml` | `acu/precision_controller/pdk/gf180/librelane/precision_controller.yaml` |
| `librelane/token_importance_unit.yaml` | `tiu/pdk/gf180/librelane/token_importance_unit.yaml` |
| `librelane/vecu_softmax.yaml` | `acu/vecu/pdk/gf180/librelane/vecu_softmax.yaml` |
| `rtl/blocks/PROVENANCE.md` | `chip/pdk/gf180/PROVENANCE.md` |
| `rtl/blocks/kve_gf180_sram/gf180mcu_fd_ip_sram__sram512x8m8wm1__bb.v` | `kve/pdk/gf180/kve_gf180_sram/gf180mcu_fd_ip_sram__sram512x8m8wm1__bb.v` |
| `rtl/blocks/kve_gf180_sram/kv_sram.sv` | `kve/pdk/gf180/kve_gf180_sram/kv_sram.sv` |
| `rtl/blocks/kve_gf180_sram/maglef_drc/gf180mcu_fd_ip_sram__sram512x8m8wm1.mag` | `kve/pdk/gf180/kve_gf180_sram/maglef_drc/gf180mcu_fd_ip_sram__sram512x8m8wm1.mag` |
| `rtl/chip_core.sv` | `chip/rtl/chip_core.sv` |
| `rtl/lambda_acu.sv` | `chip/rtl/lambda_acu.sv` |
| `rtl/spi_loader.sv` | `chip/rtl/spi_loader.sv` |
| `scripts/harden.sh` | `chip/pdk/gf180/scripts/harden.sh` |
| `tb/Makefile.cocotb` | `chip/pdk/gf180/tb/Makefile.cocotb` |
| `tb/Makefile` | `chip/pdk/gf180/tb/Makefile` |
| `tb/chip_core_wrap.sv` | `chip/pdk/gf180/tb/chip_core_wrap.sv` |
| `tb/tb_gls_e2e.sv` | `chip/pdk/gf180/tb/tb_gls_e2e.sv` |
| `tb/tb_kv_sram_gf180.sv` | `chip/pdk/gf180/tb/tb_kv_sram_gf180.sv` |
| `tb/test_smoke.py` | `chip/pdk/gf180/tb/test_smoke.py` |
| `tb/timescale.v` | `chip/pdk/gf180/tb/timescale.v` |

### B.2 Excluded — 25, all allowed (duplicate RTL copies + duplicate vectors)
All 23 `rtl/blocks/**` `.sv` copies were verified **byte-identical** to their canonical block source
at import time (no drift); the 2 `.hex` files are byte-identical to `chip/verif/vectors/`.

| excluded source | reason | canonical / duplicate-of |
|---|---|---|
| `rtl/blocks/kve/{amax_unit,cq_fp_pkg,cq_key_path,cq_units,cq_units_syn,cq_value_path,cq_value_path_wht,cq_wht_value,fp16_addsub_syn,kv_cache_engine,kv_sram,residual_buffer,scale_bank,sram_controller,wht_inverse_out,wht_unit,wht_unit_syn}.sv` (17) | duplicate RTL copy | canonical `kve/rtl/*.sv` |
| `rtl/blocks/{mate_pv,mate_pv_fp16,mate_qkt,vecu_softmax}.sv` (4) | duplicate RTL copy | canonical `acu/{mate,vecu}/rtl/*.sv` |
| `rtl/blocks/precision_controller.sv` (1) | duplicate RTL copy | canonical `acu/precision_controller/rtl/precision_controller.sv` |
| `rtl/blocks/token_importance_unit.sv` (1) | duplicate RTL copy | canonical `tiu/rtl/token_importance_unit.sv` |
| `tb/vectors/qwen_val.hex`, `tb/vectors/qwen_vhatwht.hex` (2) | duplicate vector | identical to `chip/verif/vectors/*.hex` |

**Note:** the copy ledger itself, `rtl/blocks/PROVENANCE.md`, WAS imported (→
`chip/pdk/gf180/PROVENANCE.md`) as the historical record of the (now-eliminated) copy mechanism.
Excluded GF180 hardened outputs beyond these are gitignored `runs/`/`build/` (never tracked).

## C. Additional structural changes (not source files)
- **Cosim re-pointed:** `chip/verif/Makefile` ACU rows now reference `acu/*/rtl/*.sv` directly; the 5
  vendored `chip/verif/blocks/acu/*.sv` copies (byte-identical) were removed (drift elimination).
  `make -C chip/verif cosim` passes (ALL BLOCKS PASS). KVE/TIU vendored copies left untouched.
- **Mirror workflow:** the `acu`, `acu/mate`, `acu/vecu`, `acu/precision_controller` rows in
  `.github/workflows/mirror-blocks.yml` were uncommented.
- **New convention files** (not from source): per-sub-block `README.md`/`AGENTS.md`/`DECISIONS.md`,
  `acu/{README,AGENTS,DECISIONS}.md`, per-block `research/README.md`,
  `research/apa-precision-policy/README.md`. The legacy repo README landed at
  `acu/docs/acu_overview.md` (kept verbatim; a pre-reorg-paths banner added).

**Conclusion: 278 source files → 253 imported + 25 intentionally-excluded, 0 unaccounted. 1:1 complete.**
