# Per-layer KV-quant sensitivity — transferable methodology (turbo4-era)

**Provenance:** extracted from the retired `c1-q412-bridge-prenorm` branch (2026-06-08
per-layer PPL ablation, `findings/centroid_design_brief.md` §9). That ablation studied the
**turbo4 / TurboQuant** codec (retired, superseded by ChannelQuant), so its *concrete*
numbers do **not** transfer — ChannelQuant has no ~50× PPL noise floor. But two
methodology lessons are codec-agnostic and worth keeping for any future per-layer study:

1. **Errors compound across layers — use leave-one-out (LOO) recovery, not single-layer
   marginal, when sizing a per-layer fix.** Single-layer ablation *undercounts* each active
   layer's true cost.
2. **Per-tile rMSE is a poor proxy for end-to-end PPL/accuracy** past the first layer. In
   the turbo4 study, a layer with 3.3× baseline rMSE was essentially neutral end-to-end,
   while an unmeasured layer was the second-worst contributor. Measure PPL/acc directly.

**Hypothesis worth testing on ChannelQuant (untested):** KV-quant loss may concentrate in a
few early layers (turbo4's was ~71% in L0). If ChannelQuant shows the same, a **hybrid
per-layer mode-select** (richer codec — e.g. CQ-8 or CQ-4+ — only on the 1–3 worst layers,
CQ-4/CQ-3-rot elsewhere) could recover most of the loss at a small bit-rate cost. The
current design uses a uniform tier per cache; this is an open lever, not a committed plan.

*(The dead turbo4/centroid/Lloyd-Max/QJL design brief this came from was discarded; only
these transferable lessons were kept. See also the WHT value-rotation A/B, which
independently rejected Lloyd-Max/QJL for the value path.)*
