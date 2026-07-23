<!-- KVE settled calls. Append-only; never delete, mark superseded. what · why · date.
     Seeded from docs/prototypes/DECISIONS.seed.md at monorepo creation 2026-07-22. -->

# DECISIONS — KVE (do not re-litigate unless the premise changed)

- **WHT value rotation is RECONFIGURABLE** · the datapath carries a per-channel sign vector
  (`sign_flips_`, applied before the WHT on the value write path), which makes the rotation
  programmable: **fixed** (all-ones signs) is the accuracy-recommended default, **randomized**
  (loaded signs) is selectable — hardware supports both, no design-time pick · 2026-07.
  *(Supersedes the 2026-07-20 "FIXED locked" call, which was accuracy-only; since the sign vector
  already exists, keeping it reconfigurable costs ~nothing and preserves the option.)*
- **CQ-4 is the default at every head dim** (the "+" outlier lane is optional) · n=1000 reversed the
  n=250 screening: the lane only marginally helps at D=128, slightly hurts at D=64 · 2026-07-21.
- **KV storage behind a swappable `kv_sram` interface** (behavioral default; real gf180 SRAM macro
  in the pdk layer) · keeps block RTL PDK-agnostic · 2026-07-22.
- **Path-B value codec has a synthesizable lowering** (`cq_value_path_wht_syn` + `wht_inverse_out_syn`
  on `wht_unit_syn`/`cq_units_syn`/`fp16_addsub_syn` + a new fp32→fp16 RNE and exact fp16→fp32 ×2⁻ᵏ) ·
  the behavioral `cq_value_path_wht`/`wht_inverse_out` use `real` and block yosys; the `_syn` twin is
  bit-exact (`make sim_wht_pathb_syn` → 5120/5120 at D=64) and unblocks the GF180 full-chip synthesis
  (used by `lambda_acu` under `ifdef LAMBDA_SYN_KVE`) · 2026-07-23.
