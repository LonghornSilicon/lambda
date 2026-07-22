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
