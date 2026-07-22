# ACU — Attention Compute Unit (PLACEHOLDER — import held)

> **TODO / HELD (2026-07-22).** This block is **not imported yet**. Two source repos are
> mid-flight and must NOT be imported until their agents finish:
>
> - **`attention-compute-unit`** — an agent is adding Sky130 sign-offs
>   (`openlane/mate_qkt/` + `openlane/vecu_softmax/`).
> - **`chipathon-lambda-acu`** — an agent is re-hardening `vecu_softmax` (GF180).
>
> Importing mid-flight would multiply merge conflicts. Import this block **after both agents land**,
> following the same block-major, history-preserving flow used for `kve/` and `tiu/`
> (`git subtree add`), and split into the self-contained sub-blocks below.

## Planned structure (block-major, per `docs/repo_reorg_plan.md`)
```
acu/
├── mate/                  # MatE — matmul engine (mate_pv, mate_pv_fp16, mate_qkt).  → mirror lambda-mate
│   └── sw/ rtl/ pdk/ docs/ research/
├── vecu/                  # VecU — decode softmax slice (vecu_softmax, +rope/rmsnorm later). → mirror lambda-vecu
│   └── sw/ rtl/ pdk/ docs/ research/
├── precision_controller/  # per-tile INT8/FP16 precision gate.  → mirror lambda-precision-controller
│   └── sw/ rtl/ pdk/ docs/ research/
├── docs/  research/       # ACU-level
└── README.md              # → umbrella mirror lambda-acu
```

## When importing (checklist)
1. Clean the `attention-compute-unit` repo first (Step 0 of `docs/repo_reorg_plan.md`): archive the
   RL research (`phase1_policy/`, `phase2_kernel/`, `kv_cache/`, `common/`) into `research/`; keep
   the hardware (`rtl/`, `openlane/`, `orfs/`, `sw/reference_model/`, `docs/isa/`).
2. `git subtree add --prefix=acu/<sub> <clone> <merged-branch>` per sub-block, preserving history.
3. Re-enable the `acu`, `acu/mate`, `acu/vecu`, `acu/precision_controller` rows in
   `.github/workflows/mirror-blocks.yml` and create their mirror repos.
4. Add each sub-block's `AGENTS.md` / `DECISIONS.md` / `research/` / `## Known gotchas`.
5. Migrate the ACU-level decisions currently parked in `acu/DECISIONS.md` into
   `acu/mate/DECISIONS.md` and `acu/vecu/DECISIONS.md`.

The cross-block cosim in `chip/verif/` currently vendors copies of the ACU block RTL
(`chip/verif/blocks/acu/`); once `acu/rtl/` is the source of truth, re-point the cosim Makefile at it.
