# rtl/ — scaffold (RTL lives on the revision branch)

**This is `main` — the clean scaffold. The synthesizable RTL (`.sv`/`.v`) is NOT here.**

RTL is developed by contributors on the **`rev0`** revision branch (and its successors) and merges
into `main` only when a lead blesses it. This is deliberate: Longhorn Silicon is a talent-development
effort — students and leads write the RTL, we do not ship generated RTL as the canonical baseline.

- **See the current RTL:** `git checkout rev0` (or the block's `rev0` mirror branch).
- **Contribute RTL:** branch from `rev0`, open a PR into `rev0`. See `docs/REVISION_SYNC_SOP.md`.
- **The spec you implement to:** this block's `docs/`, `sw/reference_model/` (bit-accurate golden),
  and `arch.yml`. The `pdk/` flow configs are ready to harden once the RTL exists.
