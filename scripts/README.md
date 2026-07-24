# scripts/ — Revision-Sync tooling

Local tooling (stdlib Python 3, no repo secrets) that implements the
[Revision-Sync SOP](../docs/REVISION_SYNC_SOP.md). Run from anywhere in the repo — each script
locates the monorepo root itself.

| Script | What it does | Key modes |
|---|---|---|
| [`gen_progress.py`](gen_progress.py) | Derives the sign-off matrix from committed `*metrics*.json` and writes [`docs/PROGRESS.md`](../docs/PROGRESS.md). **Status is computed, never hand-copied** — this kills the drift class where a block advances but chip docs go stale. | `--check` (fail if stale), `--stdout` |
| [`check_block_structure.py`](check_block_structure.py) | Lints every block against the canonical template (SOP §5): required files/dirs, metrics filename schema, dangling PDK configs, ref-model minimum, DECISIONS date precision. | `--strict` (WARN also fails) |
| [`cut_revision.py`](cut_revision.py) | Cuts a coordinated cross-block revision: emits the `docs/REVISIONS.md` entry (block SHAs + ISA + sign-off) and the annotated `rev-RN[.5]` tag. **DRY-RUN by default.** | `--write`, `--tag` |
| [`rtl_doc_gate.py`](rtl_doc_gate.py) | Lab-notebook gate: a staged `rtl/` change must carry a staged `docs/`/`DECISIONS.md`/`README.md` change in the same block. | (pre-commit or standalone) |

## Sign-off classification (the shared contract)
`gen_progress.py` defines it and `cut_revision.py` reuses it. Per `docs/REVISION_SYNC_SOP.md` §5.2:
- **signed-off** — full-signoff flow (sky130/gf180), all headline checks 0, GDS present.
- **route-clean** — ORFS/ASAP7: route-DRC + antenna only, **no Magic-DRC/LVS**. *Not* full sign-off.
- **caveated** — full-signoff flow, GDS present, a headline check nonzero.
- **config-only** — a macro declared with no results ("declared, not run").
- **no-gds** — metrics present, no GDS committed.
- **prose-only** — GDS present, no machine-readable metrics JSON (e.g. chip full-chip today).

## Typical use
```bash
# after a PD run lands new metrics:
python3 scripts/gen_progress.py            # refresh docs/PROGRESS.md

# before pushing:
python3 scripts/check_block_structure.py   # structure lint (ERRORs block)
python3 scripts/gen_progress.py --check    # PROGRESS.md is current

# cut a revision (architecture/revision lead only, on a clean tree):
python3 scripts/cut_revision.py R1.5 --milestone "Full-chip GF180 assembly" --write --tag

# enable the commit-time gates once:
pip install pre-commit && pre-commit install
```

All four are also wired in [`../.pre-commit-config.yaml`](../.pre-commit-config.yaml).
