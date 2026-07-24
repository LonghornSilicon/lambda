# tools/ — chamber launcher framework

**Status: home for the Cadence-chamber launcher framework (to be reconciled in from the
`architecture` repo).** The validated launcher infrastructure — `lambda-stratus`, `lambda-genus`,
`lambda-innovus`, `lambda-xcelium`, `lambda-verisium`, `chamber-diagnose`, the shared
`lib/lambda-*.sh` helpers, and `install.sh` — currently lives in the `architecture` repo
(`tools/`), most recently as the v0.4.x launcher-audit-fixes wave. That work was captured
non-destructively on the `architecture` branch `rescue/2026-06-audit-uncommitted`.

**Why a stub here:** the reorg establishes the canonical structure now; importing the chamber
tooling is a deliberate reconcile step (it crosses the `architecture` → `lambda` boundary — see
[`../docs/REVISION_SYNC_SOP.md`](../docs/REVISION_SYNC_SOP.md) §7). The launchers target the shared
hosted Cadence chamber (all-Cadence flow: Stratus HLS → Genus → Innovus → Pegasus DRC/LVS →
Tempus/SSV STA), not the open-PDK OpenLane/LibreLane flows used for the sky130/gf180 proxy hardening
(those live per-block under `src/blocks/<block>/pdk/`).

Layout when populated (matches the `architecture` repo):
```
tools/
├── bin/   lambda-{stratus,genus,innovus,xcelium,verisium}, chamber-diagnose, *-here wrappers
├── lib/   lambda-{env,run,detach}.sh   (shared helpers)
└── install.sh
```
