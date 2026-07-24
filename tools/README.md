# tools/ — Cadence-chamber launcher framework

The validated launcher infrastructure for the shared hosted **Cadence chamber** (all-Cadence flow:
Stratus HLS → Genus → Innovus → Pegasus DRC/LVS → Tempus/SSV STA; Verisium/SimVision debug).
Migrated 2026-07 from the `lambda-arch` repo (formerly `architecture/tools/`) into the implementation
monorepo, where the flows actually run. Distinct from the open-PDK OpenLane/LibreLane proxy flows,
which live per-block under `src/blocks/<block>/pdk/`.

```
tools/
├── bin/
│   ├── lambda-stratus / stratus-gui / stratus-batch   HLS (C++ → RTL)
│   ├── lambda-genus / genus-here                       synthesis
│   ├── lambda-innovus / innovus-here                   place & route (Stylus Common UI)
│   ├── lambda-xcelium / xrun-here                       simulation
│   ├── lambda-verisium / verisium-here                 waveform debug (SimVision fallback)
│   └── chamber-diagnose / lambda-diagnose              environment check
├── lib/   lambda-{env,run,detach}.sh                   shared helpers (autofs/compute-node aware)
└── install.sh                                          provisions the run-area (~/work/lambda)
```

**Chamber notes** (see `lambda-arch` STATUS change-log for the full bring-up history):
- Tools live on **compute nodes**, not the login node — `lambda_require_tool` autofs-detects and hints.
- Run-area is **outside the repo** (`~/work/lambda/`), cross-node NFS; `install.sh` provisions it.
- `latest` symlink = most recent invocation; `STATUS` sidecar = PASS/FAIL; `MANIFEST` = published artifacts.

Usage: `cd ~/work/lambda && make <target>` after `tools/install.sh` (see `lambda-arch` docs for the
per-tool run/log dataflow). The N16 hardening (private overlay) uses this chamber flow; the open-PDK
proxy hardening uses the per-block `pdk/` OpenLane/LibreLane configs.
