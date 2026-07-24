# Lambda — Architecture Status

> **⚠️ Timeline of record is [`docs/ROADMAP.md`](ROADMAP.md).** The tapeout schedule below (any
> "Summer 2027" / charter-schedule mentions) is superseded by the canonical two-track roadmap:
> proxy (GF180/Chipathon) runs in parallel with the N16 product track; **N16 tapeout ~Fall 2027**
> (+1 semester buffer). Where this file and `ROADMAP.md` disagree on dates, `ROADMAP.md` wins.

> **Codec-of-record note:** The KV-cache compression codec is **ChannelQuant** (per-channel INT4 keys, grouped G=128, D per-channel FP16 scales; per-token INT4 values; static top-k k=2 FP16 outlier-channel lane via calibrated ROM mask; tiers CQ-8 / CQ-4 / CQ-4+; ~3.8× KV compression at ~4 bits/value, near-lossless), packaged as the **KV Cache Engine (KVE)**. Decompression is per-channel `INT4·FP16` (+ FP16 replay for outlier channels); keys are dequantized per-channel before the score matmul (no compressed-domain read path). The full block RTL is complete through Sky130 sign-off in the `kve` block. The KVE's **physical PD numbers at 16nm (area / power / Fmax) are TBD — pending re-measurement for ChannelQuant**, and the per-model throughput/capacity/accumulator numbers derived for the retired 3-5B target are **TBD — pending re-derivation for ChannelQuant / Qwen2-1.5B**. TurboQuant remains a cited prior work only; its pure history is on the `legacy/turboquant` branch.

**Last updated: 2026-07-21 (RTL-maturity honest-status refresh + CQ-3-rot value tier; see the change log below. Prior tooling entry: 2026-06-06 chamber v0.4.1.)**

## Change log

- **2026-07-20 (CQ-3-rot value tier + RTL moved to `rtl` branches):** Two org-wide changes.
  **(1) CQ-3-rot** — a fixed Walsh-Hadamard rotation of each per-token VALUE row before quant
  (idea: **Abhiram Bandi + Chaithu Talasila**) drops values from 4 to a flat **3 bits/value**,
  ~4.8× KV compression (~15×→~19× stacked with the TIU), near-lossless (~0.005 of FP16 on
  Qwen2-0.5B/1.5B + Llama-3.2-1B), no per-model calibration. Keys are **untouched** (per-channel)
  — the rotation only mixes a token's own dims, so it can't smear the key structure that sank
  TurboQuant+ (whose *value* rotation was innocent; only its *key* rotation was guilty). Reference
  codec bit-exact to the RTL (348,160/348,160 on real Qwen); it retires the TIU `tier_keep`
  value-precision lever (flat 3-bit ⇒ nothing to select). Merged non-RTL to the block mains;
  value RTL (`wht_unit`/`cq_wht_value`/`wht_inverse_out`, Path B) on the kv-cache-engine `rtl`
  branch. See `kv-cache-engine/docs/wht_value_rotation.md`, `arch.yml` codec-of-record note.
  **(2) RTL → `rtl` branch per block repo.** All SystemVerilog/testbenches/OpenLane/golden
  vectors + the RTL & reference-model CI gates now live on a dedicated `rtl` branch in each
  block repo; `main` carries docs/analysis/reference-source/paper only. Rationale: we claim TSMC
  16nm but the RTL/sign-off is **130nm (Sky130)** — the best open-PDK proxy for 16nm *estimates*,
  so it is kept off `main` (each `rtl` README states this). Shared CI gained a `has-rtl` toggle;
  main callers set `has-rtl:false`. See `.github/CLAUDE.md` and `docs/documentation_standard.md` §0.

- **2026-07-20 (MatE P·V accumulator width re-derived for Qwen2 dims):** The INT24 K-axis accumulator margin was flagged "TBD, pending re-derivation for ChannelQuant/Qwen2-1.5B." Re-derived it (`attention-compute-unit/analysis/pv_accumulator_width.py`) and found the two integer reduction axes need **different** widths. **(A)** W4A8 GEMM + Q·Kᵀ reduce over the hidden/head dim (≤4096) → 10-bit product × 4096 = 22b → **INT24 is correct**. **(B)** The INT8 P·V path reduces over the **token** dim, so its width scales with context: a maximally flat causal row of length L makes every P_int≈127 → worst case 127·127·L = 16129·L → 14+ceil(log2 L) bits. INT24 guarantees no overflow only to **~520 tokens**; **INT32** guarantees it to ~133k. Empirically on real Qwen2-1.5B the P·V accumulator stays at ~2²¹ (21b) and does **not** grow with context (599k@1k → 711k@4k) because real softmax is peaky — so INT24 works in practice — but a corner-case flat tile silently corrupts an output, so **the P·V accumulator is specced INT32**. Resolves the MatE README open-question #2 ("INT32 vs INT24 for headroom" — it's a hard requirement on the token axis, not headroom). Updated `arch.yml` (`pe_op_attention_pv`, `accumulator_rationale`, `key_decisions`) and `src/blocks/mate/README.md`.

- **2026-06-06 (v0.4.1 — launcher framework hardening, post-v0.4 audit):** **Closes the three structural gaps a v0.4 audit identified in the chamber-launcher framework**, with audit-grounded scoping (gap #3 RUNINFO sidecar + gap #5 ShellCheck CI deferred to v0.4.2 by design — STATUS is the cheap pass/fail signal, RUNINFO is the rich-metadata followup). **What landed (~263 new lines / 16 deleted across 7 files), branch `v0.4.1-launcher-audit-fixes` commit `77a7e42`, PR #3:** (a) `tools/lib/lambda-run.sh` adds `lambda_finalize_rundir <run-dir> <rc>` — writes `<run-dir>/STATUS` as `PASS|FAIL <UTC-ISO> rc=<n>`. Best-effort, never fails the run. (b) `lambda-stratus batch` retiered from a SHARED `$LAMBDA_WORK/<block>/stratus/` dir to per-invocation `<run-id>/` via `lambda_rundir batch` — closes the race where concurrent `cynth BASIC` + `cynth FALLBACK` from two shells collided on `bdw_work/` / `scverify_work/`. Also captures rc instead of exec, publishes `<run-id>/<CFG>/<b>.v` → `release/<b>.hls.v` via `lambda_publish_release` on rc=0 with MANIFEST entry (closes the Stratus→Genus release-contract gap; previously the Genus stub reached at a sibling `stratus/<CFG>/<b>.v` path). Then finalizes STATUS. (c) `lambda-genus` + `lambda-innovus` add one-line `lambda_finalize_rundir` after the existing publish block. (d) `lambda-xcelium sim` and `batch` drop `exec` so rc is capturable; finalize STATUS. `gui` keeps `exec` (no defined exit-success semantics — same asymmetry as Stratus/Genus/Innovus GUI modes). (e) `src/blocks/mate/genus/synth.tcl` skeleton comment updated: `read_hdl` now reads from `$LAMBDA_WORK/mate/release/mate.hls.v` (config-stable contract path), not `stratus/BASIC/mate.v` (sibling-dir reach). (f) `docs/tools-overview.md` "Directory dependencies and log/run dataflow" appendix added: per-tool reads/writes map, four resolution ladders (project-file / run-dir / storage / module), full back-end pipeline ASCII diagram, three-signal model (latest = invocation, STATUS = pass/fail, MANIFEST = published-artifact provenance). Pitfall list updated: gaps #1/#2/#4 closed; only Innovus→Pegasus edge deferred (PDK-gated, v0.5). **Three orthogonal signals now answer three distinct questions without grepping any tool log** — see "Reading the run-area at a glance" in `docs/tools-overview.md`. **Verification:** `bash -n` clean on all 5 modified shell scripts. ShellCheck not available in dev env (that's gap #5 itself — deferred to v0.4.2 as a CI workflow). Chamber-side smoke pending: 6-step test plan in the PR body covers per-runid isolation, STATUS pass/fail, concurrent shells (proves the gap-#1 fix), release/MANIFEST emission, cross-tool parity, diagnose regression. **Risk posture:** all interactive paths unchanged from v0.4. The retier touches only batch paths and adds new helpers; existing flows survive untouched. `latest` symlink semantics preserved as "most recent invocation" (debug tools depend on this — `lambda-verisium` follows `latest` to a crashed run's `waves.shm`). STATUS is the orthogonal pass/fail signal.

- **2026-06-06 (v0.4 — chamber flow infra, late session):** **Innovus brought up LIVE on a compute node + Genus/Xcelium/Verisium launchers + run-area architected outside the repo.** Live-debug session (compute node `ip-10-2-6-68`): Innovus Stylus GUI brought up, license `invs` checked out, *21.18* (not 25.14 — see pin reasoning below). This corrected **five v0.3 assumptions** and seeded v0.4: (1) **`/apps/<TOOL>` is autofs** — `module avail` lists the catalog; payload mounts only when `module load` + PATH-scan touches it; v0.3's "tools not installed" call was an `ls` of a cold automount. (2) **Tools live on COMPUTE nodes, not login** — `ae03ut01` carries only Virtuoso + vManager; digital + sim tools autofs on compute (`qsh -q normal.q -now n -V`). Launchers now detect this via `lambda_require_tool` and emit a clear hint. (3) **Pin three-level leaves matched to the installed family** — not because two-level fails (it doesn't; modules resolves to default leaf), but for reproducibility + matched DB family. Newest Genus is `genus/211/21.18.000`; Innovus matches `innovus/211/21.18.000` (catalog default `innovus/251`→25.14 would force cross-version handoff). Confirmed-installed set: stratus/2201/22.01.009, genus/211/21.18.000, innovus/211/21.18.000, xcelium/2403/24.03.005. (4) **Runs were dumping inside the repo mirror** — `~/architecture/build/` survived `git reset --hard` only because gitignored (luck, not design). v0.4 relocates to `~/work/lambda/` (NFS home, cross-node). (5) **`simvision` ships inside `XCELIUM2403`** — so the waveform GUI is free wherever Xcelium is, making Verisium's binary-name uncertainty a non-blocker. **What landed (~1,030 new lines, ~13 files changed/added):** (a) `tools/lib/lambda-run.sh` — shared helpers `lambda_require_tool` (autofs/compute-node aware), `lambda_rundir` (interactive vs `<utc-runid>` + `latest` symlink), `lambda_publish_release` (cross-stage handoff to `release/` + MANIFEST). (b) New launchers `genus-here`+`lambda-genus` (Common UI default; no `-stylus`), `xrun-here`+`lambda-xcelium`, `verisium-here`+`lambda-verisium` (Verisium primary, SimVision fallback). (c) Retrofitted `innovus-here`/`stratus-gui`/`stratus-batch`/`lambda-innovus`/`chamber-diagnose`/`lambda-diagnose` to use `lambda_require_tool` + load-then-test + node-type detection. (d) **Filesystem retier**: `LAMBDA_WORK=~/work/lambda` is the new run root; `LAMBDA_BUILD` aliased for back-compat; `LAMBDA_FAST=/tmp/$USER-lambda` ephemeral fallback. Crucial Tcl fix at `src/blocks/mate/stratus/project.tcl:46`: reads `$::env(LAMBDA_BUILD)` so Stratus follows the retier (Tcl doesn't see bash defaults). (e) `Makefile` expanded with `genus`/`sim`/`waves`/etc. + `$LAMBDA_WORK`-rooted FLOW DAG. (f) `tools/install.sh` provisions `~/work/lambda/{logs,inputs}` and writes a 2-line stub `~/work/lambda/Makefile` so daily use is `cd ~/work/lambda && make <target>`. (g) Genus synth stub `src/blocks/mate/genus/synth.tcl`. (h) Doc correctness: README + handoff "Calibre" → **Pegasus** (DRC/LVS), "PrimeTime" → **Tempus/SSV** (STA) — this is an all-Cadence chamber. `docs/tools-overview.md` gets new "Chamber execution model" and "Filesystem & run-area architecture" sections. **Risk posture:** authored off-chamber; first compute-node run is the smoke test. Interactive paths use only verified flags + correct pins. Two unverified items isolated with named fallbacks — `verisium` binary name falls back to `simvision`; `xcelium/2403` may need a pin-bump on other nodes (per-user override in `~/.longhorn/lambda.env`).

- **2026-06-06 (v0.3, earlier same day):** **Chamber scoped over SFTP + back-end tooling v0.3 landed.** (a) **Chamber map (SFTP enumeration).** The Cadence University hosted chamber (`10.2.6.6:222`, multi-tenant: OkState/aamu/howard/nd/pvam/muse) is a **list-and-create-only SFTP dropbox — file reads (`get`) and SSH exec are both blocked**, even on one's own files. So scripted access can map structure (names/sizes/mtimes/modes) but cannot read contents or run tools; anything real needs an interactive ETX/X11 session. Full Cadence flow present: `stratus/{...,2402}`, `xcelium/2509`, `genus/211`, `innovus/251`, `pegasus/251` (DRC/LVS), `ssv/251` (Tempus/Voltus/Quantus STA), `virtuoso`, `jasper`, `confrml`. **No TSMC PDK:** `/process/hosted` has only `gpdk` (incl. `advgpdk` = `cds_ff_mpt`, the only FinFET-class vehicle) + `skywater/sky130`. N16FFC must be delivered via `/process/hosted/xfer/incoming/` (admin-gated, TSMC University FinFET NDA) — this, not a "broken module," is the real back-end gate. **Doc correctness flag:** README/handoff name "Calibre (DRC/LVS) + PrimeTime (STA)" — neither exists on this all-Cadence chamber; the real signoff path is **Pegasus + Tempus/SSV + Quantus**. (b) **Tooling v0.3.** Added `tools/bin/innovus-here` + `lambda-innovus` (Innovus **Stylus Common UI**: `gui`/`shell`/`batch`/`diagnose`/`clean`), a root `Makefile` flow wrapper over the launchers, a stub `src/blocks/mate/innovus/setup.tcl`, and pinned `INNOVUS_MODULE=innovus/251` + `GENUS/PEGASUS/SSV` from the observed tree. Innovus runs foreground (it's a REPL, not a pure-GUI app like `stratus_ide`); interactive paths use only verified flags (`-stylus`/`-no_gui`/`-log`/`gui_show`), `-files` quarantined to `batch` with a documented fallback. **Authored off-chamber — the first ETX run is its smoke test** (the read/exec block means it cannot be tested from the sync side). (c) **Sync hygiene:** the Mac GitHub-Actions runner had been down since ~May 22, so the **May 29 push never synced** (queued 24h then auto-cancelled). Runner restarted 2026-06-06; the `10bab0b` bundle is now in `~/inbox/` awaiting `sync-promote`. See [`docs/tools-overview.md`](docs/tools-overview.md) for the v0.3 launcher + verified Innovus 25.1 Stylus syntax.

- **2026-05-17:** Chamber tooling framework v0.1 landed (commits `c6aa57d` → `7a5034f`, 9 commits). Two-layer launcher infrastructure: generic project-agnostic helpers (`tools/bin/stratus-{gui,batch}`, `chamber-diagnose`) plus Lambda-specific wrappers (`tools/bin/lambda-{stratus,diagnose}`) plus shared lib (`tools/lib/{lambda-env,lambda-detach}.sh`) plus idempotent installer (`tools/install.sh`). Stub `src/blocks/mate/stratus/project.tcl` committed so the launcher exercises end-to-end on day one. Chamber-specific resilience baked in: module init auto-detect via `$MODULESHOME/init/bash`, `/tmp` fallback when `/rscratch` isn't writable on compute nodes, defensive log-dir creation, correct `stratus_ide -project` flag, verified Tcl syntax for `define_hls_*` commands per ESP Columbia open reference. **Smoke test passed** on `ae03ut01` (utility) + compute node 2026-05-17: `lambda-stratus mate gui` opens IDE cleanly on the stub project. Eight chamber-bring-up issues iterated through and documented in `docs/tools-overview.md` "Lessons learned" table. Next: MatE PE HLS source (`pe.h/.cpp`, `mate.h/.cpp`, `tb/`) — Phase E work proper.

- **2026-05-14 (second pass):** Phase 0 arch updates applied. (a) HIF redesigned from USB-C 2.0 to **PCIe Gen3 x1 on M.2 2280 form factor** — adds +0.25 mm² area, gives 1.5 sec weight-load and standard form factor, uses vendor IP with public 16nm datasheets. (b) **ACU naming convention adopted** as the umbrella for MatE + VecU + KVE (honors Chaithu's framework). (c) **TIU (Token Importance Unit) block added** — 0.03 mm², modeled on arXiv 2604.04722, drives adaptive-precision KV and H2O-style eviction. (d) **Area accounting bug fixed** — routing_overhead_buffer was silently dropped from total in earlier drafts; gross area is now honestly 4.354 mm², with a recommended shrink path landing at 4.014 mm² contingent on PHY best case + I/O ring tightening + activation buffer trim. (e) Created `docs/literature_audit.md` (frontier attention/FFN survey, scaffolded) and `docs/reconciliation_chaithu.md` (shareable critique for teammate). Plan file at `~/.claude/plans/proud-yawning-hopcroft.md`.
- **2026-05-14 (first pass):** 8 spec bugs corrected (KV math, pre-pivot KV-codec bpe, MatE accumulator, headroom claims, SRAM/buffer sizes); repo restructured to single canonical chip target (v1/v3 retired); added `STATUS.md` LPDDR5X-vs-LPDDR4X tradeoff analysis. *(The KV-codec bpe fix has since been superseded by the ChannelQuant codec of record — ~4 bits/value; see codec-of-record note.)*

This is the single live entry point for Lambda's architecture state. The canonical spec is `arch.yml`; the visual reference is `floorplan.html`; the dataflow teaching doc is `dataflow_walkthrough.md`. Everything else has been cleared.

---

## 1. State in one paragraph

Lambda is a **4 mm² (2×2 mm) standalone transformer-decoder ASIC on TSMC 16nm FinFET (N16FFC)**, targeting tape-out via IMEC / Europractice mini@sic 2.0 (~$60–100K shuttle, $170–290K total chip cost including PHY IP). It runs **up to 1.5B-parameter W4A8 LLMs, validated on Qwen2-1.5B**, at **6–8 tok/s decode in a ~2.6 W envelope** (~3.3 W peak), with ChannelQuant KV compression (per-channel INT4 K + per-token INT4 V + static top-k FP16 outlier lane; ~3.8× at ~4 bits/value) in silicon. **Seven on-die functional blocks** grouped under top-level ACU/MSC/LSU/TIU/HIF: MatE 8×8 INT8×INT4 systolic, VecU 8-lane FP16/BF16 SIMD, KVE ChannelQuant KV codec (the three under the ACU umbrella; KVE = Block 2, formerly the KCE block); MSC memory controller (canonical Memory Hierarchy Controller / MHC) with vLLM-style 128-entry block table + sparse-blocked attention CSR; LSU layer sequencer; TIU entropy-based adaptive-precision driver (new 2026-05-14); HIF PCIe Gen3 x1 on M.2 2280 form factor (revised from USB-C 2.0 on 2026-05-14). Plus 0.8 MB SRAM in four banks + LPDDR5X x16 PHY (vendor IP). Canonical schedule: Spring 2026 Charter & tooling (in progress) → Fall 2026 Architecture finalization → Spring 2027 RTL design freeze → **Summer 2027 Tapeout** (TSMC 16nm via imec / TSMC University Program) → Post-silicon bring-up & validation.

---

## 2. Iteration history

| Date | Pivot | What we left behind | Why |
|---|---|---|---|
| **2025-12** | Project starts on the 130nm Sky130 track (SkyWater Sky130 PDK, via Caravel) | — | Free PDK + open shuttle. This 130nm work now lives in the separate [Chipathon](https://github.com/LonghornSilicon/Chipathon) repo; it was never fabricated. |
| **2026-02** | 130nm Sky130 design space narrows to 4 candidate architectures (A2/A3/A3+/A4) | — | KV-codec block emerges as the headline IP |
| **2026-03** | Pivot from Sky130 to TSMC N16FFC, codename Lambda; team retired the "BEVO" working name | 130nm Sky130 track (now in the Chipathon repo) | Sky130 capacity caps at ~1B-class; Caravel ring overhead at Sky130 is large; 16nm gives 28× density and FinFET energy. **KV-codec block carries forward intact.** |
| **2026-04** | Pivot from 25 mm² Lambda flagship to 4 mm² Lambda v2 + v1 dual-candidate at IMEC mini@sic 2.0 | 25 mm² flagship | Flagship shuttle cost ~$400-500K — unfundable on academic timeline; 4 mm² serves the ≤1.5B model class at 1/6 area, 1/5 cost via LPDDR5X x16 bandwidth tier |
| **2026-04 → 05** | Three architecture candidates at 4 mm²: v1 (KV coprocessor, no LPDDR), v2 (standalone with LPDDR5X x16), v3 (all-SRAM tiny-LLM, 2-10M params) | — | Each addressed a different demo story / risk profile |
| **2026-05-13** | **Lambda v2 selected as the headline architecture and overall arch.** v1 and v3 retired. | v1, v3 | v2 is the only path that ships a demo-able standalone ≤1.5B transformer accelerator without requiring a CPU-runtime software stack (v1) or capping at sub-100M-param models (v3). v1's KV-codec-only architecture is folded into v2 via the KVE block; v3's all-SRAM idea is preserved as a future low-power variant if a sponsor asks |
| **2026-05-14** | **Pre-RTL audit completed.** 8 bugs in spec corrected (see §4 below). Repo restructured to single-arch focus. | scripts/v2_design_space, archs/_shared, the retired 130nm-track dir (now the Chipathon repo), PRDs/, roadmap.md, archs.yaml, v1/v3 YAMLs | Single source of truth before HLS work begins |

---

## 3. Current architecture summary

| Block | Area | Function |
|---|---|---|
| MatE — 8×8 INT8×INT4 weight-stationary systolic | 0.10 mm² | All GEMMs (Q/K/V proj, FFN, logits) + Q·K^T in output-stationary mode against K dequantized per-channel by the KVE (INT4·FP16 → FP16; no compressed-domain read path). **INT8 × INT4 → 11-bit product (INT16 partial register inside PE) → INT24 K-axis accumulator.** Peak 128 GOPS at 1 GHz. |
| VecU — 8-lane SIMD with online softmax | 0.144 mm² | RoPE, RMSNorm, SiLU, online-softmax recurrence (Milakov & Gimelshein 2018; FlashAttention-style tiling), residual add, sampling. 1K-inst microcode. |
| KVE — KV Cache Engine (ChannelQuant; Block 2) | *(area TBD — pending re-measurement)* | **Codec of record: ChannelQuant** — per-channel INT4 keys (grouped, G=128, D per-channel FP16 scales) + per-token INT4 values (INT8 in CQ-8) + static top-k k=2 FP16 outlier-channel lane via calibrated ROM mask; unified per-channel SRAM record {tag, D×FP16, D×INT4}; one shared fp16 scale/quant/dequant unit serialized across D channels (single divide cone). Decompress = per-channel `INT4·FP16` (+ FP16 replay for outliers); keys dequantized per-channel before the score matmul. Tiers CQ-8 / CQ-4 / CQ-4+; ~3.8× KV compression at ~4 bits/value (measured 4.13–4.38 by head dim D), near-lossless (HellaSwag within ~0.4–0.8 pt of FP16 at CQ-4+, Qwen2-0.5B/1.5B). Recipe follows KIVI (ICML 2024) / KVQuant (2024). Full block RTL complete through Sky130 sign-off in the `kve` block. Physical PD numbers at 16nm are TBD. |
| MSC — Memory Subsystem Controller (maps to canonical Memory Hierarchy Controller / MHC, Block 4) | 0.18 mm² | LPDDR5X x16 controller + 4-port SRAM crossbar + 128-entry block table (vLLM-style PagedAttention in silicon) + DMA descriptor FSM. Single-session — no continuous batching, no Tier-3 eviction. |
| LSU — Layer Sequencer | 0.10 mm² | In-order RISC, 32-instruction ISA, 4 KB microcode RAM holding pre-compiled model schedule. Single-issue scalar + vector + DMA per cycle. |
| TIU — Token Importance Unit (Block 3) | 0.03 mm² | Per-block 16-bit attention-entropy accumulator (256 B SRAM); drives MSC eviction (H2O-style) and KVE per-block precision mode. Modeled on arXiv 2604.04722. **NEW 2026-05-14.** |
| HIF — PCIe Gen3 x1 (M.2 2280) | 0.55 mm² | PCIe Gen3 x1 endpoint (~1 GB/s sustained) for CSR access + microcode load + token I/O. M.2 form factor — slot wires 4 lanes, on-die PHY drives x1 (negotiates down). JTAG via dedicated pins. **Revised from USB-C 2.0 on 2026-05-14.** |
| **On-chip SRAM (0.8 MB)** | 0.71 mm² | kv_scratchpad 0.4 MB · activation_buffer 0.3 MB · weight_stream_buffer 0.05 MB · codebook_const_rom 64 KB (holds ChannelQuant outlier-channel ROM mask + RoPE + LUTs; id kept for HLS continuity) |

### RTL maturity (honest status, 2026-07-21)

The areas above are **16nm analytical estimates**; they do **not** imply the block exists as
synthesizable RTL. Actual RTL status:

| Block | RTL status |
|---|---|
| **KVE** | RTL-complete, Sky130 sign-off (`kv-cache-engine` repo) |
| **TIU** | RTL-complete, Sky130 sign-off (`token-importance-unit` repo) |
| **ACU precision controller** | RTL, Sky130 sign-off (`attention-compute-unit` repo) |
| **MatE — P·V tile** | RTL, Sky130 sign-off — `mate_pv` (INT8) + `mate_pv_fp16` (FP16 escape). *Only the P·V vector-MAC.* |
| **MatE — Q·Kᵀ decode scoring** | **RTL** — `mate_qkt` (INT8 Q × per-channel FP16 K → L scores), bit-exact to `mac_array_ref` sequential-fp32 golden, live in the cosim BLOCK 1 (Phase 1 done 2026-07-21, `attention-compute-unit` `rtl` `93e9960`). **Sky130 sign-off** (9-corner, DRC/LVS/antenna 0, residual ss slew) + **GF180-hardened** (gate-level e2e verified). |
| **MatE — 8×8 systolic array (general GEMM/FFN)** | **no RTL** — off-chip for the shuttle; the general weight-stationary GEMM/FFN engine is a separate later program. |
| **VecU — decode online-softmax** | **RTL** — `vecu_softmax` (64-entry exp LUT + linear interp + online running-max/running-sum recurrence, fp32 accumulator → fp16 weights), bit-exact to `sw/reference_model/vecu_softmax_ref.py` (LUT golden ≈2% vs exact fp64 softmax), live in the cosim BLOCK 2d (Phase 2 done 2026-07-21, `attention-compute-unit` `rtl` `4a30d93`; multi-cycle revision `2c458aa`). **Sky130 sign-off** (9-corner, 105 ns/9.5 MHz, DRC/LVS/antenna/max-cap 0) + **GF180-hardened** (multi-cycle, ss +60.9 ns @ 260 ns; gate-level e2e verified). RoPE / RMSNorm slices still pending (chip-top raw-Q/K path). |
| **MSC / LSU / HIF** | spec-level; not in the decode-attention-datapath tapeout boundary. |

The cross-block cosim (`rtl/tb/tb_chip_cosim.sv`) verifies the RTL blocks end-to-end on real
Qwen tensors. The decode attention pass **Q·Kᵀ → softmax → P·V is now all real RTL**
(`mate_qkt → vecu_softmax → mate_pv_fp16`); the remaining stand-ins are RoPE / RMSNorm (the
loaded Qwen tiles are already RoPE'd, so they only matter for the chip-top raw-Q/K path). See the
[chipathon RTL-closure plan](docs/chipathon_rtl_closure_plan.md). "In silicon" language elsewhere
in this doc refers to the KV-compression path (KVE), which is the block that is actually RTL-signed-off.
| **LPDDR5X x16 PHY** (vendor IP) | 1.20 mm² ±0.3 | Synopsys DesignWare or Cadence Denali; NDA-gated; load-bearing area uncertainty |
| I/O ring + pads + ESD | 0.76 mm² | 100 µm ring, 2 kV HBM ESD |
| Clock + power + routing | 0.50 mm² | ~12.5% of die at 16nm |
| **GROSS TOTAL (PHY @ 1.2 mm² mid-case, no shrinks)** | **4.354 mm²** | over budget by 0.354 mm² — 2026-05-14 audit caught earlier 3.974 figure dropped routing_overhead_buffer |
| **WITH RECOMMENDED SHRINKS** (PHY @ 1.0 best-case + 80 µm ring + activation buffer 0.2 MB) | **4.014 mm²** | within 0.4% of 4.0 mm² target — contingent on Q2 2026 PHY quote |

Off-chip: 1× LPDDR5X-8533 x16 package (4–8 GB capacity, mobile-grade, ~$5–15 BoM, holds 1.5 GB W4 weights for a 3B model plus scratch). Plus the M.2 2280 carrier card with the Lambda die mounted next to the LPDDR5X package on PCB.

---

## 4. Pre-RTL audit log — bugs fixed on 2026-05-14

The 2026-05-14 audit (before HLS work begins) caught 8 bugs in the spec. All are corrected in `arch.yml` and `floorplan.html`:

1. **KV bytes/token off by ~2.3×.** Spec dropped the K+V factor of 2 in the per-token byte formula AND used flagship's 3.5 bpe instead of the pre-pivot codec's 4.0 bpe. Example: Llama-3.2-3B per-token-per-layer was claimed 448 B (correct value 1024 B). Capacity claims like "Qwen2.5-3B 32K context fits on-die per layer" propagated from this error. *(Superseded: with the ChannelQuant codec of record, per-value cost is ~4 bits/value and all capacity numbers are TBD — pending re-derivation for ChannelQuant / Qwen2-1.5B.)*

2. **"Qwen2.5-3B 32K on-die" claim was wrong.** With corrected math, 32K context for any v2 target model requires per-layer KV well beyond the 0.4 MB scratchpad. Reframed: **32K is serviceable via LPDDR streaming at ~24 ms/tok overhead** (Qwen2.5-3B 2-head layout). Qwen2.5-3B remains the best long-context target on Lambda due to its 4× lower per-token KV vs 8-head peers.

3. **"INT16 accumulator" in MatE was a real bug.** INT8 × INT4 → 11-bit signed product; reducing K=128 (head_dim) needs 18 signed bits — INT16 (max ±32767) saturates after ~64 accumulations in the worst case. **Corrected to INT24 K-axis accumulator** with INT16 partial-product register inside each PE.

4. **"3-5× compute headroom" was wrong.** In the bandwidth-bound regime, GOPS_needed scales with bandwidth (not model size), so headroom is **constant 1.6× across all reasonable models**. Earlier framing implied headroom grew with smaller models — fortunate phrasing of a wrong derivation.

5. **"Comfortable headroom at PHY=1.0 mm²" was misleading.** The sensitivity sweep shows the chip *exactly* fits at 1.0 MB SRAM with **+0.002 mm² headroom** — no margin for surprise. SRAM stays at 0.8 MB baseline; 1.0 MB is contingent on PHY landing ≤ 1.0 mm².

6. **weight_stream_buffer was inconsistent** (0.05 vs 0.15 MB in two sections of the same YAML). Canonical: **0.05 MB** (40× LPDDR latency-hiding minimum at 12 GB/s × 100 ns first-byte).

7. **Total SRAM was inconsistent** (0.8 / 0.85 / 1.0 MB across sections). Canonical baseline: **0.8 MB** at PHY=1.2 mm²; upgradable to 1.0 MB if PHY lands ≤ 1.0 mm².

8. **KV-codec bpe values inconsistent** (3.5 / 4.0 / 5.3 across sections) in the pre-pivot codec. *(Superseded by the ChannelQuant codec of record: ~4 bits/value (measured 4.13–4.38 by head dim D) → ~3.8× vs FP16. The old 16-pt/3-bit bpe reconciliation no longer applies.)*

**No architectural changes** — every fix was math, derivation, or stale-claim. Audit traceability preserved in the YAML as "earlier drafts claimed X" comments next to each correction.

---

## 5. LPDDR5X x16 vs LPDDR4X x16 — the real tradeoff

> **PENDING RE-ANCHOR (post-2026-06-22 pivot):** the model-class arguments in this section are framed around the old 3–5B / 3–4B / 1–2B target and the TurboQuant codec. The canonical target is now **up to 1.5B parameters (validated on Qwen2-1.5B)**, and the codec is ChannelQuant. The bandwidth-vs-model-class reasoning below (4.8B vs 2.4B, "3B chip vs 1B chip") no longer maps cleanly onto a ≤1.5B target and needs human re-derivation. Numbers and model names retained verbatim below as the pre-pivot analysis of record; do not treat them as the current target.

You asked whether to consider Cadence LPDDR4X over Synopsys/Cadence LPDDR5X to get area breathing room + better public datasheets. **My read: it's a defensible Plan B but should not be the primary choice yet.** Here's the math.

### The two options side-by-side

| Property | **LPDDR5X x16 (current baseline)** | **LPDDR4X x16 (Cadence fallback)** | Delta |
|---|---|---|---|
| Peak bandwidth (8533/4266 Mbps × 16/8) | 17 GB/s | 8.5 GB/s | ½× |
| Sustained at 70% | **12 GB/s** | **6 GB/s** | ½× |
| PHY area at 16nm | ~1.2 mm² (NDA est.; ±0.3 swing) | ~1.0 mm² (~±0.15 swing) | −0.2 mm² + tighter variance |
| Power (mW/Gbps × Gbps sustained) | 7.5 × 96 = **0.72 W** | 12 × 48 = **0.57 W** | −0.15 W (lower BW → lower power) |
| Largest model at 5 tok/s ceiling | **4.8B** | **2.4B** | **½× the model class** |
| Comfortable model class (8 tok/s) | 3–4B (Llama-3.2-3B, Mistral-NeMo-3B) | 1–1.5B (Llama-3.2-1B, Qwen2.5-1.5B) | 2× class drop |
| Cadence public collateral at 16nm | thin (NDA-only) | **thick** (reference designs, app notes, Linley reports) | major asymmetry |
| Synopsys public collateral at 16nm | thin | thick | same asymmetry |
| First-FinFET-PHY risk for the team | **high** (R-Lv2-01) | lower — LPDDR4X has been shipping at 16nm since ~2018 with documented references | major de-risk |
| Area headroom at 4 mm² (post-Phase-0, with PCIe + TIU) | −0.354 mm² gross → +0.014 mm² after shrinks (PHY best case) | **−0.154 mm² gross → +0.046 mm² after shrinks** (LPDDR4X PHY at 1.0 mm² instead of LPDDR5X at 1.2 mm² saves 0.20 mm²) | LPDDR4X gives ~0.2 mm² more breathing room — the load-bearing argument |
| Decode regime | bandwidth-bound everywhere | bandwidth-bound everywhere | same |
| Compute headroom over BW floor | 1.6× | 3.2× | LPDDR4X leaves compute idle |

### How to think about it

There are really three axes:

**(a) Model class.** 4.8B vs 2.4B is the load-bearing difference. At 4.8B you can demo Llama-3.2-3B, Mistral-NeMo-3B, Qwen2.5-3B at 6-8 tok/s — recognizable, deployable models. At 2.4B you're capped at Llama-3.2-1B / Qwen2.5-1.5B / Phi-3-mini-1.3B — still useful but the headline pitch changes from "3B LLM on a 4 mm² chip" to "1B LLM on a 4 mm² chip." The 1B class is also where mobile NPUs (Apple, Qualcomm) commoditize; the 3B class is what differentiates an open-source academic chip from a Snapdragon/A-series NPU.

**(b) PHY risk.** LPDDR5X x16 at 16nm has fewer public reference designs than LPDDR4X at 16nm. The vendor will quote both confidently, but the team's *first FinFET PHY* tape-out is concentrated risk no matter who quotes. LPDDR4X has demonstrably shipped at 16nm in multiple academic and commercial parts; LPDDR5X x16 at 16nm is less well-documented in the open literature. If the team doesn't have or can't recruit a senior FinFET PD engineer with DDR PHY experience, the LPDDR5X risk is real.

**(c) Area.** LPDDR4X frees ~0.2 mm² that can go to SRAM (1.0+ MB feasible) or to a 12×12 MatE (180 GOPS, useful at lower BW). The chip becomes objectively easier to floorplan with comfortable margins instead of zero-headroom.

### My recommendation

**Don't pre-commit. Get both quotes in parallel.** Specifically:

1. **Synopsys + Cadence LPDDR5X x16 quote** (the primary). Two vendors so you have leverage and comparison.
2. **Cadence LPDDR4X x16 quote in parallel** (the explicit Plan B). The marginal effort to ask for this alongside is low and the data is decisive.

Then apply this decision rule:

- **If LPDDR5X x16 quote ≤ 1.25 mm² with a senior FinFET PD engineer in-place by Q3 2026 → go LPDDR5X.** The 3-5B model class is the publishable story.
- **If LPDDR5X x16 quote > 1.35 mm², OR no senior FinFET PD engineer is available → go LPDDR4X.** Recover the 0.2 mm² of breathing room and the public-collateral certainty. Reframe headline as "first open-source academic standalone 1-2B transformer accelerator at 4 mm²" — still a publishable first.
- **In the middle (1.25 < quote ≤ 1.35) → faculty/sponsor call.** Weigh "publish a 3B-class chip" against "publish a chip that taped out clean."

The case for LPDDR4X is *risk and certainty*, not *intrinsic preference*. The case for LPDDR5X is *model class and headline story*. Both are defensible; the data gates the decision.

### One thing to also try if going LPDDR4X

If you do go LPDDR4X, the model-class loss can be partially recovered via:
- **W3 weight quantization** (3-bit weights instead of 4-bit). Recent ICLR'26 results (TurboQuant, QuaRot) show 3-bit weights are usable for 3-4B-class with minor MMLU loss. Doubles the model class on the same bandwidth → 2.4B → ~4B at 5 tok/s on LPDDR4X. But it's not yet a sweet spot the same way W4A8 is.
- **Asymmetric K3/V2 KV compression** (already in the spec as a CSR mode). Frees more on-die KV capacity, less help on bandwidth.
- **Speculative decoding** (architecturally compatible; host coordinates draft tokens). 1.5–2× effective tok/s for the same bandwidth.

So even on LPDDR4X, "3-4B at 5 tok/s with aggressive quant + spec decode" is reachable. But it stacks more research risk.

---

## 6. Open questions blocking progress

In strict order of how much they gate the next decision:

1. **IMEC / Europractice mini@sic 2.0 quote for 4 mm² N16FFC.** Email `eptsmc@imec.be` price-request form. Single most pressing number. *Owner: architecture lead + faculty advisor. Deadline: 2026-05.*

2. **LPDDR PHY quotes — three at once.** Synopsys LPDDR5X x16, Cadence LPDDR5X x16, Cadence LPDDR4X x16 — all at 16nm. Pull the trigger on the LPDDR4X-vs-LPDDR5X decision per §5. *Owner: architecture lead. Deadline: 2026-06.*

3. **Senior FinFET PHY PD engineer recruited (or partnered).** Hard prerequisite for LPDDR5X path. *Owner: architecture lead + faculty advisor. Deadline: 2026-07.*

4. **Demo target model locked.** Canonical target is up to 1.5B parameters, validated on **Qwen2-1.5B**. Run Python golden-model quality eval (MMLU, LongBench) at W4A8 + ChannelQuant (CQ-4 / CQ-4+ tiers). *Owner: ML student. Deadline: 2026-07.*

5. **Tool-ramp bring-up vehicle at IMEC mini@sic.** Trivial inverter ring oscillator to clear DRC/LVS in the **Cadence Innovus + Pegasus** flow (all-Cadence chamber; Calibre is NOT installed) before Lambda RTL begins. *Owner: PD lead. Deadline: 2026-09.*

6. **HLS C++ implementation begins in `src/`.** Cadence Stratus HLS as the synthesis path. MatE PE and the KVE ChannelQuant datapath are the long poles — start there. Python golden model in parallel for bit-exact reference. *Owner: RTL lead + ML student. Deadline: 2026-08 for first PE.*

---

## 7. Frontier work to plan next

These are the architecture deep dives queued behind the bug fixes and cleanup. They are scoped in plan-mode (see the next interaction):

- **~~Adaptive-precision reconciliation~~ — RESOLVED 2026-07-18.** The precision-controller + MAC-array work (`LonghornSilicon/attention-compute-unit`) assumes a heterogeneous INT8/FP16 MAC with a per-tile gate. Earlier drafts had MatE as INT8×INT4 only. **Decision: MatE gains the FP16 MAC path** — weight/FFN GEMMs stay W4A8 (INT8×INT4), but attention `P·V` routes per-tile INT8/FP16 via the ACU precision controller (FP16 for peaked tiles). See `arch.yml` `matrix_engine` (`pe_op_attention_pv`, `precision_control`), compiler guide §6, and `csr_map.h` `mate_precision` (default `adaptive`). Remaining: re-synthesize MatE with the FP16 mode to measure the area/power delta (TBD).

- **Attention/FFN mechanism deep dive.** PagedAttention (vLLM), FlashAttention-2/3, sparse-blocked attention, MLA (DeepSeek), GQA, MQA, batched-grouped attention. Lambda currently commits to FA-3 + paged-attention + GQA/MQA via MSC. Audit each for what's actually frontier vs what's reasonable middle ground; identify hardware implications.

- **Etched patent (US 2024/0419516 A1) implications.** Etched splits the systolic array (no previous-token dependency) from a separate self-attention circuit (uses previous-token data). Lambda's MatE multiplexes both via dataflow mode. Is the patent's split right at our scale (4 mm², 64 PEs), or does multiplexing dominate at this die size? Hardware schedule analysis required.

- **Research corpus alignment.** Cross-check the KVE ChannelQuant modes against KIVI, KVQuant, TurboQuant, QuaRot, RotateKV, Oaken, Titanus, GEAR, Lexico. Identify the *next-gen mode* worth adding as a CSR option for the chip's research narrative.

---

## 8. Risks that still worry me

**Non-architectural:**

- **IMEC mini@sic 2.0 actual pricing for 4 mm² N16FFC.** Pricing is non-public; ranges are academic-discount-tier estimates. A $120K or $150K return would not kill the project but would change the funding plan.

- **The LPDDR PHY tape-out itself.** First FinFET-era PHY for the team. Senior PD engineer or partner is a hard prerequisite (R-Lv2-01). LPDDR4X fallback (§5) materially de-risks but at the cost of model class.

**Architectural / research-claim:**

- **INT24 K-axis accumulator margin needs re-derivation for the target model.** The accumulator width was originally derived from the retired 3B dims (e.g. FFN K up to 8192), where the asymmetric (−128 × −8) corner sits at the INT24 saturation edge. Under the canonical ≤1.5B / Qwen2-1.5B target the actual FFN-down K is smaller, but the exact margin is **TBD — pending re-derivation for Qwen2-1.5B**. **Fallback:** INT26 or INT28 K-axis accumulator (logic effectively free at 16nm; cost is wire width on the column-output reduction tree, not gate count). Mitigation: Python numerical sweep against real Qwen2-1.5B FFN-down activations pre-HLS, plus an `arch.yml` rationale rewrite to cite the target model's K. Tracked in `docs/handoff.md` §3.2 #2.

- **ChannelQuant KV-compression quality — characterized, but silicon PPA still to be measured.** The codec is *not* a rotation-codebook / compressed-domain scheme: keys are per-channel INT4 (grouped G=128, D FP16 scales) with a static top-k k=2 FP16 outlier lane, values per-token INT4, and decompression is per-channel `INT4·FP16` before the score matmul. Accuracy is characterized in the `kve` block — HellaSwag acc_norm within ~0.4–0.8 pt of FP16 at the CQ-4+ tier on Qwen2-0.5B/1.5B, ~3.8× at ~4 bits/value. What remains is the **16nm physical PD measurement (area / power / Fmax), which is TBD — pending re-measurement for ChannelQuant**, and the target-model throughput/capacity re-derivation. **Fallback:** the `bypass_fp16` tier keeps the chip functional at 1× if a tier underperforms on the target model. Mitigation: MMLU + LongBench + Needle-in-Haystack on Qwen2-1.5B at W4A8 + ChannelQuant (CQ-4 / CQ-4+), owned by ML student, pre-HLS commit. Tracked in `docs/handoff.md` §3.3 #9.

All four are recoverable; none is a kill. The architecture is sound.

---

*Maintained as a live document. Append dated change-log entries when state shifts.*
