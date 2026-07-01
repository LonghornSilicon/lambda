# KVCE block — Lab Notebook

Dated, provenance-bearing entries for every parity run and synth result (per
`findings/channelquant_block_revamp.md` §8). These feed the joint paper's
hardware-evaluation section.

---

## 2026-06-22 — ChannelQuant algorithm handoff landed (verification unblocked)

The algorithm lane (`channelquant`) finished Phase 1 and handed over the contract
+ golden vectors. Vendored hermetically at `rtl/tb/testvectors/channelquant/`,
pinned to **channelquant commit `08d5287`** (`SOURCE_COMMIT`).

What landed (verified upstream before handoff):
- **`HW_CONTRACT.md`** — exact quant rule (round-half-to-even, clamp INT4
  [−8,7]/INT8 [−128,127], `EPS=2^-14`), fp16 scales, per-tier packing layout (§5),
  group-flush semantics (§3), static outlier-mask format (§4), parity gate (§8).
- **9 golden vectors** (`*.npz` reference truth + `$readmemh`-loadable `hex/`) —
  CQ-8/CQ-4/CQ-4+, full key group (g=G) + partial (g<G), D ∈ {64,128}, CQ-4+ k=2.
  Each carries inputs + expected packed payload + expected reconstructed K/V.
- Upstream verification: reference reproduces c17 bit-exactly (max |Δ|=0.000 over
  6 variants × Qwen2-{0.5B,1.5B,7B}, HellaSwag n=250); `torch`==`numpy` per tier;
  `.hex` round-trips bit-exactly to `.npz`.

Effect on this repo:
- `findings/channelquant_block_revamp.md` §1 flipped from *blocked* → **landed**;
  **P3 (3-way Python↔C++↔SV parity) is now startable** once a SV simulator is on
  PATH (`make sim` is still gated on that — see TEARDOWN.md banner).
- `rtl/TEARDOWN.md` header updated to point at the vendored bundle.

Open items to confirm with the channelquant lane before pinning parity (do **not**
guess — vendored README lists them): decompress-bus product format (fp32 exact vs
fp16 cast), final `G` (Phase-2 Pareto), CQ-4+-at-scale accuracy (Phase-3 n≥1000).

Next: P0/P1/P2 RTL (datapath teardown + value/key paths + outlier ROM) proceed on
the design side; parity (P3) consumes this bundle when the build host has a sim.

---

## 2026-06-22 — local SV simulator built (verified-build gate cleared)

This aarch64 host had no simulator and no passwordless sudo, so the toolchain was
built into a repo-local prefix (`/home/chaithu/lhs/.tools`, git-ignored):

- **iverilog/vvp 12.0** — built from the `v12_0` source archive. Bootstrap needed
  `gperf` (absent from conda-forge aarch64 as `iverilog` itself is), pulled via
  micromamba; `flex`/`bison`/`autoconf`/`gcc 13.3` already on the host. Recipe:
  `sh autoconf.sh && ./configure --prefix=… && make -j && make install`.
- **verilator + gperf** — micromamba env `eda` (conda-forge, linux-aarch64).
- Convenience: `. rtl/eda-env.sh` puts both on PATH.

**Validation:** `make sim` on the unmodified TurboQuant+ TB → **14/14 PASS**
(`tb_kv_cache_engine.sv:279 $finish`). Toolchain is functional end-to-end, so the
"gated on a verified build" caveat in TEARDOWN.md is now cleared. Note: this is the
*baseline* (still TurboQuant+); no ChannelQuant RTL/TB exists yet.

Next: implement the value path first (P1 — per-token amax + uniform INT4/INT8) and
stand up `tb_channelquant.sv` to parity-check it against the vendored CQ-8/CQ-4
value vectors, before touching the key path / deleting rotation+qjl.

---

## 2026-07-01 — toolchain reprovisioned on x86_64 host (revamp re-verified green)

Continuation of the ChannelQuant revamp on a **new host** (x86_64, `/home/shadeform`),
so the prior aarch64 `.tools` prefix and the hard-coded path in `eda-env.sh` were
both absent/dead. Reprovisioned and re-verified the full board:

- **iverilog/vvp 12.0** — this time straight from **conda-forge** (`micromamba create
  -n eda -c conda-forge iverilog verilator gperf`); the aarch64-only gap that forced
  a from-source build does not exist on linux-64. apt's iverilog is only 11.0, which
  rejects the parity TB's `localparam string` — **12.0 is the tool of record.**
- **`eda-env.sh` made host-portable** — derives `<lhs>/.tools` from `BASH_SOURCE`
  instead of hard-coding `/home/chaithu/...`; prepends the conda-forge env and/or a
  from-source `iverilog/bin` if present, so it works on both hosts.

**Validation (this host):** `. rtl/eda-env.sh` then —
`make sim` → **17/17 PASS**, `make sim_cq` → **all 9 golden vectors bit-exact**
(V+K, CQ-8/CQ-4/CQ-4+, D∈{64,128}, full+partial groups, outlier lane), and
`make sim_realdata` → PASS. The revamped ChannelQuant codec is parity-green on
x86_64, reproduced from a clean checkout.

Next: P2 streaming integration (`amax_unit`/`residual_buffer`/`scale_bank` FSM +
SRAM, outlier-mask ROM load IF), then the C++ reference leg for 3-way parity.

---

## 2026-07-01 — C++ reference leg landed → 3-way parity closed (P3)

Ported `sw/reference_model` to ChannelQuant: new `channelquant_ref.{hpp,cpp}` — a
**1:1 port of the RTL behavioral cores** (`rtl/cq_fp_pkg.sv` + `rtl/cq_units.sv`),
i.e. the same double-based fp16/fp32 helpers and quant/dequant/scale/pack math,
plus `compress_*`/`decompress_*` mirroring the numpy reference
(`ChannelQuant/reference/channelquant_ref.py`). New `test_channelquant_ref.cpp`
loads the **same vendored golden hex** that `tb_channelquant.sv` drives and checks
the same four surfaces bit-for-bit (fp16 scales, packed byte stream, fp32 K/V_hat,
CQ-4+ fp16 sidecar).

**Result:** `make -C sw/reference_model test-cq` → **all 9 vectors bit-exact**
(V+K, CQ-8/CQ-4/CQ-4+, D∈{64,128}, full+partial key groups, outlier lane), exit
code 0. With Python verified upstream (handoff) and SV via `make sim_cq`, the
**3-way Python↔C++↔SV gate (contract §8) is closed.** Because the C++ core is a
verbatim port of the SV core, C++≡SV by construction, not just coincidence on
these vectors.

Toolchain: g++ 13, `-std=c++17 -O2 -Wall -Wextra`, no warnings. Legacy
TurboQuant+ C++ tests still build/pass 64/64 (untouched); `test-cq` folded into
`make test-all`. The legacy TurboQuant+ reference model (`kv_cache_engine_ref.*`)
is retained for now — retiring it (as the RTL codec was) is a later cleanup.

Next: P2 streaming integration, or P4 synth (fp16-lowered cores → Sky130/16FFC).

---

## 2026-07-01 — CI synthesis gate fixed (was red on master 8 days) [P4a]

Found via `gh run list`: CI gate 3 (Yosys FF-count) had been **failing on master
since the top-swap commit** (be9a2ce, 8 days) — the crashed chat pushed it without
seeing CI go red. Every other gate (functional TB, reference-model C++/Python,
formal RTL≡netlist equivalence, OpenLane Sky130) was green; only the FF-count
assertion failed.

Root cause (not a bug): the reusable workflow runs `yosys synth -flatten` and
asserts total FFs == `expected-ff-count`. The revamp set the top's `SRAM_WIDTH =
VECTOR_DIM*COORD_WIDTH = 1024` (raw fp16 passthrough vector) vs the old 288-bit
TurboQuant+ compressed word, so the behavioral SRAM (SRAM_DEPTH=16 → ~16384 FFs)
+ input_buf/wr_data/FSM synthesizes to **19559 FFs** (CI's apt yosys 0.33; exact
awk sum `$_DFFE_PN0P_ 17488 + $_DFFE_PP_ 2052 + $_DFF_PN0_ 17 + $_DFFE_PN1P_ 2`).
The gate still read the stale 5575. The revamped RTL itself synthesizes cleanly
(Yosys CHECK: 0 problems).

Fix: `expected-ff-count 5575 → 19559` in `.github/workflows/ci.yml` with a comment
that this is a *transitional* count — P2's compressed streaming store shrinks
SRAM_WIDTH and this number comes back down. Note: local yosys 0.65 (conda-forge)
opt-strips the undriven behavioral SRAM to ~81 FFs, so it cannot reproduce the
0.33 gate number — CI's apt-yosys is authoritative for this assertion.

Next: P4b synthesizable fp16 core lowering (G-independent), and/or P2 streaming.

---

## 2026-07-01 — re-vendored ChannelQuant contract 08d5287 → 7f5a1e1 (P2 unblocked)

The ChannelQuant lane pinned the four P2 datapath parameters and pushed contract
**v0.2** at `7f5a1e1`. Re-vendored `rtl/tb/testvectors/channelquant/`:
`HW_CONTRACT.md` + new `masks/` (calibrated outlier ROMs) + `SOURCE_COMMIT`.

Pinned for P2:
- **G = 128 for all D** (§3.1) — Phase-2 Pareto; acc_norm flat in G, G=128 is the
  eff-bits floor that still streams cheaply. D=64 golden vectors keep G=64 only as
  extra grouping/partial-flush coverage, not a shipped default. So the residual
  buffer is sized for **G=128 tokens**.
- **Outlier ROM** (§4.1): `masks/<tag>_k2.npz` — `outlier_idx` int64 `[L,n_kv,k]` +
  `outlier_bitmask` uint8 `[L,n_kv,D]`, **k=2** both D, lane **optional at D=128**
  (build it bypassable). P2's outlier lane loads this, not the vector-embedded mask.
  Vendored: `q05_k2`(D64), `q15_k2`/`q7_k2`(D128), `mistral_k2`(D128).
- **Decompress read bus = fp32** (§1) — matches what C++/SV already emit; no change.
- **EPS = 2⁻¹⁴ final** (§1) — already what we implement.

**Golden vectors are byte-identical to 08d5287** (unchanged `.npz`/`hex/`), so the
3-way parity carries over — re-ran both legs anyway as a hermetic check: C++ and SV
both still all-9 bit-exact. No re-run of parity was required.

Not P2 blockers (per the lane): Phase-3 Mistral×ARC/HellaSwag generalization (feeds
the paper's accuracy column, still running) and the C16 "+lane-optional-at-D=128"
guidance (a config default — the bypassable outlier lane covers both).

Next: **P2 streaming datapath** — residual buffer (G=128), per-channel scale bank
(depth D), outlier-mask ROM load, and streaming the cq cores through the top FSM.

---

## 2026-07-01 — P2 [1/n]: amax_unit implemented + verified

First P2 module. `amax_unit.sv` is now a real **synthesizable** streaming reduction
(replacing the inert skeleton): value path = per-token amax over D elements; key
path = per-channel running max over a group of tokens, frozen on `group_done`
(handles partial final groups). No `real` arithmetic — exploits that for finite
fp16 the magnitude (sign cleared) is monotonic in the unsigned 15-bit {exp,man}
field, so amax is a plain unsigned-integer max.

Verified by `tb_amax_unit.sv` (`make sim_amax`): drives the DUT token-by-token
and routes its amax through the P3-proven `cq_scale_unit`; the resulting fp16
scales match the golden `val_scales`/`key_scales` **bit-exact on all 9 vectors**
(V + K, CQ-8 per-token keys, CQ-4/4+ per-channel groups, full g=G and partial g<G).
TB gotcha fixed: stimulus must change on negedge, else the TB deasserting in_valid
in the DUT's posedge active-region races and the DUT misses the token.

Full board green: sim 17/17, sim_cq 9/9, sim_realdata PASS, sim_amax 9/9. amax_unit
is verified standalone but not yet in RTL_SRC/top (FSM integration is a later P2
step). Next P2 module: `scale_bank` (per-channel + per-token scale storage), then
`residual_buffer` (G=128 group hold), then stream the cores through the top FSM.

---

## 2026-07-01 — P2 [2/n]: value path integrated + verified (integration-first)

`cq_value_path.sv` — the reusable per-token VALUE datapath the top will
instantiate: `amax_unit → cq_scale_unit → D× cq_quant_unit → pack` (compress) and
`D× cq_dequant_unit` (decompress). `bits` runtime (8/4), D a parameter. A pipeline
hazard was handled by registering the token internally (`vec_reg`) so it stays
aligned with its 1-cycle-late scale — the producer presents each token once.

`tb_value_path.sv` (`make sim_vpath`) streams whole value tensors through two DUTs
(D=64, D=128) and checks **scale + packed payload + fp32 V_hat bit-exact vs golden**
on all 9 vectors (per-token in every tier; bits 8 for CQ-8 else 4). Even D → each
token is D/2 (int4) or D (int8) whole bytes, so per-token packing concatenates to
the golden flat stream. Board: sim/sim_cq/sim_realdata/sim_amax/**sim_vpath** all
green.

This proves the streaming datapath architecture end-to-end for the simple path.
Next: the KEY path (`cq_key_path`) — per-channel grouped scaling with the
`residual_buffer` (G=128 fp16 hold) + `scale_bank` (D per-channel scales) +
outlier lane (loads the vendored `masks/` ROM), verified vs golden key_scales/
key_payload/expected_k_hat/sidecar. Then wire both paths + SRAM into the top FSM.

---

## 2026-07-01 — P2 [3/n]: KEY path integrated + verified (the crux)

`cq_key_path.sv` — the per-channel grouped key datapath, the defining ChannelQuant
mechanism. Group FSM **COLLECT → SCALE → EMIT → DONE**: buffer the group
(`residual_buffer`, fp16), take the per-channel max (`amax_unit` key mode), freeze
D per-channel fp16 scales (D× `cq_scale_unit` → `scale_bank`), then walk the
buffered tokens one/cycle quantizing the keep channels (D× `cq_quant_unit`) and
packing INT4 (outlier channels excluded via `outlier_mask`). Decompress is
combinational (D× `cq_dequant_unit`); outlier channels are the top's FP16 sidecar.
`residual_buffer` and `scale_bank` are now real modules (were skeletons).

`tb_key_path.sv` (`make sim_kpath`) streams the 6 per-channel vectors group by
group and checks **scales + INT4 payload + K_hat + sidecar bit-exact**, full g=G
and partial g<G, CQ-4/CQ-4+. One D=128 DUT covers both head dims (D=64 vectors
mask the upper 64 channels as outliers → keep == the real D=64 keep set). Two TB
bugs fixed: streaming stimulus on negedge (edge race), and kc_arr sized to
G·MD (was MD → tokens ≥1 read out-of-bounds → K_hat=0).

Board green: sim/sim_cq/sim_realdata/sim_amax/sim_vpath/**sim_kpath** all pass.
Both datapaths (value + key) now exist as verified modules. **Remaining P2:** wire
cq_value_path + cq_key_path + scale/payload SRAM into the top FSM (replacing the
passthrough store) — which also shrinks the inflated FF count — plus the
outlier-mask ROM load from the vendored `masks/`. Then real-data + P4b lowering.
