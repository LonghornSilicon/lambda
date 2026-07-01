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
