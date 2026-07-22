# LonghornSilicon × Silicon-Agnostic Compiler — Integration Brief

*One-pager for the compiler integration meeting and follow-up.*
*Branch:* `compiler-integration-prep`
*Contact:* Chaithu Talasila, UT Austin — talasilachaithu105@gmail.com

---

## What LonghornSilicon is building

A four-block LLM inference accelerator targeting TSMC 16nm FinFET (N16FFC) via the
TSMC University Program (tape-out target Summer 2027):

1. **ACU (Attention Compute Unit)** — INT8/FP16 mixed-precision attention
   with a streaming "precision controller" that picks the precision per
   tile from a single-cycle ratio check. This is the **first block
   complete**: RTL frozen, 253/253 tests pass, Sky130 end-to-end PnR
   signed off (DRC/LVS/antenna/IR-drop clean), ASAP7 area projection,
   public paper. Repo: <https://github.com/LonghornSilicon/attention-compute-unit>.
2. **KV Cache Engine** — on-die SRAM (spilling to off-chip LPDDR5X) with
   ChannelQuant compression on writes / decompression on reads
   (per-channel INT4 keys / per-token INT4 values + FP16 outlier lane).
   Goal: ~3.8× more KV context in the same LPDDR5 bandwidth. *RTL complete
   through Sky130 sign-off (see the kv-cache-engine repo).*
3. **Token Importance Unit** — per-token attention-weight accumulator
   driving keep/demote/evict decisions for mixed-precision KV
   retention. *Not yet implemented.*
4. **Memory Hierarchy Controller** — routes 0.8 MB on-die SRAM ↔ off-chip
   LPDDR5X, direct (no separate eDRAM tier). *Not yet implemented.*

For this meeting, the precision controller (block #1) is what we can
commit to a stable interface for.

---

## What we offer the compiler today

Three concrete artifacts the compiler team can target *right now*,
before silicon or even FPGA exist:

| Artifact | Purpose | Status |
|---|---|---|
| **Bit-accurate Python reference model** — [`sw/reference_model/precision_controller_ref.py`](../sw/reference_model/precision_controller_ref.py) | A pure-Python class that produces exactly the INT8/FP16 decision the chip will produce | Verified 143/143 bit-exact vs. RTL TB |
| **ISA / interface spec** — [`docs/isa/precision_controller_isa.md`](isa/precision_controller_isa.md) | Versioned (`pc-isa-0.1`) memory map + AXI protocols + C API stub | Stable for the precision controller; will unify with other blocks' ISAs as they land |
| **Compiler-perspective demo** — [`sw/reference_model/example_compiler_use.py`](../sw/reference_model/example_compiler_use.py) | Three usage levels (naive / batched / calibrated) showing what compiler-emitted code looks like | Runs and self-checks |

A compiler engineer can write a backend against the Python model today
and trust that the chip will agree byte-for-byte when silicon arrives.

---

## Four-phase integration plan

The interface is stable across all four phases. Same memory map, same
streaming protocol, same model semantics throughout.

| Phase | Timeline | Target | Who delivers what |
|---|---|---|---|
| **Phase 0 — Python reference** | now | Python model | Us: model + ISA + tests already in this branch. Compiler team: backend → calls model |
| **Phase 1 — FPGA prototype** | when ZCU102/104 arrives (months) | AXI-Lite + AXI-Stream on the dev board | Us: Vivado bitstream + PYNQ driver. Compiler team: backend → driver |
| **Phase 2 — Multi-block FPGA** | after KV Cache Engine + Token Importance Unit + Memory Hierarchy Controller blocks complete | Same AXI, more blocks visible | Us: larger Vivado project. Compiler team: extend backend |
| **Phase 3 — Silicon** | 2027+ (post-tape-out) | PCIe-attached custom chip | Us: chip + Linux driver. Compiler team: re-target the same backend |

**Critically**: Phase 0 is enough to start. The compiler team does not
need to wait for anything from us beyond what's already on this branch.

---

## What we ask of the compiler team

1. **A short architecture writeup** — what's the IR (MLIR / Relay /
   custom), what's the typical hardware-target shape they integrate
   against, what's their experience adding new backends.
2. **A scoping conversation** — is the precision controller the right
   first target, or do they want to wait for more blocks?
3. **Honest timeline expectations** — phase 0 work could start
   immediately; phase 1 (FPGA) needs the ZCU102/104 to actually be in
   hand, which is a separate gating item.
4. **Their thoughts on the open ISA questions** (last section of the
   ISA spec): runtime-vs-synthesis-time `THRESHOLD`, exposing
   `max`/`sum` alongside the decision bit, AXI vs other fabric.

---

## Open questions for the meeting

These are intentionally on the table — we have not committed to them:

1. Should `THRESHOLD` be a runtime register (cost: one register +
   parameterized multiplier; benefit: per-workload tuning without
   re-synthesis)?
2. Should the chip emit `max(|s|)` and `sum(|s|)` alongside the
   one-bit decision (lets the compiler do its own gating policy
   downstream)?
3. Is AXI-Lite + AXI-Stream the right fabric, or should we plan for
   CHI / TileLink / a custom protocol that matches the other three
   blocks?
4. Phase 1 timing: do they need real FPGA hardware to begin meaningful
   work, or is the Phase 0 Python target enough to start a backend
   that the FPGA later validates?

---

## What we are NOT committing to today

To keep boundaries clear up front:

- **Schedule promises tied to tape-out** — the TSMC University Program
  shuttle date is outside our control.
- **An ISA for the full chip** — three of the four blocks are not yet
  RTL. The precision-controller ISA is stable; the chip-level ISA
  isn't.
- **Source access to private repos** — until we know the
  collaboration shape, the public repo is the boundary.
- **Exclusivity** — other compiler projects may also be interested in
  this chip. Happy to discuss exclusivity terms, but not by default.

---

## Quickest possible next step (~1 week of work)

If the meeting goes well, the smallest meaningful collaboration:

1. The compiler team writes a 2-page integration plan: which IR ops
   map to which ISA operations, what their backend's runtime expects
   from us, and what would unblock them to start coding.
2. We respond within a week with the gaps we can fill and timelines.
3. They pick the first integration target — likely a single matmul or
   a single attention block exercised against the Phase 0 reference
   model.
4. We co-author the first commit in their backend that produces
   correct decisions when run against our model.

That commit becomes the milestone we can both demo at meetings,
conferences, or to UT Austin / TSMC for further support.

---

## What's already public and shareable

Everything in this brief, the repo on the
`compiler-integration-prep` branch, the precision-controller paper at
[`paper/adaptive_precision_attention.pdf`](../paper/adaptive_precision_attention.pdf),
and the OpenLane-signed-off Sky130 GDS. Nothing here is under NDA.
The TSMC 16FFC PDK is NDA-protected and not in any of this — the
precision controller is open source through and through.
