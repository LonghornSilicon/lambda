# ROADMAP.md — Lambda tape-out (canonical timeline of record)

**Status:** of-record · **Created:** 2026-07-23 · **Owner:** architecture lead
**This is the single canonical timeline.** It supersedes the dates in `STATUS.md` ("Summer 2027")
and `lambda_tapeout_roadmap.md` ("December 2026"), which predate the two-track decision below and
will carry a pointer here. Where any doc disagrees on schedule, **this file wins**.

---

## The shape: two parallel tracks

Lambda runs **two tracks in parallel**, not one sequential path. The open-PDK proxy work
(GF180/Sky130) and the hallmark N16 product design proceed *at the same time* — the proxy track
de-risks tools, RTL, and PD, and feeds the product track; it does not gate it.

```
 PROXY TRACK (open PDK, public)         PRODUCT TRACK (TSMC N16FFC, private overlay)
 ───────────────────────────────       ────────────────────────────────────────────
 R1  block sign-offs (sky130) +
     Chipathon KV-coproc GF180 GDS
        │                                    (kicks off ~R2)
 R1.5 full-chip GF180 assembly               │
        │                                    │
 R2  full-chip GF180 baseline  ───feeds───►  N16 design kickoff
                                             │
                                        R2.5 N16 RTL assembly (port shared RTL → N16 flow)
                                             │
                                        R3   N16 RTL design freeze
                                             │
                                        R3.5 N16 PD assembly (floorplan freeze, signoff iter)
                                             │
                                        R4   ★ N16 TAPEOUT
```

**One RTL, multiple PDKs.** Both tracks point at the *same* block RTL (`<block>/rtl/`). Open-PDK
hardening lives in `<block>/pdk/{sky130,gf180}/`; the N16 hardening is a **separate private overlay**
(TSMC PDK is NDA — cannot live in this public repo) pointing at the same RTL. See `AGENTS.md`.

---

## Milestones (mapped to revisions — see `docs/REVISIONS.md`)

| Rev | Milestone | Track | Target date | Gate to advance |
|---|---|---|---|---|
| **R1** | Proxy-PDK **block baseline**: compute+memory blocks sky130-signed; GF180 Chipathon KV-coproc GDS closed | proxy | **2026-07** (now) | — (seed) |
| **R1.5** | **Full-chip GF180 assembly**: blocks integrate into full-chip build; remaining per-block GF180 hardening closes; chip metrics JSON committed | proxy | 2026 H2 | full-chip GF180 PnR closes; per-block GF180 artifacts committed |
| **R2** | **Full-chip GF180 baseline** + **N16 design kickoff** | proxy → product | ~2026 Q4 | GF180 full-chip signed off; N16 PDK access + private overlay repo stood up |
| **R2.5** | **N16 RTL assembly**: port shared RTL onto the N16 commercial flow (Genus/Innovus/Tempus) in the private overlay | product | 2027 H1 | RTL builds through synth on N16 |
| **R3** | **N16 RTL design freeze** | product | 2027 Spring | all blocks frozen; ISA versions locked; golden-model parity green |
| **R3.5** | **N16 PD assembly**: floorplan freeze, place-route-signoff iteration, LPDDR5X/PCIe PHY integration | product | 2027 H2 | timing/DRC/LVS converging at N16 |
| **R4** | ★ **N16 TAPEOUT** | product | **2027 Fall** | full signoff clean; shuttle slot confirmed |

**Buffer:** the Fall-2027 N16 tapeout carries **1+ semester of leeway** by design — the schedule is
built with slack, not against a hard external deadline.

---

## Load-bearing external dependencies (from `arch.yml`)

These gate the product track and are tracked as `arch.yml` `load_bearing_uncertainties`:
- **LPDDR5X x16 PHY area** (est. 1.2 mm² ±0.3) — Synopsys/Cadence quote. Blocks floorplan freeze (R3.5).
- **IMEC / Europractice mini@sic 2.0 pricing** — direct quote via `eptsmc@imec.be`. Blocks the funding plan.
  *(Note: an uncommitted `architecture/` audit pass re-estimated shuttle cost to ~$112–120K against
  published Europractice 2026 pricing — reconcile into `arch.yml` before quoting R2 externally.)*
- **N16 PDK access** (TSMC University FinFET / IMEC) — gates the entire product track from R2 on.

---

## Change log
- **2026-07-23** — Initial canonical roadmap. Reconciles the prior "December 2026" (roadmap doc) and
  "Summer 2027" (STATUS) tapeout dates into one **two-track** timeline with **N16 tapeout Fall 2027**
  (+1 semester buffer). Mapped to the R1–R4 revision model (`docs/REVISION_SYNC_SOP.md`).
