# Token Importance Unit — Reference Model

Bit-accurate Python reference for the LonghornSilicon **Token Importance Unit
(TIU, block 3)**. It mirrors `rtl/token_importance_unit.sv` exactly — same slot
state, same saturating accumulators, same serialized-argmin eviction, same
combinational tier comparator — so a compiler backend can target the TIU before
silicon or an FPGA prototype exists.

| Block | Python class | Op API | Parity |
|---|---|---|---|
| **Token Importance Unit** | `TokenImportanceUnit` | `load` / `acc` / `evict` / `tier` | 40/40 eviction victims vs RTL golden trace |

## Files

| File | Purpose |
|---|---|
| `tiu_ref.py` | The bit-accurate model + `__main__` self-test (directed cases). |
| `test_tiu_ref.py` | Replays `rtl/tb/testvectors/tiu_trace.hex` and checks Python↔RTL parity. |
| `example_compiler_use.py` | Compiler-facing walkthrough: budget sizing → eviction schedule → keep/demote tiers. |

## Run

```sh
python tiu_ref.py                  # self-test: directed LOAD/ACC/EVICT/TIER cases
python test_tiu_ref.py             # parity: 40/40 eviction victims match the RTL
python example_compiler_use.py     # compiler co-design walkthrough
# or, under pytest:
python -m pytest test_tiu_ref.py
```

Requires Python 3.10+ (no third-party dependencies).

## API

```python
from tiu_ref import TokenImportanceUnit, TokenImportanceUnitInfo

tiu = TokenImportanceUnit()          # N_SLOTS=8, SCORE_WIDTH=8, WEIGHT_WIDTH=8
tiu.load(slot=0)                     # install a fresh token (score:=0, valid:=1)
tiu.acc(slot=0, weight=200)          # add attention mass (saturating at 255)
victim = tiu.evict()                 # -> int slot: serialized argmin, freed on return
keep   = tiu.tier(threshold=40)      # -> list[bool]: keep[k] = valid[k] & score[k] >= thr

# Stateless eviction rule (a pure function, no hidden state):
v = TokenImportanceUnit.argmin_victim(tiu.score, tiu.valid, n_slots=8, score_max=255)
```

## Numerical semantics (frozen)

Pure integer arithmetic, matching the RTL bit-for-bit:

- **Scores**: `SCORE_WIDTH`-bit unsigned accumulators, saturating at
  `2^SCORE_WIDTH − 1 = 255`.
- **ACC**: `score[slot] := min(score[slot] + (weight & WEIGHT_MASK), SCORE_MAX)`
  — a distributed per-slot saturating adder (the RTL owns one adder per slot,
  no shared broadcast net; see the Sky130 sign-off notes).
- **EVICT**: serialized argmin seeded at slot 0
  (`min := score[0]` if `valid[0]` else `SCORE_MAX`), then a strict-less scan
  over slots `0..N−1`. Ties resolve to the lowest slot index; the victim's valid
  bit is cleared on return.
- **TIER**: `tier_keep[k] = valid[k] and (score[k] >= threshold)` — N parallel
  comparators, combinational.

The **H2O recent-window / KV-budget bookkeeping is not in this core** — it is a
control-layer wrapper (shown in `example_compiler_use.py`) that decides *when* to
issue LOAD and EVICT. The silicon core is accumulate + argmin + tier.

If the model ever disagrees with the RTL on any op stream, the model is wrong —
open an issue with the failing trace.

## Co-design context

This directory is the compiler-team deliverable: the model exposes the same
functional contract the hardware implements, so compiler backends can target the
TIU today. The tier signal it emits is consumed by the KV Cache Engine (block 2):
**`evict → drop`** plus a keep/demote lever. The historical value-precision reading
(`keep → CQ-8`, `demote → CQ-4`) is **retired** — KVE now stores a flat CQ-3-rot (INT3)
value tier, so there is no per-token value bit-width to select; only evict/keep remains. See
[`docs/tier_handshake.md`](../../docs/tier_handshake.md) and the interface spec
[`docs/isa/token_importance_unit_isa.pdf`](../../docs/isa/token_importance_unit_isa.pdf).
