"""Bit-accurate Python reference model of the LonghornSilicon Token Importance Unit.

This module mirrors `rtl/token_importance_unit.sv` exactly -- same slot state,
same saturating accumulators, same serialized-argmin eviction semantics, same
combinational tier comparator. Any silicon-agnostic compiler that wants to
target the TIU can use this as its reference: drive it with the same op stream
the RTL consumes and the outputs are guaranteed to match what the real RTL
(and eventually the silicon) will produce.

The RTL core implements exactly three operations plus one combinational read:

  LOAD  (slot)          : score[slot] := 0, valid[slot] := 1     -- install a token
  ACC   (slot, weight)  : score[slot] := sat_add(score[slot], w) -- add attention mass
  EVICT ()      -> slot : serialized argmin over valid slots     -- pick + free victim
  TIER  (thr)   -> keep : tier_keep[k] = valid[k] & score[k] >= thr (per-slot compare)

The eviction argmin mirrors the RTL bit-for-bit: the running minimum is *seeded*
with slot 0 (its score if valid, else the SCORE_MAX sentinel), then a
strict-less scan runs over slots 0..N_SLOTS-1. Ties therefore resolve to the
lowest slot index, and slot 0 wins a tie against itself. LOAD has priority over
ACC when both target the same slot in the same cycle (matched here by applying
LOAD before ACC if a caller issues both).

H2O bookkeeping (the recent-window protection and the KV-budget accounting) is
NOT in this core: it is a thin control-layer wrapper that decides *when* to issue
LOAD / EVICT. The silicon core is accumulate + argmin + tier, and that is exactly
what this model reproduces. See `example_compiler_use.py` for the wrapper.

Two abstraction levels:

  High-level op API (what a compiler driver calls):
      tiu = TokenImportanceUnit()
      tiu.load(slot=0)
      tiu.acc(slot=0, weight=200)
      victim = tiu.evict()             # int slot index, freed on return
      keep   = tiu.tier(threshold=40)  # list[bool], one per slot

  Stateless argmin (a pure function, no hidden state):
      victim = TokenImportanceUnit.argmin_victim(score, valid, n_slots)

Constants match the default RTL parameters:
    N_SLOTS      = 8
    SCORE_WIDTH  = 8   -> score saturates at 255
    WEIGHT_WIDTH = 8   -> attention-mass increment in [0, 255]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class TokenImportanceUnitInfo:
    """Read-only synthesis-time configuration; matches the chip's INFO registers."""
    n_slots: int = 8
    score_width: int = 8
    weight_width: int = 8
    # ISA version, bits[15:8] = major, bits[7:0] = minor  ->  0x0001 = "0.1".
    version: int = 0x0001

    @property
    def score_max(self) -> int:
        """Saturation value of a score accumulator (all-ones, SCORE_WIDTH bits)."""
        return (1 << self.score_width) - 1

    @property
    def weight_max(self) -> int:
        return (1 << self.weight_width) - 1

    @property
    def slot_width(self) -> int:
        """Bits to index a slot -- SLOT_WIDTH = clog2(N_SLOTS), min 1 (RTL localparam)."""
        if self.n_slots <= 1:
            return 1
        return (self.n_slots - 1).bit_length()

    def __post_init__(self) -> None:
        if self.n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {self.n_slots}")
        if self.score_width < 1 or self.weight_width < 1:
            raise ValueError("score_width and weight_width must be >= 1")


class TokenImportanceUnit:
    """Bit-accurate model of the token_importance_unit.sv DUT.

    Holds N_SLOTS (score, valid) pairs and reproduces the LOAD / ACC / EVICT /
    TIER semantics exactly. State is reset on construction and on `reset()`.
    """

    def __init__(self, info: Optional[TokenImportanceUnitInfo] = None) -> None:
        self.info = info or TokenImportanceUnitInfo()
        self.reset()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Equivalent to asserting `rst_n = 0`: control FFs cleared, slots empty.

        The RTL only resets `state`, `valid[]`, and `evict_valid`; the `score[]`
        datapath is intentionally NOT reset (a slot's score is zeroed by LOAD
        before it is ever read). We zero score[] here too for a clean, printable
        model -- it is unobservable because a slot is never read while invalid.
        """
        n = self.info.n_slots
        self._score: List[int] = [0] * n
        self._valid: List[bool] = [False] * n
        self._evict_history: List[int] = []

    # ------------------------------------------------------------------
    # Core arithmetic -- mirrors the SV `sat_add` function.
    # ------------------------------------------------------------------
    def _sat_add(self, s: int, w: int) -> int:
        """score + weight, saturating at SCORE_MAX (mirrors sat_add in the RTL)."""
        total = s + w
        smax = self.info.score_max
        return total if total <= smax else smax

    # ------------------------------------------------------------------
    # Operations -- one per RTL op.
    # ------------------------------------------------------------------
    def load(self, slot: int) -> None:
        """LOAD: install a fresh token in `slot` -- zero its score, mark valid."""
        self._check_slot(slot)
        self._score[slot] = 0
        self._valid[slot] = True

    def acc(self, slot: int, weight: int) -> None:
        """ACC: add `weight` of attention mass to `slot` (saturating).

        `weight` is masked to WEIGHT_WIDTH bits to match the hardware port, then
        saturating-added into the SCORE_WIDTH accumulator. Accumulating into an
        invalid slot is allowed (the RTL does it too); its score simply is not
        read until a LOAD makes it valid.
        """
        self._check_slot(slot)
        w = weight & self.info.weight_max
        self._score[slot] = self._sat_add(self._score[slot], w)

    def evict(self) -> int:
        """EVICT: return the minimum-mass valid slot and free it.

        Mirrors the RTL serialized argmin exactly: seed the running min with
        slot 0 (its score if valid, else the SCORE_MAX sentinel), then a
        strict-less scan over slots 0..N-1. The chosen victim's valid bit is
        cleared before returning (the RTL frees it in S_DONE). Ties resolve to
        the lowest slot index.
        """
        victim = self.argmin_victim(self._score, self._valid, self.info.n_slots,
                                    self.info.score_max)
        self._valid[victim] = False
        self._evict_history.append(victim)
        return victim

    def tier(self, threshold: int) -> List[bool]:
        """TIER read: per-slot keep flags, tier_keep[k] = valid[k] & score[k] >= thr.

        Combinational in the RTL (N parallel comparators); a pure function of the
        current state here. True = heavy hitter (keep value at CQ-8), False =
        demote (store value at CQ-4) or empty slot.
        """
        thr = threshold & self.info.score_max
        return [self._valid[k] and (self._score[k] >= thr)
                for k in range(self.info.n_slots)]

    # ------------------------------------------------------------------
    # Observability -- match the chip's readable state.
    # ------------------------------------------------------------------
    @property
    def score(self) -> List[int]:
        """Current per-slot accumulated mass (read-only snapshot)."""
        return list(self._score)

    @property
    def valid(self) -> List[bool]:
        """Current per-slot valid bits (read-only snapshot)."""
        return list(self._valid)

    @property
    def busy(self) -> bool:
        """The functional model completes each op atomically, so it is never busy.

        The RTL is busy for N_SLOTS+1 cycles during an EVICT scan; that latency
        is documented in the ISA and does not change the functional result.
        """
        return False

    @property
    def evict_history(self) -> List[int]:
        """All victims returned since reset, in order."""
        return list(self._evict_history)

    # ------------------------------------------------------------------
    # Stateless argmin -- the pure eviction rule, no hidden state.
    # ------------------------------------------------------------------
    @staticmethod
    def argmin_victim(score: Sequence[int], valid: Sequence[bool],
                      n_slots: int, score_max: Optional[int] = None) -> int:
        """Serialized-argmin victim, seeded at slot 0 with a strict-less scan.

        Bit-exact with the RTL S_SCAN loop and with the shadow model in the
        SystemVerilog testbenches. If no slot is valid, returns 0 (the seed),
        matching the RTL sentinel behavior.
        """
        if score_max is None:
            score_max = max(score) if score else 0
        m_score = score[0] if valid[0] else score_max
        m_idx = 0
        for k in range(n_slots):
            if valid[k] and score[k] < m_score:
                m_score = score[k]
                m_idx = k
        return m_idx

    # ------------------------------------------------------------------
    def _check_slot(self, slot: int) -> None:
        if not (0 <= slot < self.info.n_slots):
            raise IndexError(
                f"slot {slot} out of range [0, {self.info.n_slots})")


__all__ = ["TokenImportanceUnit", "TokenImportanceUnitInfo"]


# ---------------------------------------------------------------------------
# Self-test: exercise every operation against hand-computed expectations,
# reproducing the directed cases in tb_token_importance_unit.sv.
# ---------------------------------------------------------------------------
def _self_test() -> None:
    info = TokenImportanceUnitInfo()
    tiu = TokenImportanceUnit(info)

    # Directed: load 4 slots, give them distinct masses, evict the minimum.
    for s in range(4):
        tiu.load(s)
    tiu.acc(0, 100)
    tiu.acc(1, 30)
    tiu.acc(2, 200)
    tiu.acc(3, 30)
    # min mass = 30 at slots 1 and 3 -> lowest index wins -> slot 1.
    v = tiu.evict()
    assert v == 1, f"expected victim 1, got {v}"
    assert tiu.valid[1] is False, "slot 1 must be freed after eviction"

    # Accumulate more, evict again among {0:110, 2:250, 3:30} -> slot 3.
    tiu.acc(2, 50)
    tiu.acc(0, 10)
    v = tiu.evict()
    assert v == 3, f"expected victim 3, got {v}"

    # Saturation: hammer slot 0 past SCORE_MAX.
    for _ in range(600):
        tiu.acc(0, 0xFF)
    assert tiu.score[0] == info.score_max, \
        f"slot 0 must saturate at {info.score_max}, got {tiu.score[0]}"

    # Reload freed slots (both reset to 0), then give them distinct masses so
    # the new minimum is unambiguous: slot 1 = 5, slot 3 = 20 -> slot 1 wins.
    tiu.load(1)
    tiu.load(3)
    tiu.acc(1, 5)
    tiu.acc(3, 20)
    v = tiu.evict()
    assert v == 1, f"expected victim 1 (mass 5), got {v}"

    # Tier handshake: threshold 40, keep = valid & score >= 40.
    keep = tiu.tier(40)
    for k in range(info.n_slots):
        exp = tiu.valid[k] and (tiu.score[k] >= 40)
        assert keep[k] == exp, f"tier[{k}] = {keep[k]}, expected {exp}"

    # Empty-cache eviction returns the seed slot 0.
    empty = TokenImportanceUnit(info)
    assert empty.argmin_victim(empty.score, empty.valid, info.n_slots,
                               info.score_max) == 0

    print("tiu_ref self-test: ALL CHECKS PASSED")


if __name__ == "__main__":
    _self_test()
