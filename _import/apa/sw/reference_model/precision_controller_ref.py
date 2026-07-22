"""Bit-accurate Python reference model of the LonghornSilicon ACU precision controller.

This module mirrors `rtl/precision_controller.sv` exactly — same arithmetic,
same bit widths, same accumulator semantics, same s_last reset behavior.
Any silicon-agnostic compiler that wants to target the precision controller
can use this as its reference: feed it identical inputs and the outputs are
guaranteed to match what the real RTL (and eventually the silicon) will
produce.

Two abstraction levels:

  Low-level streaming (mirrors the SystemVerilog interface):
      pc = PrecisionController()
      pc.reset()
      pc.tick(s_valid=True, s_data=x,    s_last=False)
      pc.tick(s_valid=True, s_data=last, s_last=True)
      decision = pc.read_decision()       # d_fp16 (True = FP16, False = INT8)

  High-level batch:
      pc = PrecisionController()
      decision = pc.process_tile(scores)  # scores: iterable of N signed ints

Constants match the default RTL parameters:
    BLOCK_M = BLOCK_N = 64    -> N = 4096
    SCORE_WIDTH = 8           -> int8 scores
    THRESHOLD = 10            -> ratio gate

The decision formula, expressed as the RTL implements it (no division):
    LHS = max(|s|) << log2(N)            # = max * N
    RHS = (sum(|s|) << 3) + (sum(|s|) << 1)   # = sum * 10
    decision = (LHS > RHS)               # True => FP16, False => INT8
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional


# Mask helpers: simulate fixed-width hardware registers.
def _mask(width: int) -> int:
    return (1 << width) - 1


def _abs_two_s_complement(value: int, width: int) -> int:
    """Replicates the SV `s_data[W-1] ? (~s_data + 1) : s_data` ABS circuit.

    The result is constrained to `width` bits — note that the abs of the most
    negative value `-2**(W-1)` is `2**(W-1)`, which equals exactly `-2**(W-1)`
    again under modular arithmetic at `width` bits. This mirrors the hardware:
    `abs(-128)` returned with 8 bits is `0x80 = 128` (which is fine since the
    accumulator promotes immediately to wider arithmetic for the sum and
    comparison).
    """
    mask = _mask(width)
    if value < 0:
        # two's-complement negation under `width` bits
        return (((~value) + 1) & mask)
    return value & mask


@dataclass
class PrecisionControllerInfo:
    """Read-only synthesis-time configuration; matches the chip's INFO registers."""
    block_m: int = 64
    block_n: int = 64
    score_width: int = 8
    threshold: int = 10

    @property
    def n(self) -> int:
        return self.block_m * self.block_n

    @property
    def log2_n(self) -> int:
        return int(math.log2(self.n))

    @property
    def sum_width(self) -> int:
        return self.score_width + self.log2_n

    @property
    def cmp_width(self) -> int:
        return self.sum_width + 4  # +4 bits to hold THRESHOLD*sum without overflow

    def __post_init__(self) -> None:
        # The SV implementation hard-codes the multiply-by-10 as
        # (sum<<3) + (sum<<1). Other thresholds would require a different
        # netlist. Reflecting that constraint here.
        if self.threshold != 10:
            raise NotImplementedError(
                "Reference model currently mirrors the synthesized THRESHOLD=10. "
                "Other thresholds need a parameterized RTL build first."
            )
        # Confirm N is a power of two (needed for the free LOG2_N shift).
        n = self.block_m * self.block_n
        if (n & (n - 1)) != 0:
            raise ValueError(f"N={n} must be a power of two for the shift-only LHS path")


class PrecisionController:
    """Bit-accurate streaming model of the precision_controller.sv DUT."""

    def __init__(self, info: Optional[PrecisionControllerInfo] = None) -> None:
        self.info = info or PrecisionControllerInfo()
        self.reset()

    # ------------------------------------------------------------------
    # Low-level streaming interface — same semantics as the SV ports.
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Equivalent to asserting `rst_n = 0` for one cycle."""
        self._max_acc = 0
        self._sum_acc = 0
        self._d_valid = False
        self._d_fp16 = False
        self._tiles_processed = 0
        self._scores_in_current_tile = 0
        # Trace of all decisions issued since reset (useful for compiler verification)
        self._decision_history: List[bool] = []

    def tick(self, s_valid: bool, s_data: int, s_last: bool) -> None:
        """Advance the model by one clock cycle.

        Args:
            s_valid: Input handshake; ignored when False (no state change).
            s_data:  Signed integer in [-2^(W-1), 2^(W-1)-1]. Any width
                     mismatch is masked to SCORE_WIDTH bits to match SV.
            s_last:  Asserted on the last score of a tile. Decision is
                     latched on this cycle; accumulators reset for the
                     next tile.
        """
        info = self.info
        score_mask = _mask(info.score_width)
        sum_mask = _mask(info.sum_width)

        # By default, d_valid pulses for one cycle only.
        self._d_valid = False

        if not s_valid:
            return

        # Mask s_data into a SCORE_WIDTH two's-complement view.
        masked = s_data & score_mask
        # Convert back to signed for the absolute-value circuit.
        if masked & (1 << (info.score_width - 1)):
            signed = masked - (1 << info.score_width)
        else:
            signed = masked
        abs_score = _abs_two_s_complement(signed, info.score_width)

        # Combinational next-state: include current score so decision is final
        # on the same cycle s_last asserts (matches SV).
        max_next = abs_score if abs_score > self._max_acc else self._max_acc
        sum_next = (self._sum_acc + abs_score) & sum_mask

        if s_last:
            # Compute the decision using fixed-width unsigned arithmetic.
            lhs = max_next << info.log2_n                          # max * N
            rhs = (sum_next << 3) + (sum_next << 1)                # sum * 10
            # Both LHS and RHS fit within CMP_W bits by construction.
            cmp_mask = _mask(info.cmp_width)
            lhs &= cmp_mask
            rhs &= cmp_mask
            self._d_fp16 = (lhs > rhs)
            self._d_valid = True
            self._decision_history.append(self._d_fp16)

            # Reset accumulators (last-write wins in SV, same here).
            self._max_acc = 0
            self._sum_acc = 0
            self._tiles_processed += 1
            self._scores_in_current_tile = 0
        else:
            self._max_acc = max_next
            self._sum_acc = sum_next
            self._scores_in_current_tile += 1

    # ------------------------------------------------------------------
    # Status / output reads — match the chip's STATUS register and decision FIFO.
    # ------------------------------------------------------------------
    @property
    def d_valid(self) -> bool:
        return self._d_valid

    @property
    def d_fp16(self) -> bool:
        return self._d_fp16

    @property
    def tiles_processed(self) -> int:
        return self._tiles_processed

    @property
    def decision_history(self) -> List[bool]:
        """All decisions emitted since reset, in order."""
        return list(self._decision_history)

    def read_decision(self) -> bool:
        """Return the most recent decision (True = FP16, False = INT8).

        Caller is expected to have already streamed a complete tile and seen
        d_valid pulse on the s_last cycle. Returns the latched value.
        """
        return self._d_fp16

    # ------------------------------------------------------------------
    # High-level helpers — what a compiler runtime typically calls.
    # ------------------------------------------------------------------
    def process_tile(self, scores: Iterable[int]) -> bool:
        """Stream a full tile of scores and return its decision.

        `scores` must contain exactly N = BLOCK_M * BLOCK_N values. The
        accumulators are reset on tile completion (s_last); no manual reset
        needed between consecutive tiles.
        """
        scores = list(scores)
        if len(scores) != self.info.n:
            raise ValueError(
                f"process_tile expects exactly {self.info.n} scores, got {len(scores)}"
            )
        for i, s in enumerate(scores):
            self.tick(s_valid=True, s_data=int(s), s_last=(i == len(scores) - 1))
        return self._d_fp16

    def process_tiles(self, tiles: Iterable[Iterable[int]]) -> List[bool]:
        """Convenience: batch-process multiple tiles. Returns a list of decisions."""
        return [self.process_tile(tile) for tile in tiles]

    # ------------------------------------------------------------------
    # Pure functional ABI — useful as a one-shot reference without state.
    # ------------------------------------------------------------------
    @staticmethod
    def decide(scores: Iterable[int],
               info: Optional[PrecisionControllerInfo] = None) -> bool:
        """Stateless evaluation: compute the decision for a single tile.

        Equivalent to `PrecisionController(info).process_tile(scores)` but
        does not allocate / track history. Useful for compilers that want
        a pure function with no hidden state.
        """
        info = info or PrecisionControllerInfo()
        scores = list(scores)
        if len(scores) != info.n:
            raise ValueError(f"decide expects exactly {info.n} scores, got {len(scores)}")
        score_mask = _mask(info.score_width)

        max_v = 0
        sum_v = 0
        for s in scores:
            masked = int(s) & score_mask
            if masked & (1 << (info.score_width - 1)):
                signed = masked - (1 << info.score_width)
            else:
                signed = masked
            a = _abs_two_s_complement(signed, info.score_width)
            if a > max_v:
                max_v = a
            sum_v = (sum_v + a) & _mask(info.sum_width)

        cmp_mask = _mask(info.cmp_width)
        lhs = (max_v << info.log2_n) & cmp_mask
        rhs = ((sum_v << 3) + (sum_v << 1)) & cmp_mask
        return lhs > rhs


__all__ = ["PrecisionController", "PrecisionControllerInfo"]
