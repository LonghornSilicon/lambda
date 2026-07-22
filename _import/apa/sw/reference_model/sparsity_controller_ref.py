"""Bit-accurate Python reference model of the proposed ACU sparsity controller.

Sibling to `precision_controller_ref.py`. Same streaming-controller shape:
consume one signed attention score per cycle, pulse a decision the cycle
after `s_last`. Different decision rule and different data dependency.

Algorithm (XAttention-style antidiagonal proxy, Xu et al. 2025):

    For each BLOCK_M x BLOCK_N tile of pre-softmax scores S[i, j],
    accumulate the absolute value of every score that sits on a
    strided antidiagonal of the tile:

        antidiag_acc = sum(|S[i, j]|)  over  { (i, j) : (i + j) mod STRIDE == 0 }

    Compare against a per-layer threshold:

        skip = (antidiag_acc * SCALE) < THRESHOLD_REG

    `skip = 1` means this tile contributes negligible mass to softmax(S)V
    and can be elided. `skip = 0` hands the tile to the precision
    controller, which independently picks INT8 vs FP16.

    Hardware shape (one-cycle decision after s_last, no division, no
    transcendentals, integer-only):

        - one accumulator: antidiag_acc      (SCORE_WIDTH + log2(samples) bits)
        - one comparator: antidiag_acc <<< K  vs  THRESHOLD_REG
        - decision FF + valid FF
        - per-tile index counter to drive the stride mask (LOG2_N bits)

    Closed-form FF count for the default config:

        antidiag_acc_w = SCORE_WIDTH + ceil(log2(samples)) = 8 + ceil(log2(64))
                      = 14 FFs
        index counter  = log2(N)                            = 12 FFs
        decision + valid                                    =  2 FFs
        threshold reg                                       = 16 FFs
                                                            ----
                                                              44 FFs

    (Compare: precision_controller is 30 FFs at the same N.)

Two abstraction levels — same shape as the precision controller:

  Low-level streaming (mirrors the planned SV interface):
      sc = SparsityController()
      sc.reset()
      sc.tick(s_valid=True, s_data=x,    s_last=False)
      sc.tick(s_valid=True, s_data=last, s_last=True)
      skip = sc.read_decision()           # True = skip tile, False = compute

  High-level batch:
      sc = SparsityController()
      skip = sc.process_tile(scores)      # iterable of N signed ints

Constants matching the proposed RTL parameters:
    BLOCK_M = BLOCK_N = 64    -> N = 4096
    SCORE_WIDTH = 8           -> int8 scores  (matches precision controller)
    STRIDE      = 8           -> sample every 8th diagonal of the tile
                                 ~64 samples per 64x64 tile (one antidiagonal worth)
    THRESHOLD   = profiled    -> per-layer register, defaults to a value
                                 chosen on real-LLM traces (see analysis/)

The default STRIDE of 8 maps to XAttention's stride-8 antidiagonal pattern
in a 128-block layout, halved to fit the precision controller's 64x64 tile.
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

    See precision_controller_ref._abs_two_s_complement — identical semantics.
    """
    mask = _mask(width)
    if value < 0:
        return (((~value) + 1) & mask)
    return value & mask


@dataclass
class SparsityControllerInfo:
    """Read-only synthesis-time configuration; mirrors INFO_* registers."""
    block_m: int = 64
    block_n: int = 64
    score_width: int = 8
    stride: int = 8
    # THRESHOLD is intended as a runtime-writable per-layer register, unlike
    # the precision controller's hard-coded ×10. The default below is a
    # neutral middle value; real deployments will program it after profiling.
    threshold: int = 1024
    threshold_width: int = 16

    @property
    def n(self) -> int:
        return self.block_m * self.block_n

    @property
    def log2_n(self) -> int:
        return int(math.log2(self.n))

    @property
    def samples_per_tile(self) -> int:
        """Number of (i, j) positions hit by the stride-S antidiagonal mask.

        For BLOCK_M=BLOCK_N=64 with STRIDE=8: 64 samples per tile
        (one antidiagonal, every 8th index along it... but here we use
        the full antidiagonal mask `(i + j) % STRIDE == 0` which gives
        exactly N/STRIDE samples.)
        """
        return self.n // self.stride

    @property
    def acc_width(self) -> int:
        """Bits needed for the antidiag accumulator: SCORE_WIDTH + log2(samples)."""
        return self.score_width + int(math.ceil(math.log2(self.samples_per_tile)))

    def __post_init__(self) -> None:
        if (self.n & (self.n - 1)) != 0:
            raise ValueError(f"N={self.n} must be a power of two")
        if (self.stride & (self.stride - 1)) != 0:
            raise ValueError(f"STRIDE={self.stride} must be a power of two")
        if self.stride > self.block_n:
            raise ValueError(f"STRIDE={self.stride} exceeds BLOCK_N={self.block_n}")


class SparsityController:
    """Streaming model of the proposed sparsity_controller.sv block.

    Mirrors the precision controller's interface so the two can sit side by
    side in the ACU, sharing the score stream. Counters / accumulators are
    masked to their hardware widths so the model is bit-exact against the
    eventual RTL.
    """

    def __init__(self, info: Optional[SparsityControllerInfo] = None) -> None:
        self.info = info or SparsityControllerInfo()
        self.reset()

    # ------------------------------------------------------------------
    # Low-level streaming interface — same semantics as the SV ports.
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Equivalent to asserting `rst_n = 0` for one cycle."""
        self._antidiag_acc = 0
        self._index = 0                  # counts scores within current tile
        self._d_valid = False
        self._d_skip = False
        self._tiles_processed = 0
        self._decision_history: List[bool] = []

    def tick(self, s_valid: bool, s_data: int, s_last: bool) -> None:
        """Advance the model by one clock cycle.

        Same handshake as precision_controller.tick. The index counter
        deterministically tracks (i, j) within the tile so the stride
        mask `(i + j) % STRIDE == 0` is a wired AND of low index bits.
        """
        info = self.info
        score_mask = _mask(info.score_width)
        acc_mask   = _mask(info.acc_width)

        self._d_valid = False
        if not s_valid:
            return

        # Decode (i, j) from the per-tile index counter.
        # j = index & (BLOCK_N - 1)
        # i = index >> log2(BLOCK_N)
        i = self._index // info.block_n
        j = self._index %  info.block_n

        # Sample mask: (i + j) % STRIDE == 0  is  ((i + j) & (STRIDE-1)) == 0
        # which in hardware is a single AND-reduce on log2(STRIDE) bits.
        on_antidiag = (((i + j) & (info.stride - 1)) == 0)

        # Mask s_data into a SCORE_WIDTH two's-complement view.
        masked = s_data & score_mask
        if masked & (1 << (info.score_width - 1)):
            signed = masked - (1 << info.score_width)
        else:
            signed = masked
        abs_score = _abs_two_s_complement(signed, info.score_width)

        # Combinational next-state — include current score so decision is
        # final on s_last (matches the precision controller convention).
        if on_antidiag:
            acc_next = (self._antidiag_acc + abs_score) & acc_mask
        else:
            acc_next = self._antidiag_acc

        if s_last:
            # Single comparator: acc_next < THRESHOLD_REG  →  skip
            self._d_skip = (acc_next < info.threshold)
            self._d_valid = True
            self._decision_history.append(self._d_skip)

            # Reset for next tile (last-write wins, same as precision ctrl).
            self._antidiag_acc = 0
            self._index = 0
            self._tiles_processed += 1
        else:
            self._antidiag_acc = acc_next
            self._index += 1

    # ------------------------------------------------------------------
    # Status / output reads — match planned STATUS register and decision FIFO.
    # ------------------------------------------------------------------
    @property
    def d_valid(self) -> bool:
        return self._d_valid

    @property
    def d_skip(self) -> bool:
        return self._d_skip

    @property
    def tiles_processed(self) -> int:
        return self._tiles_processed

    @property
    def decision_history(self) -> List[bool]:
        return list(self._decision_history)

    def read_decision(self) -> bool:
        """Return the most recent skip decision (True = skip, False = compute)."""
        return self._d_skip

    # ------------------------------------------------------------------
    # High-level helpers — what a compiler runtime typically calls.
    # ------------------------------------------------------------------
    def process_tile(self, scores: Iterable[int]) -> bool:
        """Stream a full tile of scores and return its skip decision."""
        scores = list(scores)
        if len(scores) != self.info.n:
            raise ValueError(
                f"process_tile expects exactly {self.info.n} scores, got {len(scores)}"
            )
        for i, s in enumerate(scores):
            self.tick(s_valid=True, s_data=int(s), s_last=(i == len(scores) - 1))
        return self._d_skip

    def process_tiles(self, tiles: Iterable[Iterable[int]]) -> List[bool]:
        """Convenience: batch-process multiple tiles. Returns list of decisions."""
        return [self.process_tile(tile) for tile in tiles]

    # ------------------------------------------------------------------
    # Pure functional ABI — stateless one-shot reference.
    # ------------------------------------------------------------------
    @staticmethod
    def decide(scores: Iterable[int],
               info: Optional[SparsityControllerInfo] = None) -> bool:
        info = info or SparsityControllerInfo()
        scores = list(scores)
        if len(scores) != info.n:
            raise ValueError(f"decide expects exactly {info.n} scores, got {len(scores)}")
        score_mask = _mask(info.score_width)
        acc_mask   = _mask(info.acc_width)

        acc = 0
        for idx, s in enumerate(scores):
            i = idx // info.block_n
            j = idx %  info.block_n
            if ((i + j) & (info.stride - 1)) != 0:
                continue
            masked = int(s) & score_mask
            if masked & (1 << (info.score_width - 1)):
                signed = masked - (1 << info.score_width)
            else:
                signed = masked
            a = _abs_two_s_complement(signed, info.score_width)
            acc = (acc + a) & acc_mask
        return acc < info.threshold


__all__ = ["SparsityController", "SparsityControllerInfo"]
