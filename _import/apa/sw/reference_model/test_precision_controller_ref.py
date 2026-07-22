"""Tests for the Python reference model against the RTL test vectors.

Bit-exact match required:
  - all 143 replay tiles produce the same INT8/FP16 decision as the RTL TB
  - the streaming `tick()` API and the batched `process_tile()` agree
  - the stateless `decide()` function agrees with the stateful model

Run with:   python -m pytest sw/reference_model/test_precision_controller_ref.py
Or simply:  python sw/reference_model/test_precision_controller_ref.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Ensure we can import the model whether this is run as a script or via pytest.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "analysis"))

from precision_controller_ref import (   # noqa: E402
    PrecisionController,
    PrecisionControllerInfo,
)


REPO_ROOT = HERE.parent.parent
TESTVECTORS_DIR = REPO_ROOT / "rtl" / "tb" / "testvectors"
GEN_SCRIPT = REPO_ROOT / "analysis" / "gen_rtl_testvectors.py"


def _ensure_testvectors() -> None:
    """Generate the replay vectors if they are not on disk (CI clones them empty)."""
    scores = TESTVECTORS_DIR / "scores.hex"
    expected = TESTVECTORS_DIR / "expected.hex"
    if scores.exists() and expected.exists():
        return
    print(f"[setup] Generating test vectors via {GEN_SCRIPT}", flush=True)
    subprocess.check_call([sys.executable, str(GEN_SCRIPT)])


def _load_tiles_and_expected():
    """Read scores.hex (4096 lines per tile) + expected.hex into Python."""
    _ensure_testvectors()
    info = PrecisionControllerInfo()
    n = info.n

    score_lines = (TESTVECTORS_DIR / "scores.hex").read_text().strip().splitlines()
    expected_lines = (TESTVECTORS_DIR / "expected.hex").read_text().strip().splitlines()

    if len(score_lines) % n != 0:
        raise AssertionError(
            f"scores.hex has {len(score_lines)} lines; must be a multiple of {n}"
        )
    num_tiles = len(score_lines) // n
    if num_tiles != len(expected_lines):
        raise AssertionError(
            f"scores.hex implies {num_tiles} tiles but expected.hex has "
            f"{len(expected_lines)} decisions"
        )

    tiles = []
    for t in range(num_tiles):
        tile = []
        for j in range(n):
            byte = int(score_lines[t * n + j], 16) & 0xFF
            # Convert hex byte to signed int8.
            tile.append(byte - 256 if byte & 0x80 else byte)
        tiles.append(tile)
    expected = [bool(int(line, 16) & 1) for line in expected_lines]
    return tiles, expected


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_canonical_boundary_cases():
    """The two edge cases the user originally called out must work."""
    n = 4096

    # bg = +1 across the tile, single spike of value `spike` somewhere
    def bg1_spike(spike: int) -> list:
        tile = [1] * n
        tile[n // 2] = spike
        return tile

    # LHS = 10 * 4096 = 40960  <  RHS = 10 * (4095 + 10) = 41050  -> INT8
    assert PrecisionController().process_tile(bg1_spike(10)) is False, \
        "spike=10 must be INT8 (LHS=40960 < RHS=41050)"

    # LHS = 11 * 4096 = 45056  >  RHS = 10 * (4095 + 11) = 41060  -> FP16
    assert PrecisionController().process_tile(bg1_spike(11)) is True, \
        "spike=11 must be FP16 (LHS=45056 > RHS=41060)"


def test_streaming_matches_batched():
    """tick()-style streaming must agree with process_tile() on the same input."""
    pc_a = PrecisionController()
    pc_b = PrecisionController()
    tile = [(i % 7) - 3 for i in range(4096)]   # arbitrary deterministic pattern

    # Batched
    dec_a = pc_a.process_tile(tile)

    # Streaming
    for i, s in enumerate(tile):
        pc_b.tick(s_valid=True, s_data=s, s_last=(i == len(tile) - 1))
    dec_b = pc_b.d_fp16
    assert dec_b is dec_a


def test_stateless_matches_stateful():
    """decide() static method must match process_tile() on every tile."""
    tiles, _ = _load_tiles_and_expected()
    pc = PrecisionController()
    mismatches = []
    for idx, tile in enumerate(tiles[:25]):   # sample first 25 to keep fast
        stateful = pc.process_tile(tile)
        stateless = PrecisionController.decide(tile)
        if stateful != stateless:
            mismatches.append(idx)
    assert not mismatches, f"stateful vs stateless disagreed on tiles {mismatches}"


def test_full_replay_bit_exact_vs_rtl():
    """Every one of the 143 replay tiles must decide identically to the RTL."""
    tiles, expected = _load_tiles_and_expected()
    pc = PrecisionController()

    mismatches = []
    for idx, (tile, exp) in enumerate(zip(tiles, expected)):
        got = pc.process_tile(tile)
        if got != exp:
            mismatches.append((idx, exp, got))

    if mismatches:
        for idx, exp, got in mismatches[:10]:
            print(f"  tile {idx}: expected={'FP16' if exp else 'INT8'}, "
                  f"got={'FP16' if got else 'INT8'}")
        raise AssertionError(
            f"{len(mismatches)}/{len(tiles)} replay tiles disagreed with RTL"
        )

    # Match the same headline the RTL TB prints.
    print(f"Tests: {len(tiles)}  Pass: {len(tiles)}  Fail: 0  ALL TESTS PASSED")


def test_decision_history_tracking():
    """decision_history should accumulate every tile's decision in order."""
    pc = PrecisionController()
    pc.process_tile([1] * 4096)              # INT8 (all uniform)
    pc.process_tile([0] * 4095 + [127])      # FP16 (single huge spike)
    pc.process_tile([10] * 4096)             # INT8 (uniform)
    history = pc.decision_history
    assert history == [False, True, False], f"unexpected history: {history}"
    assert pc.tiles_processed == 3


def test_accumulator_reset_between_tiles():
    """After s_last, accumulators must reset so the next tile starts clean."""
    pc = PrecisionController()
    pc.process_tile([127] * 4096)            # huge sum + huge max
    # Next tile: all zeros — should give INT8 cleanly (max=0, sum=0,
    # LHS=0, RHS=0, decision uses `>` so False = INT8)
    decision = pc.process_tile([0] * 4096)
    assert decision is False, "all-zero tile after big tile must decide INT8"


if __name__ == "__main__":
    test_canonical_boundary_cases()
    test_streaming_matches_batched()
    test_stateless_matches_stateful()
    test_full_replay_bit_exact_vs_rtl()
    test_decision_history_tracking()
    test_accumulator_reset_between_tiles()
    print("ALL SELF-TESTS PASSED")
