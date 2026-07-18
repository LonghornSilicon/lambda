"""Python <-> RTL parity test for the Token Importance Unit reference model.

Replays the committed golden trace `rtl/tb/testvectors/tiu_trace.hex` -- the SAME
op stream the SystemVerilog replay testbench (`rtl/tb/tb_realdata.sv`) drives into
the DUT -- through the Python model, and checks that every EVICT victim the model
returns matches the trace's expected-victim column.

The trace was generated from a real Qwen2-0.5B forward pass by
`analysis/gen_tiu_testvectors.py`; its expected column is the eviction victim the
RTL argmin produces. If the model agrees on all 40 evictions, the Python model is
bit-exact with the RTL on real data.

Run:   python -m pytest sw/reference_model/test_tiu_ref.py
Or:    python sw/reference_model/test_tiu_ref.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tiu_ref import TokenImportanceUnit, TokenImportanceUnitInfo  # noqa: E402

REPO_ROOT = HERE.parent.parent
TRACE = REPO_ROOT / "rtl" / "tb" / "testvectors" / "tiu_trace.hex"

# Op codes in the trace, matching analysis/gen_tiu_testvectors.py:
OP_ACC, OP_LOAD, OP_EVICT = 0, 1, 2


def _parse_trace(path: Path):
    """Yield (op, slot, weight, expected_victim) for each non-comment line."""
    ops = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        fields = line.split()
        if len(fields) != 4:
            continue
        op, slot, weight, exp = (int(f, 16) for f in fields)
        ops.append((op, slot, weight, exp))
    return ops


def _replay(info: TokenImportanceUnitInfo):
    """Replay the trace through the model; return (n_ops, n_evict, mismatches)."""
    assert TRACE.exists(), f"golden trace not found: {TRACE}"
    tiu = TokenImportanceUnit(info)
    ops = _parse_trace(TRACE)

    n_evict = 0
    mismatches = []
    for idx, (op, slot, weight, exp) in enumerate(ops):
        if op == OP_ACC:
            tiu.acc(slot, weight)
        elif op == OP_LOAD:
            tiu.load(slot)
        elif op == OP_EVICT:
            got = tiu.evict()
            n_evict += 1
            if got != exp:
                mismatches.append((idx, exp, got))
        else:
            raise AssertionError(f"unknown op {op} at line {idx}")
    return len(ops), n_evict, mismatches


def test_replay_matches_rtl():
    """Every EVICT victim from the model must match the trace's expected column."""
    info = TokenImportanceUnitInfo()
    n_ops, n_evict, mismatches = _replay(info)

    assert n_evict > 0, "trace contains no evictions -- nothing to check"
    if mismatches:
        for idx, exp, got in mismatches[:10]:
            print(f"  op {idx}: expected victim {exp}, got {got}")
        raise AssertionError(
            f"{len(mismatches)}/{n_evict} eviction victims disagreed with RTL")

    print(f"Replay: {n_ops} ops streamed, {n_evict} evictions, "
          f"{n_evict}/{n_evict} victims match RTL  ALL TESTS PASSED")


def test_trace_has_expected_shape():
    """Sanity: the committed golden trace is the 48-token / 40-eviction stream."""
    ops = _parse_trace(TRACE)
    n_evict = sum(1 for op, *_ in ops if op == OP_EVICT)
    assert n_evict == 40, f"expected 40 evictions in the golden trace, got {n_evict}"


if __name__ == "__main__":
    test_trace_has_expected_shape()
    test_replay_matches_rtl()
    print("ALL SELF-TESTS PASSED")
