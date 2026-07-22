"""Synthesis parameter sweep for precision_controller.

For each (BLOCK_M, BLOCK_N, SCORE_WIDTH) point we drive Yosys with `chparam`
to override the module parameters, run `synth -flatten` then `abc -fast -g NAND`,
parse the post-mapping `stat -width` block, and record:

  - FF count          ($_DFF_P_)
  - NAND2 cell count  ($_NAND_)
  - NOT cell count    ($_NOT_)
  - NAND2-equivalent  (NAND + NOT/2, conventional shorthand)

Outputs (under analysis/):
  rtl_sweep_results.json           — raw machine-readable table
  figures/rtl_area_vs_tile.png     — area + FFs vs tile size N
  figures/rtl_area_vs_bitwidth.png — area + FFs vs SCORE_WIDTH (tile=64x64)
  rtl_sweep_notes.md               — human-readable summary

Run from rtl/:    python3 sweep_synth.py
"""

import json
import re
import shutil
import subprocess
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parent.parent
RTL_DIR = REPO / "rtl"
RTL_FILE = RTL_DIR / "precision_controller.sv"
ANALYSIS_DIR = REPO / "analysis"
FIGURES_DIR = ANALYSIS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

YOSYS = shutil.which("yosys")
assert YOSYS, "yosys not on PATH"


# Tile sweep at SCORE_WIDTH=8 — N = BLOCK_M*BLOCK_N stepped over 8 powers of two.
TILE_SWEEP = [
    (16, 16),
    (32, 32),
    (32, 64),
    (64, 64),
    (64, 128),
    (128, 128),
    (128, 256),
    (256, 256),
]
TILE_SCORE_WIDTH = 8

# Bit-width sweep at tile = 64x64 — matches the bit-width set in fixed_point_sim.py.
BITWIDTH_SWEEP_TILE = (64, 64)
BITWIDTH_SWEEP = [4, 6, 8, 10, 12, 16]

THRESHOLD = 10


SYNTH_SCRIPT = """\
read_verilog -sv {rtl}
chparam -set BLOCK_M {bm} -set BLOCK_N {bn} -set SCORE_WIDTH {sw} -set THRESHOLD {th} precision_controller
hierarchy -check -top precision_controller
synth -top precision_controller -flatten
abc -fast -g NAND
stat -width
"""


# Parse `stat -width` output. Cells appear as `     $_NAND_                       9`
CELL_RE = re.compile(r"^\s+\$_(\w+)_\s+(\d+)\s*$")
CELLS_TOTAL_RE = re.compile(r"Number of cells:\s+(\d+)")


def run_yosys(bm: int, bn: int, sw: int, th: int = THRESHOLD) -> dict:
    script = SYNTH_SCRIPT.format(rtl=RTL_FILE, bm=bm, bn=bn, sw=sw, th=th)
    proc = subprocess.run(
        [YOSYS, "-p", script],
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"yosys failed for BLOCK_M={bm} BLOCK_N={bn} SCORE_WIDTH={sw}")

    out = proc.stdout

    # We only want the *last* stat block (post-NAND mapping).
    last_stat = out.rfind("Printing statistics.")
    block = out[last_stat:] if last_stat >= 0 else out

    counts: dict = {}
    total_cells = None
    for line in block.splitlines():
        m = CELL_RE.match(line)
        if m:
            counts[m.group(1)] = int(m.group(2))
            continue
        m2 = CELLS_TOTAL_RE.search(line)
        if m2:
            total_cells = int(m2.group(1))

    ff = counts.get("DFF_P", 0)
    nand = counts.get("NAND", 0)
    nand_not = counts.get("NOT", 0)
    nand2_eq = nand + nand_not / 2.0     # NOT ≈ ½ NAND2 (textbook shorthand)

    n_tile = bm * bn
    log2_n = int(math.log2(n_tile))
    sum_w = sw + log2_n
    cmp_w = sum_w + 4  # +log2(THRESHOLD ceil)

    return {
        "BLOCK_M": bm,
        "BLOCK_N": bn,
        "N": n_tile,
        "LOG2_N": log2_n,
        "SCORE_WIDTH": sw,
        "SUM_W": sum_w,
        "CMP_W": cmp_w,
        "FF": ff,
        "NAND": nand,
        "NOT": nand_not,
        "NAND2_eq": round(nand2_eq, 1),
        "total_cells": total_cells,
        "raw_counts": counts,
    }


def md_table(rows: list, columns: list) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(str(r[c]) for c in columns) + " |")
    return "\n".join([header, sep] + body)


def plot_tile_sweep(rows: list, path: Path) -> None:
    rows = sorted(rows, key=lambda r: r["N"])
    n_vals = [r["N"] for r in rows]
    ffs = [r["FF"] for r in rows]
    gates = [r["NAND2_eq"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    color1 = "tab:blue"
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Tile size N = BLOCK_M × BLOCK_N (log2)")
    ax1.set_ylabel("NAND2-equivalent gates", color=color1)
    ax1.plot(n_vals, gates, "o-", color=color1, label="NAND2-eq gates")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Flip-flops", color=color2)
    ax2.plot(n_vals, ffs, "s--", color=color2, label="FFs")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_xticks(n_vals)
    ax1.set_xticklabels([str(n) for n in n_vals], rotation=0)

    plt.title(f"precision_controller area vs tile size  (SCORE_WIDTH={TILE_SCORE_WIDTH}, T={THRESHOLD})")
    fig.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def plot_bitwidth_sweep(rows: list, path: Path) -> None:
    rows = sorted(rows, key=lambda r: r["SCORE_WIDTH"])
    sw_vals = [r["SCORE_WIDTH"] for r in rows]
    ffs = [r["FF"] for r in rows]
    gates = [r["NAND2_eq"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    color1 = "tab:blue"
    ax1.set_xlabel("SCORE_WIDTH (bits)")
    ax1.set_ylabel("NAND2-equivalent gates", color=color1)
    ax1.plot(sw_vals, gates, "o-", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Flip-flops", color=color2)
    ax2.plot(sw_vals, ffs, "s--", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_xticks(sw_vals)
    bm, bn = BITWIDTH_SWEEP_TILE
    plt.title(f"precision_controller area vs SCORE_WIDTH  (tile={bm}x{bn}, T={THRESHOLD})")
    fig.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    print("=" * 68)
    print(" Yosys parameter sweep — precision_controller")
    print("=" * 68)

    print("\n[1/2] Tile-size sweep  (SCORE_WIDTH = 8)")
    print(f"{'BLOCK':>10} {'N':>7} {'log2N':>6} {'SUM_W':>6} {'CMP_W':>6} "
          f"{'FF':>4} {'NAND':>5} {'NOT':>5} {'NAND2eq':>8}")
    tile_rows = []
    for bm, bn in TILE_SWEEP:
        r = run_yosys(bm, bn, TILE_SCORE_WIDTH)
        tile_rows.append(r)
        print(f"  {bm:>3}x{bn:<3}  {r['N']:>7}  {r['LOG2_N']:>4}  "
              f"{r['SUM_W']:>5}  {r['CMP_W']:>5}  {r['FF']:>4}  "
              f"{r['NAND']:>5}  {r['NOT']:>5}  {r['NAND2_eq']:>8.1f}")

    print("\n[2/2] SCORE_WIDTH sweep  (tile = {0}x{1})".format(*BITWIDTH_SWEEP_TILE))
    print(f"{'SW':>3} {'SUM_W':>6} {'CMP_W':>6} {'FF':>4} {'NAND':>5} {'NOT':>5} {'NAND2eq':>8}")
    bm, bn = BITWIDTH_SWEEP_TILE
    bw_rows = []
    for sw in BITWIDTH_SWEEP:
        r = run_yosys(bm, bn, sw)
        bw_rows.append(r)
        print(f"  {sw:>3}  {r['SUM_W']:>5}  {r['CMP_W']:>5}  {r['FF']:>4}  "
              f"{r['NAND']:>5}  {r['NOT']:>5}  {r['NAND2_eq']:>8.1f}")

    # Persist
    out_json = ANALYSIS_DIR / "rtl_sweep_results.json"
    with out_json.open("w") as fh:
        json.dump({"tile_sweep": tile_rows, "bitwidth_sweep": bw_rows}, fh, indent=2)
    print(f"\nResults JSON → {out_json.relative_to(REPO)}")

    p1 = FIGURES_DIR / "rtl_area_vs_tile.png"
    p2 = FIGURES_DIR / "rtl_area_vs_bitwidth.png"
    plot_tile_sweep(tile_rows, p1)
    plot_bitwidth_sweep(bw_rows, p2)
    print(f"Plot 1     → {p1.relative_to(REPO)}")
    print(f"Plot 2     → {p2.relative_to(REPO)}")

    # Markdown notes
    cols_tile = ["BLOCK_M", "BLOCK_N", "N", "LOG2_N", "SUM_W", "CMP_W",
                 "FF", "NAND", "NOT", "NAND2_eq"]
    cols_bw = ["SCORE_WIDTH", "SUM_W", "CMP_W", "FF", "NAND", "NOT", "NAND2_eq"]

    notes = ANALYSIS_DIR / "rtl_sweep_notes.md"
    with notes.open("w") as fh:
        fh.write("# precision_controller — Yosys sweep notes\n\n")
        fh.write(f"Tool: Yosys 0.9 (apt) · `synth -flatten` then `abc -fast -g NAND`.\n")
        fh.write(f"NAND2-equivalent counted as `NAND + NOT/2`.\n\n")
        fh.write("## Tile-size sweep (SCORE_WIDTH = 8)\n\n")
        fh.write(md_table(tile_rows, cols_tile) + "\n\n")
        fh.write("![tile sweep](figures/rtl_area_vs_tile.png)\n\n")
        fh.write("## SCORE_WIDTH sweep (tile = {0}x{1})\n\n".format(*BITWIDTH_SWEEP_TILE))
        fh.write(md_table(bw_rows, cols_bw) + "\n\n")
        fh.write("![bitwidth sweep](figures/rtl_area_vs_bitwidth.png)\n")
    print(f"Notes      → {notes.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
