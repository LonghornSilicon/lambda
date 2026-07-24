#!/usr/bin/env python3
"""check_block_structure.py — lint each block against the canonical template.

Enforces docs/REVISION_SYNC_SOP.md §5 so every block is clean and succinct in the same way,
letting team leads coordinate versioned changes without per-block surprises.

Levels:
  ERROR — a required file/dir is missing. Exit code 1.
  WARN  — a convention deviation (filename schema, dangling config, date precision). Exit 0
          unless --strict.

Usage:
  python3 scripts/check_block_structure.py            # report; exit 1 on any ERROR
  python3 scripts/check_block_structure.py --strict   # exit 1 on ERROR or WARN

Stdlib only. Locates the monorepo root via this file's path.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCK_ROOT = "src/blocks"


def bpath(block: str) -> str:
    """Map a short block id to its repo-relative path."""
    return "chip" if block == "chip" else f"{BLOCK_ROOT}/{block}"


# canonical functional-block template (docs/REVISION_SYNC_SOP.md §5.1)
FUNCTIONAL_BLOCKS = ["kve", "tiu", "acu/mate", "acu/vecu", "acu/precision_controller"]
REQUIRED_FILES = ["README.md", "DECISIONS.md", "AGENTS.md"]
REQUIRED_DIRS = ["rtl", "sw/reference_model", "docs", "pdk", "research"]

# integration block — explicit exemption (§5.1)
INTEGRATION_BLOCKS = ["chip"]
INTEGRATION_REQUIRED_FILES = ["README.md", "DECISIONS.md"]
INTEGRATION_REQUIRED_DIRS = ["rtl", "pdk"]

# canonical metrics filename (§5.2): full-signoff flows use <pdk>_signoff_metrics.json;
# route-only ASAP7/ORFS uses asap7_route_metrics.json (honest: it is not a full sign-off).
GOOD_METRICS_RE = re.compile(r"^(sky130|gf180)_signoff_metrics\.json$|^asap7_route_metrics\.json$")
# date must be full ISO in DECISIONS rows (§ DECISIONS convention).
# Be PRECISE — never match a plain 4-digit number (test-vector counts like 5120/2048).
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# year-month without a day, e.g. "2026-07" (but NOT the "2026-07" inside "2026-07-18")
MONTH_ONLY_RE = re.compile(r"\b20\d{2}-\d{2}\b(?!-\d{2})")
# a bare year sitting in a date position: end-of-field "· 2026", or after decided/added/merged/dated
YEAR_IN_DATE_POS_RE = re.compile(
    r"·\s*20\d{2}\b\.?\s*$"                       # "... · 2026" (trailing date field)
    r"|\b(?:decided|added|merged|dated|as of)\s+20\d{2}\b(?!-)",  # "decided 2026"
    re.IGNORECASE,
)

findings: list[tuple[str, str, str]] = []  # (level, block, message)


def err(block: str, msg: str) -> None:
    findings.append(("ERROR", block, msg))


def warn(block: str, msg: str) -> None:
    findings.append(("WARN", block, msg))


def check_functional(block: str) -> None:
    bdir = ROOT / bpath(block)
    for f in REQUIRED_FILES:
        if not (bdir / f).is_file():
            err(block, f"missing required file: {f}")
    for d in REQUIRED_DIRS:
        if not (bdir / d).is_dir():
            err(block, f"missing required dir: {d}/")

    # reference model must have a ref + a test (§5.1)
    refdir = bdir / "sw" / "reference_model"
    if refdir.is_dir():
        pyfiles = list(refdir.rglob("*.py"))
        has_ref = any("_ref" in p.name for p in pyfiles)
        has_test = any(p.name.startswith("test_") for p in pyfiles)
        if not has_ref:
            warn(block, "sw/reference_model/ has no *_ref.* model")
        if not has_test:
            warn(block, "sw/reference_model/ has no test_* parity test")

    check_metrics_naming(block, bdir)
    check_dangling_configs(block, bdir)
    check_decision_dates(block, bdir)


def check_integration(block: str) -> None:
    bdir = ROOT / bpath(block)
    for f in INTEGRATION_REQUIRED_FILES:
        if not (bdir / f).is_file():
            err(block, f"missing required file: {f} (integration block)")
    for d in INTEGRATION_REQUIRED_DIRS:
        if not (bdir / d).is_dir():
            err(block, f"missing required dir: {d}/ (integration block)")
    check_metrics_naming(block, bdir)


def check_metrics_naming(block: str, bdir: Path) -> None:
    for m in bdir.rglob("*metrics*.json"):
        if not GOOD_METRICS_RE.match(m.name):
            warn(block, f"metrics filename not <pdk>_signoff_metrics.json: "
                        f"{m.relative_to(ROOT)}")


def check_dangling_configs(block: str, bdir: Path) -> None:
    """A declared PDK macro with no results is 'declared, not run' — flag it."""
    # sky130
    for cfg in bdir.glob("pdk/sky130/openlane/*/config.json"):
        if not list((cfg.parent / "results").rglob("*metrics*.json")):
            warn(block, f"declared sky130 macro not run (no results): {cfg.parent.name}")
    # gf180
    for cfg in bdir.glob("pdk/gf180/librelane/*.yaml"):
        macro = cfg.stem
        if macro.startswith(("config_", "pdn")):
            continue
        resdir = cfg.parent / "results" / macro
        if not (resdir.is_dir() and list(resdir.rglob("*metrics*.json"))):
            warn(block, f"declared gf180 macro not run (no results): {macro}")


def check_decision_dates(block: str, bdir: Path) -> None:
    dfile = bdir / "DECISIONS.md"
    if not dfile.is_file():
        return
    for i, line in enumerate(dfile.read_text().splitlines(), 1):
        if ISO_DATE_RE.search(line):
            continue  # has a full ISO date on the line — fine
        loose = MONTH_ONLY_RE.search(line) or YEAR_IN_DATE_POS_RE.search(line)
        if loose:
            warn(block, f"DECISIONS.md:{i} date not full-ISO (YYYY-MM-DD): "
                        f"...{loose.group(0).strip()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="exit 1 on WARN too")
    args = ap.parse_args()

    for b in FUNCTIONAL_BLOCKS:
        if (ROOT / bpath(b)).is_dir():
            check_functional(b)
        else:
            err(b, "block directory does not exist")
    for b in INTEGRATION_BLOCKS:
        if (ROOT / bpath(b)).is_dir():
            check_integration(b)

    errors = [f for f in findings if f[0] == "ERROR"]
    warns = [f for f in findings if f[0] == "WARN"]
    for level, block, msg in sorted(findings):
        print(f"{level:5} [{block}] {msg}")
    print(f"\n{len(errors)} error(s), {len(warns)} warning(s)")

    if errors or (args.strict and warns):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
