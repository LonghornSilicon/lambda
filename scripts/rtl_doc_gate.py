#!/usr/bin/env python3
"""rtl_doc_gate.py — the lab-notebook gate: RTL changes must carry docs.

Enforces AGENTS.md §Lab-notebook + docs/documentation_standard.md: a commit that touches a block's
rtl/ MUST also touch that block's docs/ or DECISIONS.md (or README.md). "Docs travel with code."

Runs as a pre-commit hook (inspects the staged index) or standalone. Exit 1 on violation.

Usage:
  python3 scripts/rtl_doc_gate.py            # check staged changes (git diff --cached)
  SKIP=rtl_doc_gate git commit ...           # bypass via pre-commit's SKIP when genuinely doc-free
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCKS = ["kve", "tiu", "acu/mate", "acu/vecu", "acu/precision_controller", "chip"]


def staged_files() -> list[str]:
    out = subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "--cached", "--name-only"], text=True)
    return [ln for ln in out.splitlines() if ln.strip()]


def main() -> int:
    files = staged_files()
    if not files:
        return 0
    violations = []
    for block in BLOCKS:
        rtl_changed = any(f.startswith(f"{block}/rtl/") for f in files)
        if not rtl_changed:
            continue
        doc_changed = (
            any(f.startswith(f"{block}/docs/") for f in files)
            or f"{block}/DECISIONS.md" in files
            or f"{block}/README.md" in files
        )
        if not doc_changed:
            violations.append(block)

    if violations:
        sys.stderr.write(
            "rtl-doc gate FAILED — RTL changed without a doc/decision update:\n")
        for b in violations:
            sys.stderr.write(
                f"  [{b}] staged rtl/ change with no staged {b}/docs/, "
                f"{b}/DECISIONS.md, or {b}/README.md\n")
        sys.stderr.write(
            "\nDocs travel with code (AGENTS.md §Lab-notebook). Update the block's docs/DECISIONS,\n"
            "or if this change is genuinely doc-free, bypass with: SKIP=rtl_doc_gate git commit ...\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
