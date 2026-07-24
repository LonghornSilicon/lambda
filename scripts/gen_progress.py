#!/usr/bin/env python3
"""gen_progress.py — derive the chip progress matrix from ground-truth sign-off artifacts.

The whole point of this script: STATUS is *computed* from committed metrics JSONs, never
hand-copied. That structurally kills the drift class where a block advances (e.g. RoPE/RMSNorm
GDSII-signed) but chip-level docs still say "no RTL".

Sign-off classification (see docs/REVISION_SYNC_SOP.md §5.2):
  signed-off  : full-signoff flow (sky130/gf180), all headline checks 0, GDS present
  caveated    : full-signoff flow, GDS present, but one or more headline checks nonzero
  route-clean : ORFS/ASAP7 flow (route-DRC + antenna only; NO Magic-DRC/LVS) — NOT full sign-off
  config-only : a macro is declared (config.json / <macro>.yaml / config.mk) with no results
  no-gds      : metrics present but no GDS artifact

Usage:
  python3 scripts/gen_progress.py            # write docs/PROGRESS.md from ground truth
  python3 scripts/gen_progress.py --check    # exit 1 if docs/PROGRESS.md is stale (for pre-commit/CI)
  python3 scripts/gen_progress.py --stdout    # print the matrix, don't write

Stdlib only. Run from the monorepo root (or anywhere — it locates the root via this file).
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# short block ids (used for display + as the ISA/summary key); functional blocks live under
# src/blocks/, the integration block `chip` stays at top level.
BLOCKS = ["kve", "tiu", "acu/mate", "acu/vecu", "acu/precision_controller", "chip"]
BLOCK_ROOT = "src/blocks"
OUT = ROOT / "docs" / "PROGRESS.md"


def bpath(block: str) -> str:
    """Map a short block id to its repo-relative path."""
    return "chip" if block == "chip" else f"{BLOCK_ROOT}/{block}"

# headline sign-off checks that must all be 0 for a full-signoff-flow macro to be "signed-off"
SIGNOFF_CHECKS = [
    "magic__drc_error__count",
    "klayout__drc_error__count",
    "design__lvs_error__count",
    "antenna__violating__nets",
    "antenna__violating__pins",
    "timing__setup_vio__count",
    "timing__hold_vio__count",
]

FREQ_RE = re.compile(r"(\d+(?:\.\d+)?)(MHz|GHz)", re.IGNORECASE)


def _gds_near(results_dir: Path) -> bool:
    """A GDS artifact exists in or below this results dir."""
    for pat in ("*.gds", "*.gds.gz"):
        if any(results_dir.rglob(pat)):
            return True
    return False


def _freq_from_name(p: Path) -> str:
    m = FREQ_RE.search(p.name)
    return f"{m.group(1)} {m.group(2).replace('m','M').replace('g','G')}" if m else ""


def _freq_from_config(results_dir: Path) -> str:
    """Fallback: derive frequency from CLOCK_PERIOD (ns) in the sibling flow config.
    Keeps frequency visible after metrics filenames are normalized (freq stripped from names)."""
    def mhz(period_ns) -> str:
        try:
            return f"{1000/float(period_ns):.0f} MHz"
        except (TypeError, ValueError, ZeroDivisionError):
            return ""
    # sky130 OpenLane: <macro>/results/  ->  <macro>/config.json
    cfg = results_dir.parent / "config.json"
    if cfg.exists():
        try:
            return mhz(json.loads(cfg.read_text()).get("CLOCK_PERIOD"))
        except Exception:  # noqa: BLE001
            pass
    # gf180 LibreLane: librelane/results/<macro>/  ->  librelane/<macro>.yaml
    yml = results_dir.parent.parent / f"{results_dir.name}.yaml"
    if yml.exists():
        m = re.search(r'CLOCK_PERIOD["\s:]+([0-9.]+)', yml.read_text())
        if m:
            return mhz(m.group(1))
    # asap7 ORFS: <macro>/results_asap7/  ->  <macro>/constraint.sdc ("set clk_period <ps>")
    sdc = results_dir.parent / "constraint.sdc"
    if sdc.exists():
        m = re.search(r'set\s+clk_period\s+([0-9.]+)', sdc.read_text())
        if m:  # picoseconds -> MHz  (1e6 / ps)
            try:
                return f"{1e6/float(m.group(1)):.0f} MHz"
            except (ValueError, ZeroDivisionError):
                pass
    return ""


def classify(metrics_path: Path) -> dict:
    """Classify one metrics JSON into a sign-off record."""
    try:
        d = json.loads(metrics_path.read_text())
    except Exception as e:  # noqa: BLE001 — a malformed metrics file is itself a finding
        return {"status": "unreadable", "detail": str(e), "die_um2": None, "freq": ""}

    results_dir = metrics_path.parent
    gds = _gds_near(results_dir)
    die = d.get("design__die__area")
    freq = _freq_from_name(metrics_path) or _freq_from_config(results_dir)

    if "magic__drc_error__count" in d:  # full-signoff flow (OpenLane / LibreLane)
        failed = [k for k in SIGNOFF_CHECKS if k in d and d[k] not in (0, 0.0)]
        if not gds:
            return {"status": "no-gds", "detail": "metrics but no GDS", "die_um2": die, "freq": freq}
        if failed:
            return {"status": "caveated", "detail": ",".join(failed), "die_um2": die, "freq": freq}
        return {"status": "signed-off", "detail": "", "die_um2": die, "freq": freq}

    if "detailedroute__route__drc_errors" in d:  # ORFS / ASAP7 route-only flow
        rc = d["detailedroute__route__drc_errors"]
        if not gds:
            return {"status": "no-gds", "detail": "route metrics but no GDS", "die_um2": die, "freq": freq}
        if rc not in (0, 0.0):
            return {"status": "caveated", "detail": f"route DRC={rc}", "die_um2": die, "freq": freq}
        return {"status": "route-clean", "detail": "no Magic-DRC/LVS step", "die_um2": die, "freq": freq}

    return {"status": "unknown-flow", "detail": "no recognized check keys", "die_um2": die, "freq": freq}


def declared_macros(block_dir: Path) -> dict:
    """Return {(pdk, macro): declaring_config_path} for every declared PDK run in a block."""
    out: dict[tuple[str, str], Path] = {}
    # sky130 OpenLane:  pdk/sky130/openlane/<macro>/config.json
    for cfg in block_dir.glob("pdk/sky130/openlane/*/config.json"):
        out[("sky130", cfg.parent.name)] = cfg
    # gf180 LibreLane:  pdk/gf180/librelane/<macro>.yaml  (skip chip-level config_*.yaml)
    for cfg in block_dir.glob("pdk/gf180/librelane/*.yaml"):
        macro = cfg.stem
        if macro.startswith("config_") or macro.startswith("pdn"):
            continue
        out[("gf180", macro)] = cfg
    # asap7 ORFS:  pdk/asap7/orfs/asap7/<macro>/config.mk
    for cfg in block_dir.glob("pdk/asap7/orfs/asap7/*/config.mk"):
        out[("asap7", cfg.parent.name)] = cfg
    return out


def found_metrics(block_dir: Path) -> dict:
    """Return {(pdk, macro): metrics_path} for every committed metrics JSON in a block."""
    out: dict[tuple[str, str], Path] = {}
    for m in block_dir.rglob("*metrics*.json"):
        parts = m.parts
        pdk = next((p for p in ("sky130", "gf180", "asap7") if p in parts), "?")
        # macro name = the results dir's parent name, except gf180 nests results/<macro>/
        if "results" in parts:
            i = parts.index("results")
            macro = parts[i + 1] if (i + 1 < len(parts) and parts[i + 1] != m.name) else m.parent.parent.name
        else:
            macro = m.parent.parent.name
        out[(pdk, macro)] = m
    return out


def scan_block(block: str) -> list[dict]:
    bdir = ROOT / bpath(block)
    rows = []
    declared = declared_macros(bdir)
    metrics = found_metrics(bdir)
    keys = sorted(set(declared) | set(metrics))
    for (pdk, macro) in keys:
        if (pdk, macro) in metrics:
            rec = classify(metrics[(pdk, macro)])
            rec.update(block=block, pdk=pdk, macro=macro,
                       artifact=str(metrics[(pdk, macro)].relative_to(ROOT)))
        else:
            rec = dict(block=block, pdk=pdk, macro=macro, status="config-only",
                       detail="declared, not run", die_um2=None, freq="",
                       artifact=str(declared[(pdk, macro)].relative_to(ROOT)))
        rows.append(rec)
    # chip full-chip GDS with no metrics JSON = prose-only sign-off (special, no config macro)
    if block == "chip":
        for gds in bdir.rglob("*.gds.gz"):
            if not any(r["macro"] in gds.name for r in rows):
                rows.append(dict(block=block, pdk="gf180", macro=gds.stem.replace(".gds", ""),
                                 status="prose-only", detail="GDS present, no metrics JSON",
                                 die_um2=None, freq="", artifact=str(gds.relative_to(ROOT))))
    return rows


def render(rows: list[dict]) -> str:
    lines = [
        "# PROGRESS.md — Lambda sign-off matrix (GENERATED)",
        "",
        "> **Generated by `scripts/gen_progress.py` from committed `*metrics*.json` artifacts.**",
        "> Do not hand-edit — run the script. `--check` fails if this file is stale. Status legend +",
        "> per-flow sign-off definition: `docs/REVISION_SYNC_SOP.md` §5.2.",
        "",
        "| Block | PDK | Macro | Status | Die (µm²) | Freq | Note / artifact |",
        "|---|---|---|---|---|---|---|",
    ]
    order = {"signed-off": 0, "route-clean": 1, "caveated": 2, "prose-only": 3,
             "config-only": 4, "no-gds": 5, "unknown-flow": 6, "unreadable": 7}
    for r in sorted(rows, key=lambda r: (r["block"], order.get(r["status"], 9), r["pdk"], r["macro"])):
        die = f'{r["die_um2"]:.0f}' if isinstance(r["die_um2"], (int, float)) else ""
        note = r["detail"] or ""
        lines.append(f'| `{r["block"]}` | {r["pdk"]} | {r["macro"]} | **{r["status"]}** | '
                     f'{die} | {r["freq"]} | {note} |')
    # summary counts
    from collections import Counter
    c = Counter(r["status"] for r in rows)
    lines += ["", "**Totals:** " + " · ".join(f"{k}={c[k]}" for k in sorted(c)), ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/PROGRESS.md is stale")
    ap.add_argument("--stdout", action="store_true", help="print, do not write")
    args = ap.parse_args()

    rows = []
    for b in BLOCKS:
        rows += scan_block(b)
    content = render(rows)

    if args.stdout:
        sys.stdout.write(content)
        return 0
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != content:
            sys.stderr.write("PROGRESS.md is STALE — run `python3 scripts/gen_progress.py`.\n")
            return 1
        print("PROGRESS.md is current.")
        return 0
    OUT.write_text(content)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} macro×PDK rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
