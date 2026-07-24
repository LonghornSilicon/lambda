#!/usr/bin/env bash
# ============================================================================
# lambda-env.sh
# ----------------------------------------------------------------------------
# Committed defaults for Lambda chamber tooling. Sourced by every lambda-*
# launcher AND by every generic helper in tools/bin/ (stratus-gui etc).
#
# Per-user / per-machine overrides go in ~/.longhorn/lambda.env (gitignored,
# parallel to ~/.longhorn/chamber.env used by sync-chamber.sh).
#
# Precedence (v0.4.2, M3 fix): ~/.longhorn/lambda.env is sourced FIRST, then
# committed defaults are applied with the ${VAR:=default} form. Net effect:
#   1. plain `VAR=...` assignments in lambda.env  — user values win
#   2. values already exported in the environment — win over the defaults
#   3. committed defaults below                   — fill whatever is left
# Sourcing lambda.env BEFORE the derivations matters: derived vars
# (LAMBDA_LOGS/BUILD/SCRATCH) are computed from LAMBDA_WORK, so an override of
# LAMBDA_WORK in lambda.env now propagates into them instead of leaving them
# pointing at the stale default tree (the pre-v0.4.2 bug). A lambda.env that
# wants the inherited environment to beat it can itself use `: "${VAR:=...}"`.
# LAMBDA_BLOCKS is a project invariant and is assigned unconditionally below —
# it cannot be overridden per-user. Sourcing this file is idempotent.
# ============================================================================

# ---- Per-user overrides (sourced FIRST — see precedence note above) --------
if [[ -f "$HOME/.longhorn/lambda.env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.longhorn/lambda.env"
fi

# ---- Project paths ---------------------------------------------------------
# Three storage classes; see docs/tools-overview.md "Filesystem & run-area".
#   LAMBDA_ROOT     ~/architecture  — source/methodology, git mirror (read-only).
#                                     `sync-promote` does `git reset --hard`, so
#                                     NO tool output should land here.
#   LAMBDA_WORK     ~/work/lambda   — run/work area, home-backed (NFS, cross-node).
#                                     The only reliable persistent writable space:
#                                     /rscratch/$USER is unprovisioned; /projects
#                                     has no group dir; /tmp is node-local/ephemeral.
#   LAMBDA_FAST     /tmp/$USER-lambda — node-local ephemeral; for huge transients
#                                     when LAMBDA_WORK is tight (wiped on reboot).
: "${LAMBDA_ROOT:=$HOME/architecture}"
: "${CHAMBER_WORK:=$HOME/work}"
: "${LAMBDA_WORK:=$CHAMBER_WORK/lambda}"
: "${LAMBDA_LOGS:=$LAMBDA_WORK/logs}"
: "${LAMBDA_FAST:=/tmp/${USER}-lambda}"
# Back-compat alias: every existing $LAMBDA_BUILD/... reference in launchers
# (lambda-stratus, lambda-innovus, lambda-diagnose) follows LAMBDA_WORK with no
# per-file edits. NB: src/blocks/<b>/stratus/project.tcl reads this from the env
# explicitly (Tcl can't see bash defaults) — see that file for the contract.
: "${LAMBDA_BUILD:=$LAMBDA_WORK}"
# Back-compat alias: LAMBDA_SCRATCH was the v0.1-v0.3 name; some launcher logs
# and chamber-diagnose strings still reference it. Aliased to LAMBDA_WORK so any
# straggler "$LAMBDA_SCRATCH/..." path resolves correctly during the migration.
: "${LAMBDA_SCRATCH:=$LAMBDA_WORK}"

# ---- Chamber SGE queue -----------------------------------------------------
: "${LAMBDA_QUEUE:=normal.q}"

# ---- Cadence tool module pins ---------------------------------------------
# Three-level leaves matching the INSTALLED versions per the 2026-06-06 debug
# session on compute node ip-10-2-6-68. Two-level specs (e.g. innovus/251) are
# valid — Environment Modules resolves to the default leaf — but pinning the
# leaf gives (a) reproducibility against silent default drift, and (b) MATCHED
# RELEASE FAMILY across Genus+Innovus, which share database format within a
# family. Newest installed Genus is 21.18.000; Innovus must match (innovus/251
# default = 25.14 → cross-version handoff, supported but ugly).
# Override per-user in ~/.longhorn/lambda.env. lambda-diagnose verifies.
: "${STRATUS_MODULE:=stratus/2201/22.01.009}"            # matches /apps/STRATUS2201
: "${XCELIUM_MODULE:=xcelium/2403/24.03.005}"            # matches /apps/XCELIUM2403
: "${GENUS_MODULE:=genus/211/21.18.000}"                 # matches /apps/GENUS211
: "${INNOVUS_MODULE:=innovus/211/21.18.000}"             # matches /apps/INNOVUS211 (same family as Genus)
: "${VERISIUM_MODULE:=verisiumdebug/2403/24.03.001}"     # primary; fallback to simvision (ships in XCELIUM2403)
: "${PEGASUS_MODULE:=pegasus/232/23.24.000}"             # observed compute-node payload (catalog also has 251)
: "${SSV_MODULE:=ssv/251/25.12.000}"                     # Tempus/Voltus/Quantus signoff (deferred use)
: "${VIRTUOSO_MODULE:=}"                                 # ic/icadv/icadvm present; pin when needed
: "${LAMBDA_PDK_MODULE:=}"                               # no TSMC N16FFC on chamber — pending PDK delivery

# ---- Lambda block list (canonical) ----------------------------------------
# Source of truth for which block names lambda-* launchers accept.
# Order matches src/README.md HLS build order (long poles first).
# Assigned unconditionally AFTER the lambda.env source: project invariant,
# not overridable per-user.
# shellcheck disable=SC2034  # consumed by the sourcing launchers
LAMBDA_BLOCKS=(mate kce vecu tiu msc lsu hif)

# ---- Bootstrap module() in non-interactive bash ----------------------------
# Chamber compute nodes run csh interactively; the Modules system is set up
# at login via shell rc. When we invoke a launcher from a script (or via
# `qsub`), bashrc is not sourced and `module` is not in scope. Try common
# init paths; ignore failure (lambda-diagnose / chamber-diagnose report it).
#
# Per-chamber path varies. If none of the fallbacks match this chamber,
# discover the correct path and set LAMBDA_MODULE_INIT in
# ~/.longhorn/lambda.env:
#
#   bash $ find / -name 'modulecmd' -type f 2>/dev/null | head -3
#   bash $ find / -name '*.sh' -path '*module*init*' 2>/dev/null | head -3
#   csh  $ which modulecmd; echo $MODULEPATH
#
# Then in ~/.longhorn/lambda.env:
#   export LAMBDA_MODULE_INIT=/the/path/to/init/bash
if ! type module >/dev/null 2>&1; then
    if [[ -n "${LAMBDA_MODULE_INIT:-}" ]] && [[ -f "$LAMBDA_MODULE_INIT" ]]; then
        # Per-user override always wins
        # shellcheck disable=SC1090
        source "$LAMBDA_MODULE_INIT"
    elif [[ -n "${MODULESHOME:-}" ]] && [[ -f "${MODULESHOME}/init/bash" ]]; then
        # Standard Environment Modules layout: $MODULESHOME/init/<shell>.
        # MODULESHOME is usually inherited from the parent csh shell (set by
        # /etc/csh.cshrc or equivalent). This auto-detects any chamber that
        # exports it. Verified on ae03ut01 (UT/Cadence): MODULESHOME =
        # /apps/modules-v3.2.6a-64bit/Modules; init/bash works.
        # shellcheck disable=SC1090
        source "${MODULESHOME}/init/bash"
    else
        # Last-resort static fallback list (covers chambers that don't export
        # MODULESHOME). If none match, set LAMBDA_MODULE_INIT in
        # ~/.longhorn/lambda.env per the discovery commands in chamber-diagnose.
        for _init in \
            /etc/profile.d/modules.sh \
            /etc/profile.d/lmod.sh \
            /etc/profile.d/cadence.sh \
            /etc/profile.d/cad.sh \
            /usr/share/Modules/init/bash \
            /usr/share/lmod/lmod/init/bash \
            /usr/share/modules/init/bash \
            /apps/modules-v3.2.6a-64bit/Modules/init/bash \
            /apps/hosted/Modules/init/bash \
            /apps/hosted/modules/init/bash \
            /apps/Modules/default/init/bash \
            /apps/Modules/init/bash \
            /apps/modules/init/bash \
            /grid/common/pkgs/Modules/init/bash \
            /grid/common/pkgs/Modules/default/init/bash \
            /grid/common/pkgs/lmod/lmod/init/bash \
            /grid/common/pkgs/modules/init/bash \
            /opt/modules/init/bash \
            /opt/Modules/init/bash \
            /cad/scripts/modules.sh \
            /cad/Modules/init/bash; do
            if [[ -f "$_init" ]]; then
                # shellcheck disable=SC1090
                source "$_init"
                break
            fi
        done
        unset _init
    fi
fi

# ---- Ensure work area + logs exist; fall back to /tmp if not writable -----
# Home is the primary target (NFS, cross-node, persistent). Fallback to LAMBDA_FAST
# (/tmp/$USER-lambda) only if home is unavailable — wiped on reboot, but better
# than failing. Per-user override is still possible via ~/.longhorn/lambda.env.
mkdir -p "$LAMBDA_WORK" "$LAMBDA_LOGS" 2>/dev/null || true
if [[ ! -w "$LAMBDA_WORK" ]] || [[ ! -d "$LAMBDA_WORK" ]]; then
    LAMBDA_WORK="$LAMBDA_FAST"
    LAMBDA_LOGS="$LAMBDA_WORK/logs"
    LAMBDA_BUILD="$LAMBDA_WORK"
    LAMBDA_SCRATCH="$LAMBDA_WORK"
    mkdir -p "$LAMBDA_WORK" "$LAMBDA_LOGS" 2>/dev/null
fi
# LAMBDA_ROOT exported too (v0.4.2, M7): flow Tcl reads it from the process
# env — e.g. src/blocks/mate/genus/synth.tcl uses $::env(LAMBDA_ROOT).
export LAMBDA_ROOT LAMBDA_WORK LAMBDA_BUILD LAMBDA_LOGS LAMBDA_SCRATCH LAMBDA_FAST
