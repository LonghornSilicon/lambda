#!/usr/bin/env bash
# ============================================================================
# lambda-detach.sh
# ----------------------------------------------------------------------------
# Single-function library: gui_detach.
#
# Encapsulates the "launch a GUI tool on a compute node, fork it cleanly,
# log its output to scratch, leave a PID file" pattern. Replaces the csh-only
# `<tool> < /dev/null >& /tmp/tool.log &` idiom from the chamber reference.
#
# Source this file (after lambda-env.sh has run) and call:
#     gui_detach <tag> <tool> [args...]
#
# Example:
#     gui_detach stratus.mate.gui stratus_ide -project project.tcl
#
# Side effects:
#   - Writes "$LAMBDA_LOGS/<tag>.<UTC-timestamp>.log"
#   - Writes "/tmp/lambda-<tag>.<pid>.pid" containing the child PID
#   - Prints PID + log path + watch hint to stdout
# ============================================================================

# shellcheck disable=SC2034  # consumed by sourcing scripts
LAMBDA_DETACH_LOADED=1

gui_detach() {
    local tag="${1:?gui_detach: missing tag}"; shift
    local tool="${1:?gui_detach: missing tool}"; shift

    : "${LAMBDA_LOGS:?LAMBDA_LOGS not set; source tools/lib/lambda-env.sh first}"

    # Preflight: tool must exist
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: '$tool' not in PATH. Module not loaded?" >&2
        echo "Hint:  module list   (check what's loaded)" >&2
        echo "       lambda-diagnose   (full chamber probe)" >&2
        return 1
    fi

    # Preflight: X11 must be available (every GUI tool needs DISPLAY)
    if [[ -z "${DISPLAY:-}" ]]; then
        echo "ERROR: \$DISPLAY is not set; X11 forwarding required for GUI tools." >&2
        echo "Hint:  ssh -X (or -Y) into the chamber, then re-qsh." >&2
        echo "       Test forwarding:  xclock &" >&2
        return 1
    fi

    local timestamp
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    local log_file="$LAMBDA_LOGS/${tag}.${timestamp}.log"
    local pid_file="/tmp/lambda-${tag}.$$.pid"

    # Defensive: ensure the log dir exists at launch time. /rscratch is
    # node-local on some chambers, so install.sh's mkdir on the utility node
    # doesn't propagate to compute nodes. If we still can't write, fall back
    # to /tmp so the GUI actually starts (with a warning).
    mkdir -p "$(dirname "$log_file")" 2>/dev/null
    if [[ ! -w "$(dirname "$log_file")" ]]; then
        echo "WARN: $(dirname "$log_file") not writable; falling back to /tmp for logs" >&2
        log_file="/tmp/$(basename "$log_file")"
    fi

    # nohup + redirect: stdin from /dev/null avoids tty-output suspension,
    # &> sends both stdout and stderr to the log, & backgrounds the process.
    # ${1+"$@"} (v0.4.2, M2): a no-args `gui_detach <tag> <tool>` under bash
    # 4.2 `set -u` would crash on the empty "$@" expansion; the idiom is safe.
    nohup "$tool" ${1+"$@"} </dev/null &>"$log_file" &
    local pid=$!

    echo "$pid" > "$pid_file"

    # ${1+$*} (v0.4.2, M2): "$*" suffers the same bash-4.2 set -u empty-args
    # abort as "$@" — and crashing HERE would kill the caller AFTER the tool
    # was already forked (orphaned GUI, no PID echoed). No quotes inside the
    # heredoc: they'd be literal characters there.
    cat <<EOF
Launched: $tool ${1+$*}
  PID:    $pid
  log:    $log_file
  pidfile:$pid_file

To watch progress: tail -f $log_file
To stop:           kill $pid
EOF
}
