#!/usr/bin/env bash
#
# clear_capture_data.sh - empty the capture data directories on one Pi.
#
# Two directories can hold captures:
#   /media/pi/HIKSEMI/data          the SSD, where captures belong
#   /home/pi/PiCameraArray/data     the SD card fallback, which fills in
#                                   under an hour and should stay empty
#
# ---------------------------------------------------------------------------
#  Runs in DRY-RUN mode by default and deletes nothing until CONFIRM=YES.
# ---------------------------------------------------------------------------
#
# Usage on the Pi:
#     sudo bash clear_capture_data.sh                       # report only
#     sudo CONFIRM=YES bash clear_capture_data.sh           # clear the SSD
#     sudo CONFIRM=YES TARGET=both bash clear_capture_data.sh
#
# Environment overrides:
#     TARGET   ssd | sd | both        (default: ssd)
#     CONFIRM  "YES" to actually delete. Anything else = dry run.
#     MOUNT    SSD mount point        (default: /media/pi/HIKSEMI)
#     SD_DIR   SD fallback directory  (default: /home/pi/PiCameraArray/data)
#
set -uo pipefail

MOUNT="${MOUNT:-/media/pi/HIKSEMI}"
SSD_DIR="${MOUNT}/data"
SD_DIR="${SD_DIR:-/home/pi/PiCameraArray/data}"
TARGET="${TARGET:-ssd}"
CONFIRM="${CONFIRM:-no}"
SERVICE="picamera-capture.service"

HOST="$(hostname)"
log()  { echo "[$HOST] $*"; }
warn() { echo "[$HOST] WARNING: $*" >&2; }
die()  { echo "[$HOST] ERROR: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "must run as root (use sudo)"

case "$TARGET" in
    ssd|sd|both) ;;
    *) die "TARGET must be ssd, sd or both (got '$TARGET')" ;;
esac

# Guard against a typo turning this into rm -rf on something important.
sane_path() {
    local p="$1"
    [[ "$p" == /* ]]                  || return 1   # absolute
    [[ "${#p}" -ge 12 ]]              || return 1   # not / or /data
    [[ "$(basename "$p")" == "data" ]] || return 1  # always a .../data dir
    return 0
}

usage_of() { du -sh "$1" 2>/dev/null | cut -f1; }
count_of() { find "$1" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l; }

service_stopped=0
stop_service_once() {
    [[ "$service_stopped" -eq 0 ]] || return 0
    log "stopping $SERVICE"
    systemctl stop "$SERVICE" 2>/dev/null || true
    service_stopped=1
}

clear_dir() {
    local dir="$1" kind="$2" require_mount="$3"

    if [[ "$require_mount" == "yes" ]] && ! mountpoint -q "$MOUNT"; then
        # Without this check, "clearing the SSD" would delete a same-named
        # directory sitting on the root filesystem instead.
        warn "$MOUNT is not mounted - skipping $kind"
        return 0
    fi

    if [[ ! -d "$dir" ]]; then
        log "$kind: $dir does not exist - nothing to do"
        return 0
    fi

    sane_path "$dir" || die "refusing to clear suspicious path: $dir"

    local size count
    size="$(usage_of "$dir")"
    count="$(count_of "$dir")"
    log "$kind: $dir  ${size:-0}  in ${count} entries"

    if [[ "$CONFIRM" != "YES" ]]; then
        return 0
    fi

    if [[ "$count" -eq 0 ]]; then
        log "$kind: already empty"
        return 0
    fi

    stop_service_once
    # -delete implies -depth, so non-empty directories are removed correctly.
    find "$dir" -mindepth 1 -delete
    mkdir -p "$dir"
    chown "${OWNER:-pi}":"${OWNER:-pi}" "$dir"
    log "$kind: cleared ${size} -> $(usage_of "$dir")"
}

log "target=$TARGET  confirm=$CONFIRM"

if [[ "$TARGET" == "ssd" || "$TARGET" == "both" ]]; then
    clear_dir "$SSD_DIR" "SSD" yes
fi
if [[ "$TARGET" == "sd" || "$TARGET" == "both" ]]; then
    clear_dir "$SD_DIR" "SD-fallback" no
fi

if [[ "$CONFIRM" != "YES" ]]; then
    log "DRY RUN - nothing was deleted. Re-run with CONFIRM=YES to clear."
    exit 0
fi

if [[ "$service_stopped" -eq 1 ]]; then
    log "restarting $SERVICE"
    # Restarting also makes the agent resolve a fresh session directory,
    # which is what moves it off the SD card once the SSD is mounted.
    systemctl start "$SERVICE" 2>/dev/null || warn "could not start $SERVICE"
fi

if mountpoint -q "$MOUNT"; then
    log "SSD free: $(df -h --output=avail "$MOUNT" | tail -1 | tr -d ' ')"
fi
log "done"
