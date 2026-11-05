#!/usr/bin/env bash
# =============================================================================
# ingest_ssd_all.sh - survey and ingest EVERY Pi SSD attached to this machine.
#
# ingest_ssd.sh already handles one disk, including the local case:
#
#     construct/ingest_ssd.sh --device /dev/sdc1 -y --all-sessions
#
# and push_ssd.ps1 is only its Windows wrapper.  What is missing when several
# SSDs are carried over at once - which is what happens after a run where a
# few Pis never reached the network - is doing that for all of them without
# typing the same command five times with a different letter, and, before any
# copying, being told WHICH camera each disk belongs to and whether it
# actually fills a gap in the dataset.
#
# So this script is a survey first and an ingest second:
#
#   1  find every partition that looks like a Pi SSD (ext4, ~233 GB)
#   2  mount each one READ-ONLY on a temporary directory
#   3  read the camera name and the sessions from the file names
#   4  compare against what the dataset already holds, and say how many
#      bursts each disk would ADD
#   5  only with --go, hand each disk to ingest_ssd.sh
#   6  unmount, always, including on interrupt
#
# The disk is never written to and never deleted from - mounting is read-only
# and the copy is one-way.  Mounting needs root, so the script asks sudo for
# that step alone; the copy itself runs as the invoking user, so the files it
# creates belong to that user and not to root.
#
# Usage:
#   construct/ingest_ssd_all.sh                 # survey only, copies nothing
#   construct/ingest_ssd_all.sh --go            # survey, then ingest each disk
#   construct/ingest_ssd_all.sh --devices /dev/sdc1,/dev/sdd1 --go
#   construct/ingest_ssd_all.sh --go --tmux     # survives an SSH disconnect
#   sudo construct/ingest_ssd_all.sh --umount   # release them before unplugging
#
# Options:
#   --umount             unmount every /mnt/picam_ssd_* this script made and
#                        remove the empty directories, so the disks can be
#                        unplugged.  Refuses while anything still has a file
#                        open on them.
#   --go                 actually ingest (default: survey only)
#   --tmux               run the copy in a detached tmux session, so an SSH
#                        disconnect does not kill it.  The disks are mounted
#                        FIRST, in the foreground, because a detached pane has
#                        nobody to answer sudo's password prompt; the copy
#                        itself needs no privilege and is what gets detached.
#   --devices LIST       comma-separated partitions instead of auto-detection
#   --data-root DIR      dataset root (default /data/CameraArray)
#   --session NAME       force the collection to merge into
#   --keep-mounted       leave the disks mounted afterwards
#   --checksum           pass --checksum through to ingest_ssd.sh
#   --dry-run            pass --dry-run through to ingest_ssd.sh
#   -h | --help
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$HERE/ingest_ssd.sh"
DATA_ROOT="/data/CameraArray"
DEVICES=""
SESSION=""
GO=0
KEEP=0
USE_TMUX=0
UMOUNT=0
TMUX_SESSION="picam_ingest_ssd"
PASSTHRU=()
MNT_BASE="/mnt/picam_ssd"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --go)           GO=1; shift ;;
        --umount|--unmount) UMOUNT=1; shift ;;
        --tmux)         USE_TMUX=1; shift ;;
        --devices)      DEVICES="$2"; shift 2 ;;
        --data-root)    DATA_ROOT="$2"; shift 2 ;;
        --session)      SESSION="$2"; shift 2 ;;
        --keep-mounted) KEEP=1; shift ;;
        --checksum|--dry-run) PASSTHRU+=("$1"); shift ;;
        -h|--help)      sed -n '2,48p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -x "$ENGINE" || -f "$ENGINE" ]] || { echo "ingest_ssd.sh not found next to this script" >&2; exit 1; }

# The copy must not run as root, or every file it creates would be root-owned
# in a dataset that belongs to a user.  Only the mount needs privilege.
RUN_AS=""
if [[ ${EUID} -eq 0 && -n "${SUDO_USER:-}" ]]; then
    RUN_AS="sudo -u ${SUDO_USER}"
    echo "[all] running the copy as ${SUDO_USER}, not root"
fi

log() { echo "[all] $*"; }

# ------------------------------------------------------------------- umount --
# Pulling a disk out from under a live mount is how a filesystem gets damaged,
# so this checks for open files first and never forces anything.  The mounts
# are read-only, so there is nothing to flush, but sync costs nothing and makes
# the guarantee obvious.
if (( UMOUNT )); then
    shopt -s nullglob
    dirs=(/mnt/picam_ssd_*)
    shopt -u nullglob
    if (( ${#dirs[@]} == 0 )); then
        log "no /mnt/picam_ssd_* directories; nothing to release"
        exit 0
    fi
    sync
    rc=0
    for m in "${dirs[@]}"; do
        if ! mountpoint -q "$m"; then
            rmdir "$m" 2>/dev/null && log "removed the empty directory $m" \
                                   || log "$m is not a mount point (left as it is)"
            continue
        fi
        busy="$(lsof +D "$m" 2>/dev/null | tail -n +2 | wc -l)"
        if (( busy > 0 )); then
            log "$m is in use by $busy open file(s) - NOT unmounting"
            lsof +D "$m" 2>/dev/null | tail -n +2 | awk '{print "        " $1, $2, $NF}' | head -5
            rc=1
            continue
        fi
        src="$(findmnt -n -o SOURCE --target "$m" 2>/dev/null)"
        if umount "$m" 2>/dev/null || sudo umount "$m"; then
            log "unmounted $m  (was $src)"
            rmdir "$m" 2>/dev/null
        else
            log "could not unmount $m"
            rc=1
        fi
    done
    if (( rc == 0 )); then
        log "all released - the disks can be unplugged now"
    else
        log "some mounts are still held; see above"
    fi
    exit $rc
fi

# ------------------------------------------------------------------ discovery
if [[ -n "$DEVICES" ]]; then
    IFS=',' read -r -a PARTS <<< "$DEVICES"
else
    # Mounted or not: a disk the user mounted beforehand is exactly the case
    # that needs no privilege at all, so it must not be filtered out here.
    mapfile -t PARTS < <(
        lsblk -rno NAME,FSTYPE,SIZE,LABEL |
        awk '$2 == "ext4" && $3 ~ /^2[0-9][0-9](\.[0-9]+)?G$/ { print "/dev/" $1 }')
fi
if [[ ${#PARTS[@]} -eq 0 ]]; then
    log "no unmounted ext4 partition of about 233 GB found."
    log "attached filesystems:"
    bash "$ENGINE" --list-disks
    exit 1
fi
log "candidate SSD partitions: ${PARTS[*]}"

MOUNTED=()
cleanup() {
    [[ $KEEP -eq 1 ]] && { log "leaving the disks mounted (--keep-mounted)"; return; }
    for m in "${MOUNTED[@]:-}"; do
        [[ -n "$m" ]] && mountpoint -q "$m" && sudo umount "$m" && log "unmounted $m"
    done
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------- survey pass
declare -A DISK_CAM DISK_MNT DISK_INFO DISK_UUID
echo
printf '%-12s %-8s %-22s %8s %10s   %s\n' DEVICE CAMERA SESSION FILES "IN DATASET" ADDS
printf '%s\n' "-------------------------------------------------------------------------------------"
need_priv=0
for dev in "${PARTS[@]}"; do
    [[ -n "$(lsblk -no MOUNTPOINT "$dev" 2>/dev/null | head -1)" ]] || need_priv=1
done
if [[ $need_priv -eq 1 ]] && ! sudo -n true 2>/dev/null; then
    cat >&2 <<MSG

[all] Some of these disks are not mounted, and mounting needs root.  This
      shell has no way to ask for a password, so do one of the two:

      run this script from a terminal that can:
          sudo construct/ingest_ssd_all.sh$( ((GO)) && echo " --go")

      or mount them yourself, read-only, and re-run with no privilege at all:
$(for d in "${PARTS[@]}"; do
      m="/mnt/picam_ssd_$(basename "$d")"
      [[ -n "$(lsblk -no MOUNTPOINT "$d" 2>/dev/null | head -1)" ]] ||
          echo "          sudo mkdir -p $m && sudo mount -o ro $d $m"
  done)

      Nothing is ever written to the disks; -o ro is what makes that certain.
MSG
    exit 2
fi

for dev in "${PARTS[@]}"; do
    mnt="$(lsblk -no MOUNTPOINT "$dev" 2>/dev/null | head -1)"
    if [[ -z "$mnt" ]]; then
        mnt="${MNT_BASE}_$(basename "$dev")"
        sudo mkdir -p "$mnt" || { log "cannot create $mnt"; continue; }
        if ! sudo mount -o ro "$dev" "$mnt" 2>/dev/null; then
            log "could not mount $dev read-only - skipping"
            continue
        fi
        MOUNTED+=("$mnt")
    else
        log "$dev is already mounted on $mnt - using it as it is"
    fi
    DISK_MNT[$dev]="$mnt"
    DISK_UUID[$dev]="$(lsblk -no UUID "$dev" 2>/dev/null | head -1)"
    # The SSD layout is the one the Pi writes and ingest_ssd.sh reads:
    #     [data/]<YYYYmmdd_HHMMSS>/<host>_burst####_<ts>.npz
    # so the session directory carries only a timestamp and the CAMERA NAME
    # lives in the file names.  The destination is laid out the other way -
    # <collection>/<host>_<session>/ - which is why the two must not be
    # matched by directory name alone.
    src="$mnt"
    [[ -d "$mnt/data" ]] && src="$mnt/data"
    found=0
    while IFS= read -r sdir; do
        sess="$(basename "$sdir")"
        one="$(find "$sdir" -maxdepth 1 -name '*_burst*.npz' -printf '%f\n' 2>/dev/null | head -1)"
        [[ -z "$one" ]] && continue
        cam="${one%%_*}"
        n=$(find "$sdir" -maxdepth 1 -name "${cam}_burst*.npz" | wc -l)
        found=1
        DISK_CAM[$dev]="$cam"
        have=0
        target="$(find "$DATA_ROOT" -maxdepth 2 -type d -name "${cam}_${sess}" 2>/dev/null | head -1)"
        [[ -n "$target" ]] && have=$(find "$target" -maxdepth 1 -name "${cam}_burst*.npz" | wc -l)
        adds=$(( n - have )); (( adds < 0 )) && adds=0
        flag=""
        (( adds > 0 )) && flag="  <-- fills a gap"
        (( adds == 0 )) && [[ -n "$target" ]] && flag="  already complete"
        (( adds > 0 )) && [[ -z "$target" ]] && flag="  <-- NOT in the dataset at all"
        printf '%-12s %-8s %-22s %8d %10d %+8d%s\n' \
               "$dev" "$cam" "$sess" "$n" "$have" "$adds" "$flag"
        DISK_INFO[$dev]="${DISK_INFO[$dev]:-}$sess:$n:$adds "
    done < <(find "$src" -maxdepth 2 -type d -regextype posix-extended \
                  -regex '.*/[0-9]{8}_[0-9]{6}$' 2>/dev/null | sort)
    if (( found == 0 )); then
        printf '%-12s %-8s %-22s %8s %10s   %s\n' \
               "$dev" "?" "-" "-" "-" "no capture session found"
        # Say what IS there, so a layout that does not match can be diagnosed
        # without mounting everything a second time - each run costs a password.
        echo "             top of $mnt:"
        ls -1 "$mnt" 2>/dev/null | head -8 | sed 's/^/               /'
        nnpz=$(find "$mnt" -name '*.npz' 2>/dev/null | head -20000 | wc -l)
        echo "             .npz files anywhere on the disk: $nnpz"
        if (( nnpz > 0 )); then
            echo "             first few, with their paths:"
            find "$mnt" -name '*.npz' 2>/dev/null | head -3 | \
                sed "s|^$mnt|               .|"
        fi
    fi
done
echo

if [[ $GO -eq 0 ]]; then
    log "survey only. Re-run with --go to ingest the disks that fill a gap."
    exit 0
fi

# ------------------------------------------------------- detach into tmux ----
# Everything above has already run: the disks are mounted and the survey has
# been printed, so the long part - the copy - is all that is left, and it needs
# no privilege.  That is why the detach happens HERE and not at the top: sudo
# cannot ask for a password in a pane nobody is attached to.
if [[ $USE_TMUX -eq 1 && -z "${TMUX:-}" ]]; then
    command -v tmux >/dev/null 2>&1 || {
        echo "tmux not installed: sudo apt install tmux" >&2; exit 1; }
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "tmux session '$TMUX_SESSION' already exists (a run is in progress?)." >&2
        echo "  attach: tmux attach -t $TMUX_SESSION" >&2
        exit 1
    fi
    SELF="$(readlink -f "$0")"
    ARGS=(--go --keep-mounted --data-root "$DATA_ROOT"
          --devices "$(IFS=,; echo "${PARTS[*]}")")
    [[ -n "$SESSION" ]] && ARGS+=(--session "$SESSION")
    (( ${#PASSTHRU[@]} )) && ARGS+=("${PASSTHRU[@]}")
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
    LOG="$DATA_ROOT/logs/ingest_ssd_all_$RUN_ID.log"
    mkdir -p "$DATA_ROOT/logs" 2>/dev/null || true
    # The copy runs as the invoking user even when this script was started with
    # sudo, so the tmux session must belong to that user too - otherwise the
    # files it writes, and the session itself, end up owned by root.
    # Three levels of quoting - shell, tmux, and the command it runs - is how
    # a launcher like this usually breaks.  Write the inner command to a file
    # and let tmux run that instead; there is then nothing to quote.
    RUNNER="$(mktemp /tmp/picam_ingest_XXXXXX.sh)"
    {
        echo '#!/usr/bin/env bash'
        printf '%s' "$(printf '%q ' "$SELF" "${ARGS[@]}")"
        printf '2>&1 | tee -a %q\n' "$LOG"
    } > "$RUNNER"
    chmod +x "$RUNNER"
    inner="$RUNNER; echo; echo '[finished - press q]'; read -r -n1"
    if [[ -n "$RUN_AS" ]]; then
        chown "${SUDO_USER}" "$RUNNER" 2>/dev/null || true
        sudo -u "${SUDO_USER}" tmux new-session -d -s "$TMUX_SESSION" -c "$PWD" "$inner"
    else
        tmux new-session -d -s "$TMUX_SESSION" -c "$PWD" "$inner"
    fi
    echo
    log "the copy is running in tmux session '$TMUX_SESSION'; it survives an SSH disconnect."
    echo "  attach  : tmux attach -t $TMUX_SESSION      (detach again: Ctrl-b d)"
    echo "  log     : tail -f $LOG"
    echo "  disks   : left mounted read-only; when it has finished, release them with"
    for m in "${MOUNTED[@]:-}"; do
        [[ -n "$m" ]] && echo "              sudo umount $m"
    done
    KEEP=1        # the trap must not pull the disks out from under tmux
    exit 0
fi

# --------------------------------------------------------------- ingest pass
rc_all=0
for dev in "${PARTS[@]}"; do
    mnt="${DISK_MNT[$dev]:-}"
    [[ -z "$mnt" ]] && continue
    cam="${DISK_CAM[$dev]:-?}"

    # A USB disk can drop off the bus and come back under a different kernel
    # name; the old mount then survives as an empty directory and the copy
    # silently finds nothing.  That is exactly what happened to e05 on
    # 2026-09-02: /dev/sdd1 re-enumerated as /dev/sdh1 between the survey and
    # the copy.  So the disk is identified again here by its UUID, which does
    # not change, and the mount is checked for content before anything is read.
    uuid="${DISK_UUID[$dev]:-}"
    now="$(lsblk -no KNAME,UUID | awk -v u="$uuid" '$2 == u { print "/dev/" $1 }' | head -1)"
    if [[ -n "$uuid" && -n "$now" && "$now" != "$dev" ]]; then
        log "$dev has re-enumerated as $now (same UUID $uuid)"
        log "  the old mount $mnt is stale; remount and re-run:"
        log "     sudo umount $mnt; sudo mkdir -p /mnt/picam_ssd_$(basename "$now"); sudo mount -o ro $now /mnt/picam_ssd_$(basename "$now")"
        rc_all=1; continue
    fi
    if [[ -z "$(ls -A "$mnt" 2>/dev/null)" ]]; then
        log "$mnt is empty - the disk went away since the survey. Skipping $dev ($cam)."
        log "  attached now: $(lsblk -no KNAME,UUID | awk -v u="$uuid" '$2 == u { print "/dev/" $1 }' | head -1)"
        rc_all=1; continue
    fi
    log "ingesting $dev ($cam) from $mnt"
    args=(--all-sessions -y --data-root "$DATA_ROOT" "${PASSTHRU[@]:-}")
    [[ -n "$SESSION" ]] && args+=(--session "$SESSION")
    if [[ -n "$RUN_AS" ]]; then
        $RUN_AS bash "$ENGINE" "${args[@]}" "$mnt"
    else
        bash "$ENGINE" "${args[@]}" "$mnt"
    fi
    rc=$?
    (( rc != 0 )) && { log "$dev exited with $rc"; rc_all=$rc; }
done

log "re-packing so the new bursts become visible to the analysis"
if [[ -n "$RUN_AS" ]]; then
    for s in "$DATA_ROOT"/2*_*; do
        [[ -d "$s" ]] && $RUN_AS python3 "$HERE/pack_bursts.py" "$s" | tail -2
    done
else
    for s in "$DATA_ROOT"/2*_*; do
        [[ -d "$s" ]] && python3 "$HERE/pack_bursts.py" "$s" | tail -2
    done
fi
exit $rc_all
