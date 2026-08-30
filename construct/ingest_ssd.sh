#!/usr/bin/env bash
# =============================================================================
# ingest_ssd.sh - put a capture session from a Pi's SSD into the dataset,
#                 either on this machine or on a remote one over SSH.
#
# Two ways to run it:
#
#   local    the SSD is plugged into the machine that holds $DATA_ROOT
#            (Thunder itself):        ./ingest_ssd.sh -y /mnt
#
#   push     the SSD is plugged into another machine - typically the Windows
#            laptop over Tailscale - and the data is pushed to Thunder:
#            ./ingest_ssd.sh -y --target takatani@thunder /mnt
#            (on Windows this runs inside WSL; construct/push_ssd.ps1 wraps it)
#
# Either way the result is the same layout construct/get_data.sh produces, so
# a session collected over the network and one carried over by hand are
# indistinguishable afterwards:
#
#     $DATA_ROOT/<collection>/<host>_<session>/<host>_burst####_<ts>.npz
#
# It covers two situations:
#   - a Pi that never joined the run (powered off, off the network): its data
#     exists only on the SSD, so carry the disk over and ingest it;
#   - a Pi whose network transfer died half way: ingest tops the directory up,
#     copying only what is missing.
#
# Identity works like get_data.sh: a session is identified by its FILE COUNT,
# and timestamps only break ties. With one correction that matters here - a
# collection that is still filling is SHORT of the disk, not equal to it, so
# what a collection expects is what its FULLEST camera holds, and a session on
# the disk qualifies when it holds at least that many. An exact match still
# wins; otherwise the shortest session that could still be the source is used,
# and it says so. Name a collection with --session and the matching session on
# the SSD is found automatically.
#
# The source disk is only ever READ - never written to, never deleted from.
#
# Usage:
#   ./ingest_ssd.sh [options] <path>          # mount point, data dir or session dir
#   ./ingest_ssd.sh [options] --device /dev/sdc1
#
#  where the data goes:
#   --target USER@HOST     push over SSH instead of copying locally
#   --ssh-cmd CMD          ssh binary to use (default: ssh). From WSL, point
#                          this at /mnt/c/Windows/System32/OpenSSH/ssh.exe to
#                          borrow the Windows host's Tailscale connection.
#   --ssh-opt OPT          extra option for ssh; repeatable
#   --data-root DIR        default /data/CameraArray (on the target)
#   --link-root DIR        maintain the collected_data symlink here
#                          (default: $PWD/collected_data locally, off when pushing)
#
#  what gets ingested:
#   --device DEV           mount DEV read-only on a temp dir, ingest, unmount
#   --session NAME         collection folder to merge into (default: detected)
#   --source-session NAME  which session on the SSD (default: the matching one)
#   --all-sessions         ingest every session found on the SSD
#   --host eNN             override the host name parsed from the file names
#   --new-collection       allow creating a collection folder that is not there
#
#  how it runs:
#   --list                 show what is on the disk and what would happen; copy nothing
#   --list-disks           show attached disks that look like a Pi SSD, then exit
#   --checksum             compare by content, not size+mtime (slow, thorough)
#   --bwlimit KBPS         cap the transfer rate (useful over a shared uplink)
#   --dry-run              rsync --dry-run; no directory or symlink side effects
#   --progress-interval S  seconds between progress reports (default 10, 0 off)
#   -y | --yes             do not ask for confirmation before copying
#   -h | --help
#
# Requires: bash 4.3+, rsync (both ends when pushing), sudo only for --device.
# =============================================================================

# Bash only (arrays, associative arrays). Re-exec when run as `sh ingest_ssd.sh`.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -uo pipefail

# ----------------------------- configuration ---------------------------------
DATA_ROOT="${DATA_ROOT:-/data/CameraArray}"
LINK_ROOT="${LINK_ROOT:-}"
TARGET="${TARGET:-}"
SSH_CMD="${SSH_CMD:-ssh}"
SSH_OPTS=(-o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

SRC=""
DEVICE=""
COLLECTION=""
SOURCE_SESSION=""
ALL_SESSIONS=0
FORCE_HOST=""
NEW_COLLECTION=0
LIST_ONLY=0
CHECKSUM=0
BWLIMIT=""
DRY_RUN=0
ASSUME_YES=0
PROGRESS_INTERVAL=10
LIST_DISKS=0

# ------------------------------ arg parsing ----------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)          TARGET="$2"; shift 2 ;;
        --ssh-cmd)         SSH_CMD="$2"; shift 2 ;;
        --ssh-opt)         SSH_OPTS+=("$2"); shift 2 ;;
        --device)          DEVICE="$2"; shift 2 ;;
        --session)         COLLECTION="$2"; shift 2 ;;
        --source-session)  SOURCE_SESSION="$2"; shift 2 ;;
        --all-sessions)    ALL_SESSIONS=1; shift ;;
        --host)            FORCE_HOST="$2"; shift 2 ;;
        --new-collection)  NEW_COLLECTION=1; shift ;;
        --list)            LIST_ONLY=1; shift ;;
        --list-disks)      LIST_DISKS=1; shift ;;
        --checksum)        CHECKSUM=1; shift ;;
        --bwlimit)         BWLIMIT="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift ;;
        --data-root)       DATA_ROOT="$2"; shift 2 ;;
        --link-root)       LINK_ROOT="$2"; shift 2 ;;
        --progress-interval) PROGRESS_INTERVAL="$2"; shift 2 ;;
        -y|--yes)          ASSUME_YES=1; shift ;;
        -h|--help)         awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"; exit 0 ;;
        -*)                echo "Unknown option: $1" >&2; exit 2 ;;
        *)                 SRC="$1"; shift ;;
    esac
done
DATA_ROOT="${DATA_ROOT%/}"
LINK_ROOT="${LINK_ROOT%/}"
# Locally the symlink is the point of collected_data; when pushing, the path
# would be on the far machine and is only touched if asked for explicitly.
[[ -z "$TARGET" && -z "$LINK_ROOT" ]] && LINK_ROOT="$PWD/collected_data"

# --------------------------- disks worth looking at --------------------------
list_disks() {
    echo "attached filesystems (a Pi SSD is one ext4 partition, ~233 GB, labelled HIKSEMI):"
    lsblk -o NAME,LABEL,FSTYPE,SIZE,MOUNTPOINT,MODEL -P 2>/dev/null |
    while read -r line; do
        eval "$line"
        [[ -z "${FSTYPE:-}" || "$FSTYPE" == "LVM2_member" ]] && continue
        printf '  /dev/%-12s %-10s %-6s %8s  %s %s\n' \
            "$NAME" "${LABEL:--}" "$FSTYPE" "$SIZE" \
            "${MOUNTPOINT:+mounted on $MOUNTPOINT}" "${MODEL:-}"
    done
    echo
    echo "then:  $0 --device /dev/sdX1 ...      (mounts read-only by itself)"
    echo "or  :  sudo mount -o ro /dev/sdX1 /mnt  &&  $0 /mnt ..."
}
(( LIST_DISKS == 1 )) && { list_disks; exit 0; }

command -v rsync >/dev/null 2>&1 || { echo "rsync not found: sudo apt install rsync" >&2; exit 1; }

# ------------------------------- logging -------------------------------------
# Logs live next to the operator, not next to the data: when pushing from a
# laptop the far side may not even be writable by this user.
if [[ -n "$TARGET" ]]; then
    LOG_DIR="${TMPDIR:-/tmp}/picam_ingest"
else
    LOG_DIR="$DATA_ROOT/logs"
fi
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="${TMPDIR:-/tmp}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/ingest_${RUN_ID}.log"
: > "$MAIN_LOG" 2>/dev/null || MAIN_LOG=/dev/null

log() {
    local level="$1"; shift
    printf '%s [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$level" "$*" | tee -a "$MAIN_LOG"
}
report() { printf '%s\n' "$*" | tee -a "$MAIN_LOG"; }
die() { log ERROR "$*"; exit 1; }

# ------------------------- destination side (local or ssh) -------------------
# Every question about where the data should go is answered by the machine
# that holds it, through this one indirection.
dest_sh() {
    if [[ -z "$TARGET" ]]; then
        bash -c "$1"
    else
        "$SSH_CMD" "${SSH_OPTS[@]}" "$TARGET" "$1"
    fi
}

dest_label() { [[ -z "$TARGET" ]] && echo "this machine" || echo "$TARGET"; }

# One round trip: the whole state of $DATA_ROOT.
#   ROOT <root> exists|missing
#   DF   <avail_kb>
#   DIR  <collection> <host_session> <files> <bytes>
dest_snapshot_script() {
    cat <<EOF
root='$DATA_ROOT'
if [ ! -d "\$root" ]; then printf 'ROOT\t%s\tmissing\n' "\$root"; exit 0; fi
printf 'ROOT\t%s\texists\n' "\$root"
df -Pk "\$root" 2>/dev/null | awk 'NR==2 {printf "DF\t%s\n", \$4}'
for c in "\$root"/*/; do
    cn=\${c%/}; cn=\${cn##*/}
    [ "\$cn" = logs ] && continue
    for d in "\$c"e[0-9][0-9]_*/; do
        [ -d "\$d" ] || continue
        dn=\${d%/}; dn=\${dn##*/}
        find "\$d" -type f \( -name '*.npz' -o -name '*.png' \) -printf '%s\n' 2>/dev/null |
        awk -v c="\$cn" -v d="\$dn" '{n++; b+=\$1} END {printf "DIR\t%s\t%s\t%d\t%.0f\n", c, d, n+0, b+0}'
    done
done
EOF
}

dest_stats() {   # DEST_DIR -> "files bytes" as seen on the destination
    local d="$1"
    dest_sh "find '$d' -type f \\( -name '*.npz' -o -name '*.png' \\) -printf '%s\\n' 2>/dev/null |
             awk '{c++; b += \$1} END {print c+0, b+0}'"
}

# ------------------------------- helpers -------------------------------------
human_bytes() {
    awk -v b="${1:-0}" 'BEGIN {
        split("B KiB MiB GiB TiB", u, " "); i = 1
        while (b >= 1024 && i < 5) {b /= 1024; i++}
        printf (i == 1 ? "%.0f%s" : "%.1f%s"), b, u[i] }'
}

human_dur() {
    local s="${1:-0}"
    (( s < 0 )) && s=$(( -s ))
    if   (( s >= 86400 )); then printf '%dd%02dh' $(( s / 86400 )) $(( (s % 86400) / 3600 ))
    elif (( s >= 3600 ));  then printf '%dh%02dm' $(( s / 3600 ))  $(( (s % 3600) / 60 ))
    elif (( s >= 60 ));    then printf '%dm%02ds' $(( s / 60 ))    $(( s % 60 ))
    else                        printf '%ds' "$s"
    fi
}

session_epoch() {   # 20260728_073612 -> epoch seconds; 0 when unparsable
    local n="$1"
    if [[ "$n" =~ ^([0-9]{4})([0-9]{2})([0-9]{2})_([0-9]{2})([0-9]{2})([0-9]{2}) ]]; then
        date -d "${BASH_REMATCH[1]}-${BASH_REMATCH[2]}-${BASH_REMATCH[3]} ${BASH_REMATCH[4]}:${BASH_REMATCH[5]}:${BASH_REMATCH[6]}" \
             +%s 2>/dev/null || echo 0
    else
        echo 0
    fi
}

src_stats() {   # DIR -> "files bytes"  (png/npz only, the payload)
    local d="$1"
    [[ -d "$d" ]] || { echo "0 0"; return; }
    find "$d" -type f \( -name '*.png' -o -name '*.npz' \) -printf '%s\n' 2>/dev/null |
        awk '{c++; b += $1} END {print c+0, b+0}'
}

progress_bar() {
    local pct="${1:-0}" w=20 filled
    (( pct < 0 )) && pct=0
    (( pct > 100 )) && pct=100
    filled=$(( pct * w / 100 ))
    printf '[%s%s]' "$(printf '%*s' "$filled" '' | tr ' ' '#')" \
                    "$(printf '%*s' $(( w - filled )) '')"
}

# host name lives in every file name: e06_burst0001_20260728_073612.npz
detect_host() {
    local d="$1" f base host seen=""
    while IFS= read -r f; do
        base="$(basename "$f")"
        host="${base%%_*}"
        [[ "$host" =~ ^e[0-9]{2}$ ]] || continue
        [[ " $seen " == *" $host "* ]] || seen="$seen $host"
    done < <(find "$d" -maxdepth 1 -type f \( -name '*.npz' -o -name '*.png' \) 2>/dev/null | head -50)
    seen="${seen# }"
    [[ -z "$seen" ]] && return 1
    [[ "$seen" == *" "* ]] && { echo "MIXED:$seen"; return 1; }
    printf '%s\n' "$seen"
}

# ------------------------- mount the disk if asked ---------------------------
MOUNT_TMP=""
PROGRESS_PID=""
cleanup() {
    [[ -n "$PROGRESS_PID" ]] && kill "$PROGRESS_PID" 2>/dev/null
    if [[ -n "$MOUNT_TMP" ]] && mountpoint -q "$MOUNT_TMP" 2>/dev/null; then
        sync
        sudo umount "$MOUNT_TMP" 2>/dev/null && log INFO "unmounted $DEVICE"
        rmdir "$MOUNT_TMP" 2>/dev/null
    fi
}
trap cleanup EXIT
trap 'echo; log WARN "interrupted -- re-run the same command to continue"; exit 130' INT TERM

if [[ -n "$DEVICE" ]]; then
    [[ -b "$DEVICE" ]] || die "$DEVICE is not a block device (see --list-disks)"
    MOUNT_TMP="$(mktemp -d /tmp/picam_ssd.XXXXXX)"
    log INFO "mounting $DEVICE read-only on $MOUNT_TMP"
    sudo mount -o ro "$DEVICE" "$MOUNT_TMP" || die "mount failed (already mounted? see --list-disks)"
    SRC="$MOUNT_TMP"
fi

[[ -n "$SRC" ]] || { awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"; exit 2; }
[[ -d "$SRC" ]] || die "$SRC is not a directory"

# --------------------------- reach the destination ---------------------------
if [[ -n "$TARGET" ]]; then
    log INFO "destination: $TARGET (via $SSH_CMD)"
    out="$(dest_sh 'echo READY; command -v rsync >/dev/null && echo HAVE_RSYNC' 2>&1)"
    [[ "$out" == *READY* ]] || die "cannot reach $TARGET over SSH: ${out:-no response}"
    [[ "$out" == *HAVE_RSYNC* ]] || die "$TARGET has no rsync in PATH"
fi

# ---------------------- find the session dirs on the disk --------------------
# Accept the SSD root (has data/), the data dir itself, or one session dir.
SINGLE_SESSION=""
if [[ -d "$SRC/data" ]]; then
    SRC_DATA="$SRC/data"
elif [[ -n "$(compgen -G "$SRC"/*_burst*.npz 2>/dev/null)" ]]; then
    SRC_DATA="$(dirname "$SRC")"
    SINGLE_SESSION="$(basename "$SRC")"
else
    SRC_DATA="$SRC"
fi
log INFO "source: $SRC_DATA"

SESSIONS=()
if [[ -n "$SINGLE_SESSION" ]]; then
    SESSIONS=("$SINGLE_SESSION")
else
    while IFS= read -r d; do SESSIONS+=("$d"); done \
        < <(cd "$SRC_DATA" 2>/dev/null && ls -1 2>/dev/null | grep -E '^[0-9]{8}_[0-9]{6}$' | sort -r)
fi
(( ${#SESSIONS[@]} > 0 )) || die "no session directories (YYYYmmdd_HHMMSS) under $SRC_DATA"

# ------------------- what the destination already holds ----------------------
log INFO "asking $(dest_label) what it already has under $DATA_ROOT"
SNAP="$(dest_sh "$(dest_snapshot_script)")" || die "could not read $DATA_ROOT on $(dest_label)"
[[ "$SNAP" == *"	exists"* ]] || {
    (( NEW_COLLECTION == 1 )) || die "$DATA_ROOT does not exist on $(dest_label)"
    log WARN "$DATA_ROOT does not exist on $(dest_label); it will be created"
}

declare -A DEST_FILES DEST_BYTES     # keyed "collection/host_session"
declare -A COLL_SEEN
DEST_AVAIL_KB=0
while IFS=$'\t' read -r key a b c d; do
    case "$key" in
        DF)  DEST_AVAIL_KB="${a:-0}" ;;
        DIR) DEST_FILES["$a/$b"]="${c:-0}"; DEST_BYTES["$a/$b"]="${d:-0}"; COLL_SEEN["$a"]=1 ;;
    esac
done <<< "$SNAP"

collection_max() {     # what the most complete camera in a collection holds
    # This, not the modal count, is the collection's session length. The whole
    # reason this script exists is that transfers stop half way, so most of a
    # collection can be short - 20260729_053037 held 14 to 33 files per camera
    # while the disk had the entire run - and the modal count of a collection
    # that is still filling says nothing about how long the session was. A
    # camera that finished does say, and no camera can hold MORE files than
    # the session it came from.
    local c="$1" k
    for k in "${!DEST_FILES[@]}"; do
        [[ "${k%%/*}" == "$c" ]] && printf '%s\n' "${DEST_FILES[$k]}"
    done | sort -n | tail -1
}
collections() { printf '%s\n' "${!COLL_SEEN[@]}" | sort -r; }

# ------------------------------ inventory ------------------------------------
declare -A S_COUNT S_BYTES S_HOST
report ""
report "=========================== SSD INVENTORY ====================================="
report "source     : $SRC_DATA"
report "destination: $DATA_ROOT on $(dest_label)"
report ""
report "$(printf '  %-16s %-5s %8s %10s  %-19s %s' \
         SESSION HOST FILES SIZE 'START' 'ALREADY AT THE DESTINATION')"
report "-------------------------------------------------------------------------------"
for s in "${SESSIONS[@]}"; do
    read -r c b <<< "$(src_stats "$SRC_DATA/$s")"
    S_COUNT[$s]="$c"; S_BYTES[$s]="$b"
    h="$(detect_host "$SRC_DATA/$s")" || h="?"
    if [[ -n "$FORCE_HOST" && "$FORCE_HOST" != "$h" ]]; then
        log WARN "$s : files are named '$h' but --host says '$FORCE_HOST'; using '$FORCE_HOST'"
        h="$FORCE_HOST"
    fi
    S_HOST[$s]="$h"
    here="-"
    for cl in $(collections); do
        if [[ -n "${DEST_FILES[$cl/${h}_${s}]:-}" ]]; then
            here="$cl (${DEST_FILES[$cl/${h}_${s}]}/$c files)"
            break
        fi
    done
    report "$(printf '  %-16s %-5s %8s %10s  %-19s %s' \
             "$s" "$h" "$c" "$(human_bytes "$b")" \
             "$(date -d "@$(session_epoch "$s")" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)" "$here")"
done
report "-------------------------------------------------------------------------------"
report ""

# --------------------- decide what goes where --------------------------------
PLAN=()
plan_add() { PLAN+=("$1"$'\t'"${S_HOST[$1]}"$'\t'"$2"); }

collection_for() {   # SESSION -> the folder it belongs in, or "" when unclear
    # Every diagnostic here goes to stderr: this function's STDOUT is its
    # return value, captured by the caller, so a log line on stdout would end
    # up inside a directory name.
    #
    # Three rules, strongest first.
    #   1 the collection already holds a directory for this host and session.
    #     Nothing beats that, and it is what tops up an interrupted transfer.
    #   2 a camera in the collection finished with exactly this many files.
    #     Two different runs almost never agree on a file count.
    #   3 the collection is merely COMPATIBLE: every camera in it is short of
    #     this session, which is what a half-transferred collection looks
    #     like. Only decisive when exactly one collection qualifies.
    # Assigned on separate lines on purpose: `local a=$1 b=${X[$a]}` expands
    # every argument before the builtin runs, so b would read whatever global
    # `a` happened to hold. It works today only because the caller's loop
    # variable is also called s.
    local s="$1"
    local h="${S_HOST[$s]}" c="${S_COUNT[$s]}" cl m exact=() compat=()
    for cl in $(collections); do
        [[ -n "${DEST_FILES[$cl/${h}_${s}]:-}" ]] && { printf '%s\n' "$cl"; return 0; }
    done
    for cl in $(collections); do
        m="$(collection_max "$cl")"; m="${m:-0}"
        if (( m == c )); then exact+=("$cl")
        elif (( m < c )); then compat+=("$cl")
        fi
    done
    (( ${#exact[@]} == 1 )) && { printf '%s\n' "${exact[0]}"; return 0; }
    if (( ${#exact[@]} > 1 )); then
        log WARN "$s : $c files matches several collections exactly: ${exact[*]}" >&2
        return 1
    fi
    if (( ${#compat[@]} == 1 )); then
        log INFO "$s : nothing at the destination has $c files yet; '${compat[0]}' is the" >&2
        log INFO "  only collection short enough to be this session (fullest camera: $(collection_max "${compat[0]}"))" >&2
        printf '%s\n' "${compat[0]}"
        return 0
    fi
    if (( ${#compat[@]} > 1 )); then
        log WARN "$s : $c files could top up any of: ${compat[*]} -- name one with --session" >&2
        return 1
    fi
    [[ -n "${COLL_SEEN[$s]:-}" ]] && { printf '%s\n' "$s"; return 0; }
    return 1
}

if [[ -n "$COLLECTION" ]]; then
    if [[ -z "${COLL_SEEN[$COLLECTION]:-}" ]]; then
        (( NEW_COLLECTION == 1 )) || die "collection '$COLLECTION' does not exist under $DATA_ROOT on $(dest_label) (use --new-collection)"
        log INFO "collection '$COLLECTION' will be created"
    fi
    want="$(collection_max "$COLLECTION")"; want="${want:-0}"
    if [[ -n "$SOURCE_SESSION" ]]; then
        if (( want > 0 )) && (( ${S_COUNT[$SOURCE_SESSION]:-0} < want )); then
            log WARN "'$SOURCE_SESSION' holds ${S_COUNT[$SOURCE_SESSION]:-0} file(s) but a camera in '$COLLECTION' already has $want --"
            log WARN "  the disk cannot be the source of a longer run, so these are almost certainly"
            log WARN "  DIFFERENT sessions; continuing because you named it explicitly"
        fi
        plan_add "$SOURCE_SESSION" "$COLLECTION"
    elif (( ALL_SESSIONS == 1 )); then
        for s in "${SESSIONS[@]}"; do plan_add "$s" "$COLLECTION"; done
    else
        (( want > 0 )) || die "collection '$COLLECTION' holds no data yet -- name the session with --source-session"
        # A collection that is still filling is SHORT of the disk, not equal to
        # it, so the disk's session has to hold at least what the fullest
        # camera holds. An exact match is the confident case and wins; failing
        # that, the shortest session that could still be the source wins, and
        # the timestamp only breaks ties - the same order get_data.sh uses.
        best=""; bestgap=-1; bestcount=-1; exact=0
        ref="$(session_epoch "$COLLECTION")"
        for s in "${SESSIONS[@]}"; do
            c="${S_COUNT[$s]}"
            (( c >= want )) || continue
            ep="$(session_epoch "$s")"
            gap=$(( ep > ref ? ep - ref : ref - ep ))
            if (( c == want )); then
                (( exact == 1 )) || { best=""; bestgap=-1; }
                exact=1
            elif (( exact == 1 )); then
                continue
            fi
            if (( bestcount < 0 || c < bestcount || (c == bestcount && gap < bestgap) )); then
                best="$s"; bestgap="$gap"; bestcount="$c"
            fi
        done
        if [[ -z "$best" ]]; then
            die "no session on this disk holds at least $want files (the fullest camera in '$COLLECTION') -- use --source-session NAME or --all-sessions"
        fi
        if (( exact == 1 )); then
            log INFO "matched by file count: '$best' ($want files) -> collection '$COLLECTION'"
        else
            log INFO "'$COLLECTION' is still filling -- its fullest camera has $want file(s)."
            log INFO "  '$best' ($bestcount files) is the shortest session on the disk that could be its source"
            log INFO "  -> collection '$COLLECTION'. Use --source-session NAME if that is the wrong one."
        fi
        plan_add "$best" "$COLLECTION"
    fi
else
    for s in "${SESSIONS[@]}"; do
        [[ -n "$SOURCE_SESSION" && "$s" != "$SOURCE_SESSION" ]] && continue
        (( ALL_SESSIONS == 0 && ${#PLAN[@]} > 0 )) && break
        (( ${S_COUNT[$s]} > 0 )) || continue
        if cl="$(collection_for "$s")"; then
            plan_add "$s" "$cl"
        elif (( NEW_COLLECTION == 1 )); then
            log WARN "$s : no matching collection; starting a new one named '$s'"
            plan_add "$s" "$s"
        else
            log WARN "$s : no matching collection on $(dest_label) (--session NAME or --new-collection)"
        fi
    done
fi
(( ${#PLAN[@]} > 0 )) || die "nothing to ingest -- name the target with --session, or use --all-sessions / --new-collection"

# ------------------------------- the plan ------------------------------------
report "PLAN"
TOTAL_MISSING=0
for p in "${PLAN[@]}"; do
    IFS=$'\t' read -r s h cl <<< "$p"
    dest="$DATA_ROOT/$cl/${h}_${s}"
    dc="${DEST_FILES[$cl/${h}_${s}]:-0}"; db="${DEST_BYTES[$cl/${h}_${s}]:-0}"
    TOTAL_MISSING=$(( TOTAL_MISSING + ${S_BYTES[$s]} - db ))
    report "  $h  $s  ${S_COUNT[$s]} files $(human_bytes "${S_BYTES[$s]}")"
    report "      -> $(dest_label):$dest"
    report "      already there: $dc file(s) $(human_bytes "$db")   to copy: ~$(human_bytes $(( ${S_BYTES[$s]} - db )))"
done
report "  free space on $(dest_label): $(human_bytes $(( DEST_AVAIL_KB * 1024 ))) at $DATA_ROOT"
(( DEST_AVAIL_KB * 1024 < TOTAL_MISSING )) && \
    log WARN "free space is below what this ingest needs (~$(human_bytes "$TOTAL_MISSING"))"
report ""

(( LIST_ONLY == 1 )) && { log INFO "--list: nothing was copied"; exit 0; }

if (( ASSUME_YES == 0 )); then
    if [[ -t 0 ]]; then
        read -r -p "Copy as planned? [y/N] " ans
        [[ "$ans" =~ ^[yY] ]] || { log INFO "aborted by the user"; exit 1; }
    else
        die "not an interactive terminal: re-run with -y once the plan above looks right"
    fi
fi

# ------------------------------- progress ------------------------------------
PROGRESS_FILE="$LOG_DIR/ingest_${RUN_ID}.progress"

progress_monitor() {   # DEST EXPECTED_FILES EXPECTED_BYTES LABEL
    local dest="$1" expf="$2" expb="$3" label="$4"
    local start_t; start_t="$(date +%s)"
    local prev_b=0 prev_t="$start_t" ema=0 first=1 f b now dt rate pct eta line
    read -r _ prev_b <<< "$(dest_stats "$dest")"
    while :; do
        sleep "$PROGRESS_INTERVAL"
        read -r f b <<< "$(dest_stats "$dest")"
        f="${f:-0}"; b="${b:-0}"
        now="$(date +%s)"; dt=$(( now - prev_t )); (( dt <= 0 )) && dt=1
        rate=$(( (b - prev_b) / dt )); (( rate < 0 )) && rate=0
        if (( first == 1 )); then ema="$rate"; first=0
        else ema=$(( (ema * 7 + rate * 3) / 10 )); fi
        pct=0; (( expb > 0 )) && pct=$(( b * 100 / expb ))
        eta=""
        (( ema > 0 && expb > b )) && eta="  ETA $(human_dur $(( (expb - b) / ema )))"
        line="$(printf '%s  %3s%% %s  %s/%s  %s/%s files  %s/s%s' \
                "$label" "$pct" "$(progress_bar "$pct")" \
                "$(human_bytes "$b")" "$(human_bytes "$expb")" \
                "$f" "$expf" "$(human_bytes "$ema")" "$eta")"
        printf '%s\nupdated %s   elapsed %s\n' \
            "$line" "$(date '+%Y-%m-%d %H:%M:%S')" "$(human_dur $(( now - start_t )))" \
            > "$PROGRESS_FILE" 2>/dev/null
        log INFO "$line"
        prev_b="$b"; prev_t="$now"
    done
}

# ------------------------------- the copy ------------------------------------
log INFO "==============================================================="
log INFO "ingest run=$RUN_ID  $SRC_DATA -> $(dest_label):$DATA_ROOT"
(( PROGRESS_INTERVAL > 0 && DRY_RUN == 0 )) && \
    log INFO "progress every ${PROGRESS_INTERVAL}s; live line: $PROGRESS_FILE"

OK=0 SHORT=0 FAILED=0
for p in "${PLAN[@]}"; do
    IFS=$'\t' read -r s h cl <<< "$p"
    src="$SRC_DATA/$s"
    dest="$DATA_ROOT/$cl/${h}_${s}"

    log INFO "$h : $s -> $(dest_label):$dest (${S_COUNT[$s]} files, $(human_bytes "${S_BYTES[$s]}"))"
    if (( DRY_RUN == 0 )); then
        dest_sh "mkdir -p '$dest'" || { log ERROR "$h : cannot create $dest on $(dest_label)"; FAILED=$((FAILED+1)); continue; }
    fi

    opts=(-a --partial --info=stats1 --human-readable)
    (( CHECKSUM == 1 )) && opts+=(--checksum)
    (( DRY_RUN == 1 )) && opts+=(--dry-run)
    [[ -n "$BWLIMIT" ]] && opts+=(--bwlimit="$BWLIMIT")

    if (( PROGRESS_INTERVAL > 0 && DRY_RUN == 0 )); then
        progress_monitor "$dest" "${S_COUNT[$s]}" "${S_BYTES[$s]}" "$h $s" &
        PROGRESS_PID=$!
    fi
    if [[ -z "$TARGET" ]]; then
        rsync "${opts[@]}" "$src/" "$dest/" >>"$MAIN_LOG" 2>&1
    else
        rsync "${opts[@]}" -e "$SSH_CMD ${SSH_OPTS[*]}" "$src/" "$TARGET:$dest/" >>"$MAIN_LOG" 2>&1
    fi
    rc=$?
    if [[ -n "$PROGRESS_PID" ]]; then kill "$PROGRESS_PID" 2>/dev/null; wait "$PROGRESS_PID" 2>/dev/null; PROGRESS_PID=""; fi

    if (( rc != 0 )); then
        log ERROR "$h : rsync failed (rc=$rc) -- see $MAIN_LOG"
        FAILED=$((FAILED + 1))
        continue
    fi
    if (( DRY_RUN == 1 )); then
        log INFO "$h : dry-run OK"
        OK=$((OK + 1))
        continue
    fi

    read -r dc db <<< "$(dest_stats "$dest")"
    dc="${dc:-0}"; db="${db:-0}"
    if (( dc == ${S_COUNT[$s]} && db == ${S_BYTES[$s]} )); then
        log INFO "$h : OK -- $dc files, $(human_bytes "$db") on $(dest_label), identical to the disk"
        OK=$((OK + 1))
    else
        log WARN "$h : INCOMPLETE -- $dc/${S_COUNT[$s]} files, $(human_bytes "$db")/$(human_bytes "${S_BYTES[$s]}")"
        SHORT=$((SHORT + 1))
    fi

    # keep the analysis-side path working, exactly like get_data.sh does
    if [[ -n "$LINK_ROOT" ]]; then
        dest_sh "[ -L '$LINK_ROOT' ] || { mkdir -p '$LINK_ROOT' 2>/dev/null;
                 [ -e '$LINK_ROOT/$cl' ] && [ ! -L '$LINK_ROOT/$cl' ] ||
                 ln -sfn '$DATA_ROOT/$cl' '$LINK_ROOT/$cl'; }" >/dev/null 2>&1 &&
            log INFO "symlink kept: $LINK_ROOT/$cl -> $DATA_ROOT/$cl"
    fi
done

# ------------------------------- summary -------------------------------------
log INFO "---------------------------------------------------------------"
log INFO "SUMMARY  ingested: $OK   incomplete: $SHORT   failed: $FAILED"
for p in "${PLAN[@]}"; do
    IFS=$'\t' read -r s h cl <<< "$p"
    read -r dc db <<< "$(dest_stats "$DATA_ROOT/$cl/${h}_${s}")"
    log INFO "  $h -> $(dest_label):$DATA_ROOT/$cl/${h}_${s} : ${dc:-0} file(s) $(human_bytes "${db:-0}")"
done
log INFO "log: $MAIN_LOG"
(( FAILED > 0 || SHORT > 0 )) && exit 1
exit 0
