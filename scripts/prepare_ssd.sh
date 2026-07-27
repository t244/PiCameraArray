#!/usr/bin/env bash
#
# prepare_ssd.sh - format a replacement SSD and mount it at /media/pi/HIKSEMI
#
# Keeping the label identical to the old disk means camera_agent.py,
# startup.sh, config.sh and the analysis scripts need no changes: the mount
# path /media/pi/HIKSEMI stays valid.
#
# Unlike the old setup this does NOT rely on the desktop (udisks2) automount.
# A LABEL-based /etc/fstab entry mounts the disk during boot, before the
# capture service starts, which removes the startup race entirely.
#
# ext4 is used deliberately: the array is power-cycled without a clean
# shutdown, and the writer thread is flushing npz files between bursts, so
# unclean unmounts are guaranteed. ext4's journal limits the damage to the
# file in flight. (Read it on Windows with:  wsl --mount --bare \\.\PHYSICALDRIVEn)
#
# ---------------------------------------------------------------------------
#  THIS SCRIPT DESTROYS ALL DATA ON THE TARGET DISK.
#  It runs in INSPECT mode by default and changes nothing until you pass
#  CONFIRM=YES.  The boot disk is detected and refused unconditionally.
# ---------------------------------------------------------------------------
#
# Usage on the Pi:
#     sudo bash prepare_ssd.sh                    # inspect only, no changes
#     sudo CONFIRM=YES bash prepare_ssd.sh        # format + mount
#     sudo CONFIRM=YES DEVICE=/dev/sda bash prepare_ssd.sh
#
# Environment overrides:
#     FSTAB_ONLY  1 = do not touch the data. Reuse the existing HIKSEMI
#                 filesystem and only (re)write the fstab entry and mount it.
#                 Non-destructive, so CONFIRM is not required.
#     CONFIRM     "YES" to actually write. Anything else = inspect only.
#     DEVICE      target disk, e.g. /dev/sda. Default: auto-detect.
#     LABEL       filesystem label      (default: HIKSEMI)
#     MOUNTPOINT  where to mount it     (default: /media/pi/HIKSEMI)
#     OWNER       owner of the data dir (default: pi)
#     MIN_SIZE_GB refuse disks smaller than this (default: 50)
#     LAZY_INIT   1 = mke2fs default, inode tables zeroed by a kernel thread
#                 in the background after mount (default).
#                 0 = eager init, everything written up front. Slower, and
#                 observed to fail on this hardware, so it is not the default.
#                 The background work is ~4 GB spread over a few minutes,
#                 against 57 s of idle time per 3 s burst - not a concern.
#
set -euo pipefail

LABEL="${LABEL:-HIKSEMI}"
MOUNTPOINT="${MOUNTPOINT:-/media/pi/${LABEL}}"
OWNER="${OWNER:-pi}"
MIN_SIZE_GB="${MIN_SIZE_GB:-50}"
CONFIRM="${CONFIRM:-no}"
FSTAB_ONLY="${FSTAB_ONLY:-0}"
DEVICE="${DEVICE:-}"
LAZY_INIT="${LAZY_INIT:-1}"
SERVICE="picamera-capture.service"

HOST="$(hostname)"
log()  { echo "[$HOST] $*"; }
warn() { echo "[$HOST] WARNING: $*" >&2; }
die()  { echo "[$HOST] ERROR: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "must run as root (use sudo)"

for tool in lsblk findmnt sgdisk mkfs.ext4 blkid partprobe; do
    command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool (apt install gdisk util-linux e2fsprogs)"
done

# ---------------------------------------------------------------------------
# 1. Work out which disk holds the running system - it is never a candidate
# ---------------------------------------------------------------------------
root_src="$(findmnt -no SOURCE / || true)"
[[ -n "$root_src" ]] || die "cannot determine the root filesystem source"
boot_disk="$(lsblk -no PKNAME "$root_src" 2>/dev/null || true)"
[[ -n "$boot_disk" ]] || boot_disk="$(basename "$root_src")"
log "root filesystem: $root_src  (boot disk: /dev/$boot_disk)"

# ---------------------------------------------------------------------------
# 2. Pick the target disk
# ---------------------------------------------------------------------------
mapfile -t candidates < <(
    lsblk -dn -o NAME,TYPE,SIZE,MODEL --bytes \
    | awk -v boot="$boot_disk" -v min="$((MIN_SIZE_GB * 1000000000))" \
        '$2=="disk" && $1!=boot && $3+0 >= min {print $1}'
)

log "--- block devices ---"
lsblk -o NAME,SIZE,TYPE,FSTYPE,LABEL,MOUNTPOINT | sed "s/^/[$HOST]   /"

if [[ -n "$DEVICE" ]]; then
    [[ -b "$DEVICE" ]] || die "$DEVICE is not a block device"
    target="$DEVICE"
elif [[ ${#candidates[@]} -eq 1 ]]; then
    target="/dev/${candidates[0]}"
    log "auto-detected target: $target"
elif [[ ${#candidates[@]} -eq 0 ]]; then
    die "no candidate disk found (>= ${MIN_SIZE_GB} GB, not the boot disk). Is the SSD plugged in?"
else
    die "several candidate disks found: ${candidates[*]}. Re-run with DEVICE=/dev/sdX to choose."
fi

target_base="$(basename "$target")"
[[ "$target_base" != "$boot_disk" ]] || die "refusing to touch the boot disk ($target)"

# Belt and braces: refuse if any partition of the target currently carries /
if findmnt -no SOURCE / | grep -q "^${target}"; then
    die "refusing: $target appears to host the root filesystem"
fi

log "--- target: $target ---"
lsblk -o NAME,SIZE,TYPE,FSTYPE,LABEL,MOUNTPOINT "$target" | sed "s/^/[$HOST]   /"

# ---------------------------------------------------------------------------
# 3. Inspect mode stops here
# ---------------------------------------------------------------------------
if [[ "$FSTAB_ONLY" != "1" && "$CONFIRM" != "YES" ]]; then
    log ""
    log "INSPECT MODE - nothing was changed."
    log "To format $target as ext4 labelled '$LABEL' and mount it at $MOUNTPOINT:"
    log "    sudo CONFIRM=YES DEVICE=$target bash prepare_ssd.sh"
    log "To keep the data and only repair the fstab entry and mount:"
    log "    sudo FSTAB_ONLY=1 bash prepare_ssd.sh"
    exit 0
fi

# ---------------------------------------------------------------------------
# 4. Either reuse the existing filesystem, or wipe and rebuild it
# ---------------------------------------------------------------------------
if [[ "$FSTAB_ONLY" == "1" ]]; then
    part="$(blkid -L "$LABEL" 2>/dev/null || true)"
    [[ -n "$part" ]] || die "no partition carries LABEL=$LABEL - is the SSD attached and formatted?"
    log "FSTAB_ONLY: keeping the existing filesystem on $part"
    log "stopping $SERVICE"
    systemctl stop "$SERVICE" 2>/dev/null || true
    # It may currently be mounted somewhere else (udisks2 picks its own path).
    while IFS= read -r mp; do
        [[ -n "$mp" && "$mp" != "$MOUNTPOINT" ]] || continue
        log "  umount $mp"
        umount "$mp" 2>/dev/null || umount -l "$mp" 2>/dev/null || warn "could not unmount $mp"
    done < <(lsblk -nro MOUNTPOINT "$part" | grep -v '^$' || true)
else

log "stopping $SERVICE"
systemctl stop "$SERVICE" 2>/dev/null || true

log "unmounting anything on $target"
while IFS= read -r mp; do
    [[ -n "$mp" ]] || continue
    log "  umount $mp"
    umount "$mp" || umount -l "$mp" || warn "could not unmount $mp"
done < <(lsblk -nro MOUNTPOINT "$target" | grep -v '^$' || true)

# A disk with the same label already mounted elsewhere (e.g. the old SSD
# still attached, showing up as /media/pi/HIKSEMI1) makes the fstab entry
# ambiguous. Refuse rather than guess.
others="$(blkid -t LABEL="$LABEL" -o device 2>/dev/null | grep -v "^${target}" || true)"
if [[ -n "$others" ]]; then
    die "another device already carries LABEL=$LABEL: $(echo "$others" | tr '\n' ' ')
       Detach the old SSD before running this, or the fstab entry is ambiguous."
fi

log "wiping partition table on $target"
wipefs -a "$target" >/dev/null
sgdisk --zap-all "$target" >/dev/null

log "creating a single GPT partition"
sgdisk --new=1:0:0 --typecode=1:8300 --change-name=1:"$LABEL" "$target" >/dev/null
partprobe "$target"
udevadm settle 2>/dev/null || sleep 2

# Partition node is sda1 / nvme0n1p1 depending on the device naming scheme
if [[ -b "${target}1" ]]; then
    part="${target}1"
elif [[ -b "${target}p1" ]]; then
    part="${target}p1"
else
    die "cannot find the new partition on $target"
fi
log "partition: $part"

mkfs_opts=(-F -L "$LABEL" -m 1)
if [[ "$LAZY_INIT" == "1" ]]; then
    log "formatting ext4 (lazy init - finishes fast, background IO afterwards)"
else
    log "formatting ext4 (eager init - takes a few minutes, no background IO later)"
    mkfs_opts+=(-E lazy_itable_init=0,lazy_journal_init=0)
fi
mkfs.ext4 "${mkfs_opts[@]}" "$part"

fi   # end of the format-vs-reuse branch

uuid="$(blkid -s UUID -o value "$part")"
log "filesystem in use: LABEL=$LABEL UUID=$uuid"

# ---------------------------------------------------------------------------
# 5. Deterministic mount via /etc/fstab
# ---------------------------------------------------------------------------
mkdir -p "$MOUNTPOINT"

# nofail so a missing SSD never blocks boot; a generous device timeout because
# a USB SSD can take a while to enumerate. startup.sh retries the mount anyway.
fstab_line="LABEL=${LABEL}  ${MOUNTPOINT}  ext4  defaults,noatime,nofail,x-systemd.device-timeout=60  0  2"

cp /etc/fstab "/etc/fstab.bak.$(date +%Y%m%d_%H%M%S)"
# Drop any previous entry for this label or mount point, then append ours.
sed -i "\#^[^#]*[[:space:]]${MOUNTPOINT}[[:space:]]#d" /etc/fstab
sed -i "\#^LABEL=${LABEL}[[:space:]]#d" /etc/fstab
printf '\n# PiCameraArray capture SSD (added by prepare_ssd.sh)\n%s\n' "$fstab_line" >> /etc/fstab
log "fstab entry: $fstab_line"

systemctl daemon-reload
if ! mountpoint -q "$MOUNTPOINT"; then
    mount "$MOUNTPOINT" || mount "/dev/disk/by-label/${LABEL}" "$MOUNTPOINT"
fi
mountpoint -q "$MOUNTPOINT" || die "mount failed - check /etc/fstab and 'journalctl -xe'"

mkdir -p "$MOUNTPOINT/data"
chown -R "$OWNER":"$OWNER" "$MOUNTPOINT/data"
log "created $MOUNTPOINT/data (owner $OWNER)"

# ---------------------------------------------------------------------------
# 6. Verify: it must be mounted, writable by the capture user, and fast
#    enough to absorb a burst between triggers
# ---------------------------------------------------------------------------
testfile="$MOUNTPOINT/data/.write_test_$$"
if ! sudo -u "$OWNER" dd if=/dev/zero of="$testfile" bs=1M count=256 oflag=direct \
        status=none 2>/dev/null; then
    sudo -u "$OWNER" dd if=/dev/zero of="$testfile" bs=1M count=256 conv=fsync status=none \
        || die "write test failed - $MOUNTPOINT/data is not writable by $OWNER"
fi
speed="$(LC_ALL=C dd if=/dev/zero of="$testfile" bs=1M count=256 conv=fsync 2>&1 \
        | tail -1 | sed 's/.*, //')"
rm -f "$testfile"
log "write test OK (256 MB: $speed)"

avail_gb=$(( $(stat -f -c '%a' "$MOUNTPOINT") * $(stat -f -c '%S' "$MOUNTPOINT") / 1000000000 ))
log "free space: ${avail_gb} GB"
# readme: 1.98 MB x fps x burst_s per burst per camera; defaults 30 fps x 3 s
log "at default 30 fps x 3 s / 60 s period that is ~$(( avail_gb * 1000 / 178 / 60 )) hours of capture"

log "restarting $SERVICE"
systemctl start "$SERVICE" 2>/dev/null || warn "could not start $SERVICE"

log ""
log "DONE. $target -> $MOUNTPOINT (ext4, LABEL=$LABEL)"
log "Mount now survives reboots via /etc/fstab, independent of the desktop automount."
findmnt "$MOUNTPOINT" | sed "s/^/[$HOST]   /"
