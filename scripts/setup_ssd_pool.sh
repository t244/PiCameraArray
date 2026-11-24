#!/usr/bin/env bash
#
# setup_ssd_pool.sh - pool several smaller SSDs on one Pi into the capture
# path /media/pi/HIKSEMI with mergerfs.
#
# For a Pi that has no single disk large enough (e13 in Nov 2026: four
# 250 GB SSDs instead of one 1 TB). Every disk keeps its own ext4 filesystem:
#
#     disk N   ext4, LABEL=HIKSEMI<N>, mounted at /mnt/hiksemi<N>
#              holds  pool/data/<session>/<host>_burst....npz
#     pool     mergerfs of /mnt/hiksemi*/pool  ->  /media/pi/HIKSEMI
#
# Why mergerfs rather than LVM/RAID0/btrfs: files are placed whole on one
# disk, so a disk that dies or drops off the bus loses only its own bursts,
# and each disk can be read alone on a PC. Each new file goes to the disk
# with the most free space (category.create=mfs), which spreads the bursts
# evenly over USB3 and USB2 disks alike.
#
# The pool branches are the "pool" directories INSIDE each disk, not the
# mount points: a disk that failed to mount leaves an empty directory on the
# SD card at /mnt/hiksemiN, which must never become a place to write.
#
# Reading the data back: each disk is a normal capture disk one level down.
#     ./construct/ingest_ssd.sh ... /mnt/hiksemi1/pool
# Every disk holds a part of the same session directories; ingest them one
# after another into the same collection.
#
# ---------------------------------------------------------------------------
#  THIS SCRIPT DESTROYS ALL DATA ON EVERY TARGET DISK.
#  It runs in INSPECT mode by default and changes nothing until you pass
#  CONFIRM=YES. The boot disk is never a candidate.
# ---------------------------------------------------------------------------
#
# Usage on the Pi (copy it over, or run from the repo):
#     sudo bash setup_ssd_pool.sh                       # inspect only
#     sudo CONFIRM=YES bash setup_ssd_pool.sh           # format + pool
#     sudo CONFIRM=YES DEVICES="sda sdb" bash setup_ssd_pool.sh
#
# Environment:
#     CONFIRM      YES to write. Anything else = inspect only.
#     DEVICES      disks to use, e.g. "sda sdb sdc sdd". Default: every
#                  non-boot disk >= MIN_SIZE_GB, USB3 ones first.
#     LABEL        base label / pool name (default HIKSEMI)
#     MOUNTPOINT   pool path (default /media/pi/HIKSEMI)
#     MIN_SIZE_GB  ignore disks smaller than this (default 50)
#     OWNER        owner of the data directories (default pi)
#
set -euo pipefail

LABEL="${LABEL:-HIKSEMI}"
MOUNTPOINT="${MOUNTPOINT:-/media/pi/${LABEL}}"
MIN_SIZE_GB="${MIN_SIZE_GB:-50}"
OWNER="${OWNER:-pi}"
CONFIRM="${CONFIRM:-no}"
DEVICES="${DEVICES:-}"
SERVICE="picamera-capture.service"
BRANCH_ROOT="/mnt/$(echo "$LABEL" | tr '[:upper:]' '[:lower:]')"   # /mnt/hiksemi<N>

HOST="$(hostname)"
log()  { echo "[$HOST] $*"; }
warn() { echo "[$HOST] WARNING: $*" >&2; }
die()  { echo "[$HOST] ERROR: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "must run as root (use sudo)"
for tool in lsblk findmnt sgdisk mkfs.ext4 blkid partprobe wipefs; do
    command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"
done

# --- 1. candidate disks ------------------------------------------------------
root_src="$(findmnt -no SOURCE /)"
boot_disk="$(lsblk -no PKNAME "$root_src" 2>/dev/null || true)"
[[ -n "$boot_disk" ]] || boot_disk="$(basename "$root_src")"

usb_speed() {   # sda -> 5000 | 480 | ?
    local dev; dev="$(readlink -f "/sys/block/$1/device" 2>/dev/null)"
    cat "$dev/../../../../speed" 2>/dev/null || echo "?"
}

if [[ -n "$DEVICES" ]]; then
    read -r -a disks <<< "$DEVICES"
else
    mapfile -t disks < <(
        lsblk -dnb -o NAME,TYPE,SIZE \
        | awk -v boot="$boot_disk" -v min="$((MIN_SIZE_GB * 1000000000))" \
              '$2=="disk" && $1!=boot && $3+0 >= min {print $1}' \
        | while read -r d; do printf '%s %s\n' "$(usb_speed "$d")" "$d"; done \
        | sort -k1,1nr -k2,2 | awk '{print $2}')
fi
(( ${#disks[@]} >= 1 )) || die "no candidate disk (>= ${MIN_SIZE_GB} GB, not the boot disk)"
for d in "${disks[@]}"; do
    d="${d#/dev/}"
    [[ -b "/dev/$d" ]] || die "/dev/$d is not a block device"
    [[ "$d" != "$boot_disk" ]] || die "refusing to touch the boot disk /dev/$d"
done

log "boot disk: /dev/$boot_disk"
log "--- disks to pool (in order; disk 1 is HIKSEMI1) ---"
n=0
for d in "${disks[@]}"; do
    d="${d#/dev/}"; n=$((n + 1))
    log "  ${LABEL}${n} <- /dev/$d  $(lsblk -dn -o SIZE,MODEL,SERIAL "/dev/$d" | tr -s ' ')  USB $(usb_speed "$d") Mbps"
    lsblk -n -o NAME,FSTYPE,LABEL,SIZE,MOUNTPOINT "/dev/$d" | sed "s/^/[$HOST]       /"
done

if [[ "$CONFIRM" != "YES" ]]; then
    log ""
    log "INSPECT MODE - nothing was changed."
    log "To ERASE the disks above and pool them at $MOUNTPOINT:"
    log "    sudo CONFIRM=YES bash $0"
    exit 0
fi

# --- 2. mergerfs ---------------------------------------------------------------
if ! command -v mergerfs >/dev/null 2>&1; then
    log "installing mergerfs"
    apt-get install -y mergerfs >/dev/null || die "apt-get install mergerfs failed (network?)"
fi
log "mergerfs: $(mergerfs --version 2>&1 | head -1)"

# --- 3. release everything ------------------------------------------------------
log "stopping $SERVICE"
systemctl stop "$SERVICE" 2>/dev/null || true
if mountpoint -q "$MOUNTPOINT"; then
    umount "$MOUNTPOINT" || umount -l "$MOUNTPOINT" || die "cannot unmount $MOUNTPOINT"
fi
for d in "${disks[@]}"; do
    d="${d#/dev/}"
    while IFS= read -r mp; do
        [[ -n "$mp" ]] || continue
        log "  umount $mp"
        umount "$mp" || umount -l "$mp" || die "cannot unmount $mp"
    done < <(lsblk -nro MOUNTPOINT "/dev/$d" | grep -v '^$' || true)
done

# --- 4. fstab: drop every earlier entry for the label, the pool or members ----
cp /etc/fstab "/etc/fstab.bak.$(date +%Y%m%d_%H%M%S)"
sed -i "\#^LABEL=${LABEL}[0-9]*[[:space:]]#d" /etc/fstab
sed -i "\#^[^#]*[[:space:]]${MOUNTPOINT}[[:space:]]#d" /etc/fstab
sed -i "\#^[^#]*[[:space:]]${BRANCH_ROOT}[0-9]*[[:space:]]#d" /etc/fstab
sed -i "/^# PiCameraArray capture SSD/d" /etc/fstab

# --- 5. format and mount each disk ----------------------------------------------
fstab_add="\n# PiCameraArray capture SSD pool (added by setup_ssd_pool.sh)\n"
after=""
n=0
for d in "${disks[@]}"; do
    d="/dev/${d#/dev/}"; n=$((n + 1))
    lbl="${LABEL}${n}"; mp="${BRANCH_ROOT}${n}"
    log "--- $d -> $lbl at $mp ---"
    wipefs -a "$d" >/dev/null
    sgdisk --zap-all "$d" >/dev/null
    sgdisk --new=1:0:0 --typecode=1:8300 --change-name=1:"$lbl" "$d" >/dev/null
    partprobe "$d"; udevadm settle 2>/dev/null || sleep 2
    part="${d}1"; [[ -b "$part" ]] || die "no partition on $d"
    mkfs.ext4 -q -F -L "$lbl" -m 1 "$part"
    mkdir -p "$mp"
    fstab_add+="LABEL=${lbl}  ${mp}  ext4  defaults,noatime,nofail,x-systemd.device-timeout=60  0  2\n"
    unit="$(systemd-escape -p --suffix=mount "$mp")"
    after+=",x-systemd.after=${unit}"
done

pool_opts="defaults,allow_other,category.create=mfs,minfreespace=1G,moveonenospc=true,fsname=${LABEL}pool,nofail${after}"
fstab_add+="${BRANCH_ROOT}*/pool  ${MOUNTPOINT}  fuse.mergerfs  ${pool_opts}  0  0\n"
printf "$fstab_add" >> /etc/fstab
systemctl daemon-reload
udevadm settle 2>/dev/null || sleep 2

for i in $(seq 1 "$n"); do
    mp="${BRANCH_ROOT}${i}"
    mount "$mp" || die "cannot mount $mp"
    mkdir -p "$mp/pool/data"
    chown -R "$OWNER:$OWNER" "$mp/pool"
done

mkdir -p "$MOUNTPOINT"
mount "$MOUNTPOINT" || die "cannot mount the pool at $MOUNTPOINT"
[[ "$(findmnt -no FSTYPE "$MOUNTPOINT")" == fuse.mergerfs ]] || die "$MOUNTPOINT is not a mergerfs pool"
[[ -d "$MOUNTPOINT/data" ]] || die "$MOUNTPOINT/data missing - no member disk in the pool?"

# --- 6. verify: writable by the capture user, and lands on a member disk -------
t="$MOUNTPOINT/data/.pool_test_$$"
sudo -u "$OWNER" dd if=/dev/zero of="$t" bs=1M count=128 conv=fsync status=none \
    || die "write test failed on $MOUNTPOINT/data"
where="$(ls -d ${BRANCH_ROOT}*/pool/data/.pool_test_$$ 2>/dev/null | head -1)"
rm -f "$t"
[[ -n "$where" ]] || die "test file did not land on any member disk"
log "write test OK (landed on ${where%%/pool/*})"

log "--- pool ---"
findmnt "$MOUNTPOINT" | sed "s/^/[$HOST]   /"
df -h "$MOUNTPOINT" ${BRANCH_ROOT}* | sed "s/^/[$HOST]   /"

log "starting $SERVICE"
systemctl start "$SERVICE" 2>/dev/null || warn "could not start $SERVICE"
log "DONE. $n disks pooled at $MOUNTPOINT (mergerfs, create=mfs)."
