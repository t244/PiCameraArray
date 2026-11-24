#!/usr/bin/env bash
#
# preflight_check.sh - report the readiness of one Pi as a single line of
# key=value pairs. Read-only; changes nothing.
#
# Emitted keys:
#   host     hostname
#   svc      picamera-capture service state
#   ssd      yes | NO   - is the SSD actually mounted (not just the directory)
#   free     free space on the SSD
#   fstab    number of LABEL=HIKSEMI entries in /etc/fstab (expect 1)
#   usbdisk  USB disks enumerated (comma-separated), - if none
#   label    yes | NO   - does /dev/disk/by-label/HIKSEMI exist
#   disks    every USB disk: name/size/speed/labels/mnt|-, comma-separated
#   hiksemi  number of partitions labelled HIKSEMI (1, or one per pooled disk)
#   fstype   filesystem at the capture path: ext4, fuse.mergerfs, or -
#   thr      vcgencmd get_throttled (0x0 = no under-voltage since boot)
#   wifi     comma-separated "profile/priority" for every wifi profile
#   mode     capture | preview
#   count    frames captured this session (0 = no trigger arriving)
#   bursts   bursts written this session
#   arduino  True | False - serial link detected by the agent
#   sess     session directory in use
#   temp     CPU temperature
#
set -uo pipefail

MOUNT="${MOUNT:-/media/pi/HIKSEMI}"

# Several of the commands below print a value AND exit non-zero (systemctl
# is-active on a stopped unit, grep -c with no match). Take the first line and
# fall back only when it is genuinely empty, so no stray newline sneaks into
# the single-line output contract.
first() { head -n 1 | tr -d '\r\n'; }

out="host=$(hostname | first)"

svc="$(systemctl is-active picamera-capture 2>/dev/null | first)"
out+=";svc=${svc:-unknown}"

if mountpoint -q "$MOUNT"; then
    out+=";ssd=yes"
    out+=";free=$(df -h --output=avail "$MOUNT" 2>/dev/null | tail -1 | tr -d ' \r\n')"
else
    out+=";ssd=NO;free=-"
fi

# The entry that mounts the capture path: LABEL=HIKSEMI (one SSD) or the
# mergerfs pool line (several SSDs, see setup_ssd_pool.sh).
fstab="$(grep -cE "^[^#[:space:]]+[[:space:]]+${MOUNT}[[:space:]]" /etc/fstab 2>/dev/null | first)"
out+=";fstab=${fstab:-0}"

# Is a USB disk even enumerated? Distinguishes "the SSD is unplugged or the
# port is dead" from "the SSD is there but was never formatted or mounted".
usbdisk="$(lsblk -dn -o NAME,TRAN 2>/dev/null \
           | grep -w usb | tr -s ' ' | cut -d' ' -f1 | paste -sd, - | first)"
out+=";usbdisk=${usbdisk:--}"

lbl=NO
compgen -G "/dev/disk/by-label/${LABEL_NAME:-HIKSEMI}*" >/dev/null && lbl=yes
out+=";label=$lbl"

# Every USB disk, not just the one that got mounted: a Pi may carry several
# (e13 pools four 250 GB SSDs). Per disk "name/size/link speed/labels/mounted":
#   sda/1000G/5000M/HIKSEMI/mnt     speed 5000M = USB3, 480M = USB2
disks=""
hik=0
while read -r name tran size; do
    [[ "$tran" == usb ]] || continue
    dev="$(readlink -f "/sys/block/$name/device" 2>/dev/null)"
    spd="$(cat "$dev/../../../../speed" 2>/dev/null | tr -d '\r\n')"
    labels=""; mnt="-"
    while IFS= read -r line; do
        NAME="" LABEL="" MOUNTPOINT=""
        eval "$line"
        [[ -n "$LABEL" ]] && labels+="${LABEL}+"
        [[ "$LABEL" =~ ^${LABEL_NAME:-HIKSEMI}[0-9]*$ ]] && hik=$((hik + 1))
        [[ -n "$MOUNTPOINT" ]] && mnt="mnt"
    done < <(lsblk -nP -o NAME,LABEL,MOUNTPOINT "/dev/$name" 2>/dev/null | tail -n +2)
    labels="${labels%+}"
    disks+="${name}/$((size / 1000000000))G/${spd:-?}M/${labels:--}/${mnt},"
done < <(lsblk -dnb -o NAME,TRAN,SIZE 2>/dev/null)
out+=";disks=${disks%,};hiksemi=$hik"

# What is mounted at the capture path: ext4 (one SSD) or fuse.mergerfs (pool).
fstype="$(findmnt -no FSTYPE "$MOUNT" 2>/dev/null | first)"
out+=";fstype=${fstype:--}"

# Several bus-powered SSDs can overdraw the Pi's USB supply. 0x0 = never
# throttled or under-voltage since boot.
thr="$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2 | first)"
out+=";thr=${thr:-?}"

wifi="$(nmcli -t -f NAME,TYPE connection show 2>/dev/null \
        | awk -F: '$NF=="802-11-wireless"{NF--; print}' OFS=: \
        | while IFS= read -r n; do
              [[ -n "$n" ]] || continue
              p="$(nmcli -g connection.autoconnect-priority connection show "$n" 2>/dev/null || echo '?')"
              printf '%s/%s,' "$n" "$p"
          done)"
out+=";wifi=${wifi%,}"

status="$(curl -s -m 5 localhost:8000/status 2>/dev/null || true)"
if [[ -n "$status" ]]; then
    agent="$(printf '%s' "$status" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    print("mode=?;count=-1;bursts=-1;arduino=?;sess=?;temp=?")
    sys.exit()
t = d.get("cpu_temp")
print("mode=%s;count=%s;bursts=%s;arduino=%s;sess=%s;temp=%s" % (
    d.get("mode", "?"),
    d.get("capture_count", -1),
    d.get("burst_count", -1),
    d.get("has_arduino", "?"),
    d.get("session_dir", "?"),
    ("%.1f" % t) if isinstance(t, (int, float)) else "?",
))' 2>/dev/null)"
    out+=";${agent:-mode=?;count=-1;bursts=-1;arduino=?;sess=?;temp=?}"
else
    out+=";mode=DOWN;count=-1;bursts=-1;arduino=?;sess=?;temp=?"
fi

# One line, always - the caller parses it as key=value pairs.
printf '%s\n' "$(printf '%s' "$out" | tr -d '\r\n')"
