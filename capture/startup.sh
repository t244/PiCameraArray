#!/bin/bash

# Wait for the SSD to be mounted (up to 60 seconds).
# Test with mountpoint, not [ -d ]: the mount point directory is created
# permanently by prepare_ssd.sh, so a directory test is always true and would
# mask an SSD that failed to mount.
SSD_MOUNT="/media/pi/HIKSEMI"
SSD_LABEL="HIKSEMI"
TIMEOUT=60
ELAPSED=0

mkdir -p "$SSD_MOUNT"

echo "Waiting for SSD at $SSD_MOUNT ..."
while ! mountpoint -q "$SSD_MOUNT" && [ $ELAPSED -lt $TIMEOUT ]; do
    # Do not just wait. A USB SSD can enumerate after systemd has already
    # passed local-fs.target, and the fstab mount is then never retried -
    # the disk sits there, healthy, and simply never gets mounted.
    # Mounting it ourselves is a no-op once it is up.
    if [ -e "/dev/disk/by-label/$SSD_LABEL" ]; then
        # Try fstab, then by label, then by device node. Keep the last error:
        # a silent failure here is what sends a whole session to the SD card.
        MOUNT_ERR="$( { mount "$SSD_MOUNT" \
            || mount -L "$SSD_LABEL" "$SSD_MOUNT" \
            || mount "/dev/disk/by-label/$SSD_LABEL" "$SSD_MOUNT"; } 2>&1 )"
    else
        MOUNT_ERR="/dev/disk/by-label/$SSD_LABEL does not exist yet"
    fi
    mountpoint -q "$SSD_MOUNT" && break
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

if mountpoint -q "$SSD_MOUNT"; then
    echo "✓ SSD mounted at $SSD_MOUNT ($(df -h --output=avail "$SSD_MOUNT" | tail -1 | tr -d ' ') free)"
else
    # The SD card holds roughly 178 MB per burst per camera at the defaults,
    # so it fills in well under an hour. Make this impossible to miss in the log.
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!! SSD NOT MOUNTED after ${TIMEOUT}s - falling back to the SD card."
    echo "!! At default settings the SD card fills in under an hour."
    echo "!! Last mount error: ${MOUNT_ERR:-none}"
    if [ -e "/dev/disk/by-label/$SSD_LABEL" ]; then
        echo "!! The disk IS present but would not mount. Filesystem damaged?"
        echo "!!   sudo fsck -f /dev/disk/by-label/$SSD_LABEL"
    else
        echo "!! No block device with LABEL=$SSD_LABEL - the SSD is not enumerating."
        echo "!!   Check the USB port and cable:  lsusb ; dmesg | tail -30"
    fi
    echo "!! Also check:  findmnt $SSD_MOUNT ; grep HIKSEMI /etc/fstab"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi

# Start the camera agent (manages trigger_mode itself; boots into capture mode)
python3 /home/pi/PiCameraArray/capture/camera_agent.py
