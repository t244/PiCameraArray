#!/bin/bash

# Wait for SSD to be mounted (up to 60 seconds)
SSD_MOUNT="/media/pi/HIKSEMI"
TIMEOUT=60
ELAPSED=0

echo "Waiting for SSD to mount..."
while [ ! -d "$SSD_MOUNT" ] && [ $ELAPSED -lt $TIMEOUT ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

if [ -d "$SSD_MOUNT" ]; then
    echo "✓ SSD mounted at $SSD_MOUNT"
else
    echo "⚠ SSD not found after ${TIMEOUT}s - will use SD card fallback"
fi

# Start the camera agent (manages trigger_mode itself; boots into capture mode)
python3 /home/pi/PiCameraArray/capture/camera_agent.py
