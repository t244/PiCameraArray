#!/bin/bash

# Set libcamera camera timeout to 10 minutes
sudo sed -i 's/^[[:space:]]*#\?[[:space:]]*"camera_timeout_value_ms":[[:space:]]*[0-9]*/"camera_timeout_value_ms": 600000/' /usr/share/libcamera/pipeline/rpi/vc4/rpi_apps.yaml

# Make the startup script executable
sudo chmod +x /home/pi/PiCameraArray/capture/startup.sh

# Install and start the picamera-capture systemd service
sudo cp /home/pi/PiCameraArray/config/picamera-capture.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable picamera-capture.service

# Create data directory for captured images
sudo mkdir -p /home/pi/PiCameraArray/data
sudo chown -R pi:pi /home/pi/PiCameraArray/data

# Create data directory on SSD (if mounted)
if [ -d "/media/pi/HIKSEMI" ]; then
    sudo mkdir -p /media/pi/HIKSEMI/data
    sudo chown -R pi:pi /media/pi/HIKSEMI/data
    echo "✓ SSD detected and data directory created"
else
    echo "⚠ SSD not detected - will use SD card fallback"
fi
