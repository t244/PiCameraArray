#!/bin/bash

# Make the startup script executable
sudo chmod +x /home/pi/PiCameraArray/capture/startup.sh

# Install and start the picamera-capture systemd service
sudo cp /home/pi/PiCameraArray/config/picamera-capture.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable picamera-capture.service
sudo systemctl start picamera-capture.service
