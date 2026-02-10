#!/bin/bash

# Enable trigger mode for the imx296 camera module
echo 1 > /sys/module/imx296/parameters/trigger_mode

# Start the trigger capture script
python3 /home/pi/PiCameraArray/capture/trigger_capture.py
