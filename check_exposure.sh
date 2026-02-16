#!/bin/bash

# Switch to internal clock (free-running) mode
echo 0 > /sys/module/imx296/parameters/trigger_mode

# Preview camera with manual exposure settings
# Exposure time in microseconds (e.g., 10000 = 10ms)
EXPOSURE_US=${1:?"Usage: $0 <exposure_us> (e.g., $0 10000 for 10ms)"}

rpicam-hello -t 0 \
    --viewfinder-width 1456 --viewfinder-height 1088 \
    --shutter $EXPOSURE_US \
    --gain 1.0 \
    --awbgains 1.0,1.0 \
    --denoise off \
    --info-text "exposure=%{exp}us gain=%{ag} focus=%{focus}"
