#!/usr/bin/env bash
# full-burst level extraction for both runs; started in tmux by Claude 2026-11-05
set -u
for s in 20260728_073439 20260729_053037; do
  echo "=== $s $(date +%T)"
  docker run --rm --name takatani_bandlevels_$s --user 1000:1000 \
    -v /home/takatani/projects/PiCameraArray:/workspace/code:ro \
    -v /data/CameraArray:/workspace/data:ro \
    -v /scratch/takatani/PiCameraArray:/workspace/out \
    -w /workspace/out -e HOME=/tmp --entrypoint python clothref:latest \
    /workspace/code/analyze/band_levels.py /workspace/data/packed/$s \
    --calib /workspace/code/calibration_results.npz \
    --out /workspace/out/moisture/levels_${s}_f6.jsonl --frames 0:90:6 --workers 8
done
echo "=== ALL DONE $(date +%T)"
