#!/usr/bin/env bash
# all 90 frames on every 8th burst, after run_full.sh has finished
set -u
# (main run already finished when this was started)
for s in 20260728_073439 20260729_053037; do
  echo "=== $s $(date +%T)"
  cp /scratch/takatani/PiCameraArray/moisture/levels_${s}_f6.gains.json /scratch/takatani/PiCameraArray/moisture/levels_${s}_f1e8.gains.json
  docker run --rm --name takatani_bandlevels90_$s --user 1000:1000 \
    -v /home/takatani/projects/PiCameraArray:/workspace/code:ro \
    -v /data/CameraArray:/workspace/data:ro \
    -v /scratch/takatani/PiCameraArray:/workspace/out \
    -w /workspace/out -e HOME=/tmp --entrypoint python clothref:latest \
    /workspace/code/analyze/band_levels.py /workspace/data/packed/$s \
    --calib /workspace/code/calibration_results.npz \
    --out /workspace/out/moisture/levels_${s}_f1e8.jsonl --frames 0:90:1 --every 8 --workers 6
done
echo "=== ALL DONE $(date +%T)"
