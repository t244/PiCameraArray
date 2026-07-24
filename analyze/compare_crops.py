#!/usr/bin/env python3
"""
Compare original e05 images with defencing_v2 results.

For each timestamp in packed_data, crops a specified ROI from:
  1. The original e05 image (undistorted)
  2. The defencing_v2 result

and saves them into a comparisons/ folder.

The v2 result is already cropped to the overlap region.  The crop offset
(y0, x0) is read from the v2 cache file so that the SAME region can be
cut from the undistorted e05 original.  Because ref=e05 uses an identity
homography, the cache crop coordinates map directly to undistorted-e05
pixel coordinates.

Usage:
  python analyze/compare_crops.py
  python analyze/compare_crops.py --frame 10
  python analyze/compare_crops.py --frame 10 --roi 400,200,250,250
  python analyze/compare_crops.py --frame 10 --depth 700
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _find_cache(v2_dir: Path, depth: int) -> Path | None:
    """Search for the v2 cache npz in v2_dir or sibling folders."""
    name = f"_v2_cache_d{depth}_ref5.npz"
    # Direct match
    p = v2_dir / name
    if p.exists():
        return p
    # Search sibling v2 subdirectories (same frame, different run name)
    parent = v2_dir.parent  # .../v2/
    if parent.exists():
        for candidate in parent.glob(f"*/{name}"):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Crop and compare e05 originals vs defencing_v2 results.",
    )
    parser.add_argument(
        "--frame", "-f", type=int, default=0,
        help="Frame number (auto zero-padded to 6 digits, default: 0)",
    )
    parser.add_argument(
        "--roi", type=str, default=None,
        help="ROI as x,y,w,h in the v2-result coordinate system",
    )
    parser.add_argument(
        "--packed", default="packed_data",
        help="Packed data root (default: packed_data)",
    )
    parser.add_argument(
        "--outputs", default="outputs",
        help="Outputs root (default: outputs)",
    )
    parser.add_argument(
        "--calib", default="calibration_results.npz",
        help="Calibration file for undistortion",
    )
    parser.add_argument(
        "--depth", "-d", type=int, default=750,
        help="Focus depth used for v2 (default: 750)",
    )
    parser.add_argument(
        "--v2-dir", default=None,
        help="v2 subdirectory name (default: d<depth>)",
    )
    parser.add_argument(
        "--out", "-o", default="comparisons",
        help="Output directory (default: comparisons)",
    )
    args = parser.parse_args()

    frame = f"{args.frame:06d}"
    packed_root = Path(args.packed)
    outputs_root = Path(args.outputs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    v2_subdir = args.v2_dir if args.v2_dir else f"d{args.depth}"

    # Parse ROI (in v2-result coordinates)
    roi = None
    if args.roi:
        parts = [int(x) for x in args.roi.split(",")]
        if len(parts) != 4:
            print("Error: --roi must be x,y,w,h (4 integers)")
            sys.exit(1)
        roi = tuple(parts)

    # Load calibration for undistortion of e05
    calib_path = Path(args.calib)
    if not calib_path.exists():
        print(f"Error: calibration file not found: {calib_path}")
        sys.exit(1)
    calib = np.load(str(calib_path), allow_pickle=True)
    K_e05 = calib["e05_camera_matrix"]
    dist_e05 = calib["e05_dist_coeffs"]

    # Discover datasets
    datasets = sorted(
        [d for d in packed_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not datasets:
        print(f"Error: no datasets found in {packed_root}")
        sys.exit(1)

    count = 0
    for ds in datasets:
        ts_name = ds.name
        frame_dir = ds / frame

        if not frame_dir.exists():
            continue

        e05_path = frame_dir / "e05.png"
        if not e05_path.exists():
            print(f"  [SKIP] {ts_name}/{frame}: e05.png not found")
            continue

        v2_dir = outputs_root / ts_name / frame / "v2" / v2_subdir
        v2_path = v2_dir / "result_defencing_v2.png"
        if not v2_path.exists():
            print(f"  [SKIP] {ts_name}/{frame}: v2 result not found")
            continue

        # Load images
        orig = cv2.imread(str(e05_path), cv2.IMREAD_GRAYSCALE)
        if orig is None:
            continue
        orig = cv2.undistort(orig, K_e05, dist_e05)

        v2_img = cv2.imread(str(v2_path), cv2.IMREAD_GRAYSCALE)
        if v2_img is None:
            continue

        v2_h, v2_w = v2_img.shape[:2]
        orig_h, orig_w = orig.shape[:2]

        # Get crop offset from cache (v2 coordinates → original coordinates)
        cache = _find_cache(v2_dir, args.depth)
        if cache is not None:
            data = np.load(str(cache))
            crop_info = data["crop"]  # [y0, y1, x0, x1]
            cy0, cx0 = int(crop_info[0]), int(crop_info[2])
        else:
            # Fallback: assume centered crop
            cy0 = (orig_h - v2_h) // 2
            cx0 = (orig_w - v2_w) // 2
            print(f"    (cache not found, using estimated offset)")

        # Determine ROI in v2 space
        if roi is not None:
            rx, ry, rw, rh = roi
        else:
            rw, rh = min(250, v2_w), min(250, v2_h)
            rx = (v2_w - rw) // 2
            ry = (v2_h - rh) // 2

        # Clamp to v2 bounds
        rx = max(0, min(rx, v2_w - rw))
        ry = max(0, min(ry, v2_h - rh))
        rw = min(rw, v2_w - rx)
        rh = min(rh, v2_h - ry)

        # Crop v2
        v2_crop = v2_img[ry:ry + rh, rx:rx + rw]

        # Map to original e05 coordinates and crop
        ox = cx0 + rx
        oy = cy0 + ry
        ox = max(0, min(ox, orig_w - rw))
        oy = max(0, min(oy, orig_h - rh))
        orig_crop = orig[oy:oy + rh, ox:ox + rw]

        # Safety resize if sizes differ slightly
        if orig_crop.shape != v2_crop.shape:
            orig_crop = cv2.resize(
                orig_crop, (v2_crop.shape[1], v2_crop.shape[0])
            )

        # Save
        cv2.imwrite(str(out_dir / f"{ts_name}_{frame}_orig.png"), orig_crop)
        cv2.imwrite(str(out_dir / f"{ts_name}_{frame}_v2.png"), v2_crop)

        count += 1
        print(f"  {ts_name}/{frame}: saved ({rw}x{rh}  "
              f"v2@({rx},{ry}) orig@({ox},{oy}))")

    print(f"\nDone. {count} pairs saved to {out_dir}/")


if __name__ == "__main__":
    main()
