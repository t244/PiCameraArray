#!/usr/bin/env python3
"""
Compute pairwise image subtraction for all combination pairs of monochrome PNG
images in a directory. Results are saved as color-mapped images.

Usage:
    python subtract.py <input_dir> [--output <output_dir>] [--colormap JET]
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import cv2


COLORMAP_OPTIONS = {
    "JET": cv2.COLORMAP_JET,
    "HOT": cv2.COLORMAP_HOT,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "TURBO": cv2.COLORMAP_TURBO,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pairwise image subtraction with colormap output"
    )
    parser.add_argument("input_dir", help="Directory containing monochrome PNG images")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory (default: <input_dir>/subtracted)"
    )
    parser.add_argument(
        "--colormap", "-c", default="JET",
        choices=COLORMAP_OPTIONS.keys(),
        help="Colormap to apply (default: JET)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else input_dir / "subtracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    colormap = COLORMAP_OPTIONS[args.colormap]

    # Load all PNG images
    png_files = sorted(input_dir.glob("*.png"))
    if len(png_files) < 2:
        print(f"Error: Need at least 2 PNG files, found {len(png_files)}")
        sys.exit(1)

    print(f"Found {len(png_files)} images")
    images = {}
    for f in png_files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Warning: Could not read {f.name}, skipping")
            continue
        images[f.stem] = img

    names = list(images.keys())
    n_pairs = len(names) * (len(names) - 1) // 2
    print(f"Computing {n_pairs} subtraction pairs -> {output_dir}")

    count = 0
    for name_a, name_b in combinations(names, 2):
        diff = cv2.absdiff(images[name_a], images[name_b])
        colored = cv2.applyColorMap(diff, colormap)
        out_path = output_dir / f"{name_a}_minus_{name_b}.png"
        cv2.imwrite(str(out_path), colored)
        count += 1

    print(f"Done. Saved {count} images to {output_dir}")


if __name__ == "__main__":
    main()
