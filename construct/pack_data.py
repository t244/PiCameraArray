#!/usr/bin/env python3
"""
Pack images by counter ID from the `collected_data` dataset structure.

Source structure expected:
  collected_data/
    <dataset_timestamp>/
      e00/
        img_A_<counter>_...png
      e01/
        img_B_<counter>_...png
      ... e15/

This script collects images that share the same counter ID (the second chunk
when splitting the filename stem by underscore) from each camera directory
and copies them into `packed_data/<counter_id>/` renamed to `e00.png`..`e15.png`.

Usage examples:
  python construct/pack_data.py --collected collected_data --counter 000123
  python construct/pack_data.py --collected collected_data --dataset 2026-02-13_120000 --counter 000123 --out packed_data
"""

import argparse
import shutil
from pathlib import Path
import sys
from typing import Optional

PI_NAMES = [f"e{i:02d}" for i in range(16)]


def find_latest_dataset(collected_root: Path) -> Optional[str]:
    """Return the name of the latest dataset directory under collected_root (by name sort)."""
    if not collected_root.exists() or not collected_root.is_dir():
        return None
    entries = [p.name for p in collected_root.iterdir() if p.is_dir()]
    if not entries:
        return None
    return sorted(entries)[-1]


def find_camera_file_for_counter(camera_dir: Path, counter_id: str) -> Optional[Path]:
    """Find a file in camera_dir whose filename stem's second chunk equals counter_id.

    Splits the stem by underscore. Example stem: 'img_000123_0001' -> second chunk '000123'.
    Returns the first matching Path or None.
    """
    if not camera_dir.exists() or not camera_dir.is_dir():
        return None
    for p in sorted(camera_dir.glob('*.png')):
        stem = p.stem
        parts = stem.split('_')
        if len(parts) >= 2 and parts[1] == counter_id:
            return p
    return None


def pack_by_counter(collected_root: Path, counter_id: str, dataset: Optional[str] = None, output_root: Optional[Path] = None) -> Path:
    """Create packed_data/<counter_id>/ with e00..e15 images copied from collected dataset.

    Args:
        collected_root: Path to `collected_data` root
        counter_id: Counter ID to select matching images
        dataset: Optional dataset directory name; if omitted, latest is used
        output_root: Root where `packed_data` will be created (default ./packed_data)

    Returns:
        Path to the created packed directory
    """
    collected_root = Path(collected_root)
    if output_root is None:
        output_root = Path.cwd() / 'packed_data'
    output_root = Path(output_root)

    if dataset is None:
        dataset = find_latest_dataset(collected_root)
        if dataset is None:
            raise FileNotFoundError(f"No dataset directories found under {collected_root}")

    dataset_dir = collected_root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    dest_dir = output_root / counter_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    missing_cameras = []
    copied = 0

    for idx, cam in enumerate(PI_NAMES):
        # camera directory may have timestamp suffix, find directory that startswith cam
        cam_dir = None
        for p in dataset_dir.iterdir():
            if p.is_dir() and p.name.startswith(cam):
                cam_dir = p
                break

        dst_name = f"{cam}.png"
        dst_path = dest_dir / dst_name

        if cam_dir is None:
            missing_cameras.append(cam)
            print(f"[WARN] {cam}: camera directory not found under {dataset_dir}")
            continue

        src = find_camera_file_for_counter(cam_dir, counter_id)
        if src is None:
            missing_cameras.append(cam)
            print(f"[WARN] {cam}: no file found for counter {counter_id} (checked {cam_dir})")
            continue

        shutil.copy2(src, dst_path)
        copied += 1

    print(f"Packed {copied} images into {dest_dir}")
    if missing_cameras:
        print(f"Missing cameras: {', '.join(missing_cameras)}")

    return dest_dir


def _cli():
    p = argparse.ArgumentParser(description='Pack images by counter ID into packed_data/<counter_id>/e00..e15.png')
    p.add_argument('--collected', '-c', required=False, default='collected_data', help='Root collected_data directory (default: collected_data)')
    p.add_argument('--dataset', '-d', required=False, help='Dataset directory name (if omitted, latest is used)')
    p.add_argument('--counter', '-k', required=True, type=int, help='Counter ID to pack (integer, will be formatted as %%06d)')
    p.add_argument('--out', '-o', required=False, default=None, help='Output root for packed data (default: ./packed_data)')

    args = p.parse_args()

    try:
        counter_str = f"{args.counter:06d}"
        out = pack_by_counter(Path(args.collected), counter_str, dataset=args.dataset, output_root=Path(args.out) if args.out else None)
        print(f"Done: {out}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    _cli()
