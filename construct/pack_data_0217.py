#!/usr/bin/env python3
"""
Pack images from 20260217_dataset into packed_data.

Source structure:
  20260217_dataset/
    e00/
      20260217_072421/   e00_000000_...png, e00_000001_...png, ...
      20260217_073931/   ...
      ...
    e01/ ...
    e13/                 (empty — camera was not running)
    e15/ ...

By default, selects the 3rd-from-latest timestamp folder from each camera
and packs all images into packed_data/<counter_id>/e00.png..e15.png.

Usage:
  python construct/pack_data_0217.py
  python construct/pack_data_0217.py --offset 0          # latest timestamp
  python construct/pack_data_0217.py --offset 5 --out my_output
  python construct/pack_data_0217.py --dataset 20260217_dataset --dry-run
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

CAM_NAMES = [f"e{i:02d}" for i in range(16)]


def get_timestamp_dirs(cam_dir: Path) -> list:
    """Return sorted list of timestamp subdirectories in a camera folder."""
    dirs = []
    for p in cam_dir.iterdir():
        if p.is_dir() and re.match(r"\d{8}_\d{6}$", p.name):
            dirs.append(p)
    return sorted(dirs, key=lambda d: d.name)


def select_target_dir(cam_dir: Path, offset: int) -> Path | None:
    """Select the target timestamp directory for a camera.

    Args:
        cam_dir: Camera directory (e.g. 20260217_dataset/e00/).
        offset:  0 = latest, 1 = 2nd from latest, 2 = 3rd from latest, ...

    Returns:
        Path to the selected timestamp directory, or None if unavailable.
    """
    ts_dirs = get_timestamp_dirs(cam_dir)
    if not ts_dirs:
        return None
    idx = len(ts_dirs) - 1 - offset
    if idx < 0:
        return None
    return ts_dirs[idx]


def extract_counter(filename: str) -> str | None:
    """Extract counter ID from filename like e00_000003_20260217_085135_476.png."""
    parts = filename.split("_")
    if len(parts) >= 2:
        return parts[1]
    return None


def pack_0217(
    dataset_dir: Path,
    output_root: Path,
    offset: int = 2,
    dry_run: bool = False,
):
    """Pack images from the target timestamp of each camera.

    Args:
        dataset_dir: Path to 20260217_dataset.
        output_root: Output directory (e.g. packed_data).
        offset:      Timestamp selection: 0=latest, 2=3rd from latest.
        dry_run:     If True, only print what would be done.
    """
    print(f"Dataset : {dataset_dir}")
    print(f"Offset  : {offset} (0=latest)")
    print()

    # Discover target directories per camera
    cam_targets: dict[str, Path | None] = {}
    for cam in CAM_NAMES:
        cam_dir = dataset_dir / cam
        if not cam_dir.is_dir():
            print(f"  [SKIP] {cam}: directory not found")
            cam_targets[cam] = None
            continue
        target = select_target_dir(cam_dir, offset)
        if target is None:
            print(f"  [SKIP] {cam}: no timestamp dirs (empty camera)")
            cam_targets[cam] = None
        else:
            print(f"  {cam}: {target.name}")
            cam_targets[cam] = target

    active_cams = {c: t for c, t in cam_targets.items() if t is not None}
    if not active_cams:
        print("\nError: no active cameras found.")
        sys.exit(1)

    # Use the representative timestamp (from first active camera) as folder name
    # e.g. packed_data/20260217_084516/000003/e00.png
    representative_ts = next(iter(active_cams.values())).name
    output_dir = output_root / representative_ts
    print(f"Output  : {output_dir}")

    # Discover all counter IDs from active cameras
    all_counters: set[str] = set()
    for cam, target_dir in active_cams.items():
        for p in target_dir.glob("*.png"):
            cid = extract_counter(p.name)
            if cid is not None:
                all_counters.add(cid)

    counters = sorted(all_counters)
    print(f"\nCounters: {len(counters)}  "
          f"({counters[0]} .. {counters[-1]})" if counters else "")
    print(f"Active cameras: {len(active_cams)}/{len(CAM_NAMES)}  "
          f"(missing: {[c for c in CAM_NAMES if c not in active_cams]})")
    print()

    # Pack each counter
    total_copied = 0
    for cid in counters:
        dest_dir = output_dir / cid
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for cam, target_dir in active_cams.items():
            # Find the image with this counter in this camera's target dir
            src = None
            for p in target_dir.glob("*.png"):
                if extract_counter(p.name) == cid:
                    src = p
                    break

            if src is None:
                continue

            dst = dest_dir / f"{cam}.png"
            if dry_run:
                print(f"  [DRY] {src.name} -> {dst}")
            else:
                shutil.copy2(src, dst)
            copied += 1

        total_copied += copied

    print(f"Done: {len(counters)} frames x {len(active_cams)} cameras "
          f"= {total_copied} images packed into {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Pack images from 20260217_dataset into packed_data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python construct/pack_data_0217.py                    "
            "# 3rd from latest\n"
            "  python construct/pack_data_0217.py --offset 0         "
            "# latest timestamp\n"
            "  python construct/pack_data_0217.py --offset 5 --dry-run\n"
        ),
    )
    parser.add_argument(
        "--dataset", "-d", default="20260217_dataset",
        help="Path to the dataset directory (default: 20260217_dataset)",
    )
    parser.add_argument(
        "--out", "-o", default="packed_data",
        help="Output root directory (default: packed_data)",
    )
    parser.add_argument(
        "--offset", "-n", type=int, default=2,
        help="Timestamp offset from latest: 0=latest, 1=2nd, 2=3rd (default: 2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without copying",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        print(f"Error: {dataset_dir} is not a directory")
        sys.exit(1)

    output_root = Path(args.out)
    pack_0217(dataset_dir, output_root, offset=args.offset, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
