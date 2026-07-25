#!/usr/bin/env python3
"""
play_bursts.py - Synchronized 16-camera burst player (4x4 mosaic).

Plays one burst (npz files produced by camera_agent.py) from all 16
cameras side by side, like the live dashboard but for recorded data.

Input layout (as produced by construct/get_data.py):
    collected_data/<session>/
        e00_<session>/e00_burst0001_<ts>.npz
        e01_<session>/e01_burst0001_<ts>.npz
        ...
Each npz contains:
    frames     : (N, 1088, 1456) uint8
    timestamps : (N,) str  "%Y%m%d_%H%M%S_%f" (ms)

Usage:
    python analyze/play_bursts.py collected_data/<session>            # latest common burst
    python analyze/play_bursts.py collected_data/<session> --burst 2
    python analyze/play_bursts.py collected_data/<session> --list
    python analyze/play_bursts.py collected_data/<session> --save out.mp4

Keys:
    space : pause / resume        a / d : step back / forward (paused)
    + / - : faster / slower       r     : restart
    q,ESC : quit
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import cv2

HOSTS = [f"e{i:02d}" for i in range(16)]
GRID = (4, 4)
BURST_RE = re.compile(r"_burst(\d+)_")


def find_bursts(session_dir: Path):
    """Map burst index -> {host: npz path}."""
    bursts = {}
    for host in HOSTS:
        host_dirs = sorted(session_dir.glob(f"{host}_*"))
        if not host_dirs:
            continue
        for f in host_dirs[0].glob(f"{host}_burst*.npz"):
            m = BURST_RE.search(f.name)
            if m:
                bursts.setdefault(int(m.group(1)), {})[host] = f
    return bursts


def load_burst(files: dict, scale: float):
    """Load all hosts' frames for one burst, downscaled."""
    data = {}
    n_min = None
    for host, path in sorted(files.items()):
        d = np.load(path)
        fr = d["frames"]
        ts = d["timestamps"]
        if scale != 1.0:
            h = max(1, int(fr.shape[1] * scale))
            w = max(1, int(fr.shape[2] * scale))
            fr = np.stack([
                cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
                for f in fr])
        data[host] = (fr, ts)
        n = len(fr)
        n_min = n if n_min is None else min(n_min, n)
        print(f"  {host}: {path.name}  {n} frames")
    return data, (n_min or 0)


def make_mosaic(data: dict, idx: int, tile_h: int, tile_w: int):
    """Assemble one 4x4 mosaic frame with host labels."""
    rows = []
    for r in range(GRID[0]):
        tiles = []
        for c in range(GRID[1]):
            host = HOSTS[r * GRID[1] + c]
            if host in data and idx < len(data[host][0]):
                tile = data[host][0][idx].copy()
            else:
                tile = np.zeros((tile_h, tile_w), np.uint8)
            cv2.putText(tile, host, (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser(
        description="16-camera synchronized burst player")
    ap.add_argument("session_dir", type=Path,
                    help="collected_data/<session> directory")
    ap.add_argument("--burst", type=int, default=None,
                    help="burst index (default: latest common)")
    ap.add_argument("--list", action="store_true",
                    help="list available bursts and exit")
    ap.add_argument("--scale", type=float, default=0.25,
                    help="per-camera scale (default 0.25 -> 1456x1088 mosaic)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="initial playback fps (default 30)")
    ap.add_argument("--save", type=Path, default=None,
                    help="write mosaic to mp4 instead of interactive play")
    args = ap.parse_args()

    if not args.session_dir.is_dir():
        sys.exit(f"Not a directory: {args.session_dir}")

    bursts = find_bursts(args.session_dir)
    if not bursts:
        sys.exit("No burst npz files found")

    if args.list:
        for idx in sorted(bursts):
            hosts = sorted(bursts[idx])
            print(f"burst {idx}: {len(hosts)} cameras "
                  f"({hosts[0]}..{hosts[-1]})")
        return

    # Default: newest burst that all available hosts share
    if args.burst is None:
        full = [i for i, f in bursts.items() if len(f) == max(
            len(f2) for f2 in bursts.values())]
        args.burst = max(full)
    if args.burst not in bursts:
        sys.exit(f"Burst {args.burst} not found "
                 f"(available: {sorted(bursts)})")

    print(f"Loading burst {args.burst} "
          f"({len(bursts[args.burst])} cameras, scale {args.scale})...")
    data, n_frames = load_burst(bursts[args.burst], args.scale)
    if n_frames == 0:
        sys.exit("No frames")

    any_host = next(iter(data))
    tile_h, tile_w = data[any_host][0].shape[1:3]
    ref_ts = data[any_host][1]
    print(f"Mosaic: {tile_w * GRID[1]}x{tile_h * GRID[0]}, "
          f"{n_frames} frames")

    # --- save mode ---
    if args.save:
        vw = cv2.VideoWriter(
            str(args.save), cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps, (tile_w * GRID[1], tile_h * GRID[0]), isColor=False)
        for i in range(n_frames):
            vw.write(make_mosaic(data, i, tile_h, tile_w))
        vw.release()
        print(f"Saved: {args.save}")
        return

    # --- interactive playback ---
    fps = args.fps
    idx = 0
    paused = False
    win = f"PiCameraArray burst {args.burst} (q:quit space:pause a/d:step +/-:speed)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Wall-clock scheduling: rendering time is subtracted from the wait so
    # 90 frames at 30 fps really take 3.0 s (not 3 s + render overhead)
    next_t = time.perf_counter()

    while True:
        mosaic = make_mosaic(data, idx, tile_h, tile_w)
        status = (f"frame {idx + 1}/{n_frames}  {fps:.0f}fps"
                  f"{'  [PAUSED]' if paused else ''}  "
                  f"t={ref_ts[min(idx, len(ref_ts) - 1)]}")
        cv2.putText(mosaic, status, (6, mosaic.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
        cv2.imshow(win, mosaic)

        if paused:
            delay_ms = 0  # wait for a key indefinitely
        else:
            next_t += 1.0 / fps
            delay_ms = max(1, int((next_t - time.perf_counter()) * 1000))
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
            next_t = time.perf_counter()  # reset schedule on resume
        elif key == ord("d"):
            idx = min(idx + 1, n_frames - 1)
        elif key == ord("a"):
            idx = max(idx - 1, 0)
        elif key in (ord("+"), ord(";")):
            fps = min(fps * 1.5, 240)
            next_t = time.perf_counter()
        elif key == ord("-"):
            fps = max(fps / 1.5, 1)
            next_t = time.perf_counter()
        elif key == ord("r"):
            idx = 0
            next_t = time.perf_counter()
        elif not paused:
            idx += 1
            if idx >= n_frames:
                idx = 0  # loop
                next_t = time.perf_counter()

        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
