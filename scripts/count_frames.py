#!/usr/bin/env python3
"""count_frames.py - count the frames in every burst npz of a session.

Checks that each burst holds the expected number of frames (fps x burst_s,
60 at 30 fps x 2 s). Only the npy header inside each npz is read, so a
120 MB burst costs a few kB of I/O; safe to run on a Pi while it records.

Runs on a Pi (default: the newest session on the SSD) or on a PC over
collected data (any directories; files are grouped by host from the file
name <host>_burstNNNN_<timestamp>.npz).

Output, one key=value line per burst, then one summary line per host:
    burst host=e00;burst=0002;frames=60;maxgap_ms=41;status=ok;file=...
    host  host=e00;total=12;bursts=12;ok=11;short=0;long=0;warmup=1;bad=0;
          expect=60;short_list=
    (total = bursts in the session(s); bursts = those checked, see --last.
    A host whose total is below the others missed whole bursts.)

status: ok      frames == expect
        warmup  the first burst of a session with fewer frames (the trigger
                box fires a 10-pulse warm-up train when capture starts)
        short   fewer frames than expected          -> dropped frames
        long    more frames than expected           -> two bursts merged?
        bad     unreadable (still being written, or truncated)

maxgap_ms is the largest gap between consecutive frame timestamps. They are
taken when the agent receives the frame, not by the sensor, so they jitter;
treat a large gap as a hint only. The frame count is what decides.

Exit status: 0 if every burst is ok/warmup, 1 otherwise, 2 if nothing found.

Usage:
    python3 count_frames.py                      # newest session on the SSD
    python3 count_frames.py --last 5             # its 5 most recent bursts
    python3 count_frames.py --expect 72 DIR ...  # other directories/files
"""
import argparse
import glob
import os
import re
import sys
import time
import zipfile
from collections import OrderedDict
from datetime import datetime

import numpy as np

SSD_DATA = "/media/pi/HIKSEMI/data"
NAME_RE = re.compile(r"^(?P<host>.+?)_burst(?P<burst>\d+)_(?P<ts>[\d_]+)\.npz$")
RECENT_S = 5.0      # a file modified this recently may still be being written


def newest_session(root):
    dirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    return max(dirs, key=os.path.getmtime) if dirs else None


def frame_count(path):
    """Number of frames in a burst npz, from the npy header only."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        member = next((m for m in ("frames_raw.npy", "frames.npy")
                       if m in names), None)
        if member is None:
            raise ValueError("no frames_raw/frames array")
        with z.open(member) as f:
            major, _ = np.lib.format.read_magic(f)
            read = (np.lib.format.read_array_header_1_0 if major == 1
                    else np.lib.format.read_array_header_2_0)
            shape, _, _ = read(f)
    return int(shape[0])


def max_gap_ms(path):
    """Largest gap between consecutive timestamps, or -1 if unavailable."""
    try:
        with np.load(path) as d:
            ts = [str(t) for t in d["timestamps"]]
        t = [datetime.strptime(s, "%Y%m%d_%H%M%S_%f") for s in ts]
        if len(t) < 2:
            return -1
        return int(round(max((b - a).total_seconds()
                             for a, b in zip(t, t[1:])) * 1000))
    except Exception:
        return -1


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "*_burst*.npz"),
                               recursive=True)
        elif p.endswith(".npz"):
            files.append(p)
    out = []
    for f in sorted(set(files)):
        m = NAME_RE.match(os.path.basename(f))
        if m:
            out.append((m.group("host"), int(m.group("burst")), f))
    out.sort(key=lambda x: (x[0], os.path.dirname(x[2]), x[1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*",
                    help="session directories or npz files "
                         f"(default: newest session in {SSD_DATA})")
    ap.add_argument("--expect", type=int, default=60,
                    help="frames per burst (fps x burst_s; default 60)")
    ap.add_argument("--last", type=int, default=0,
                    help="only the N most recent bursts per host")
    ap.add_argument("--no-gaps", action="store_true",
                    help="skip the timestamp gap check (faster)")
    args = ap.parse_args()

    paths = args.paths
    if not paths:
        s = newest_session(SSD_DATA)
        if s is None:
            print(f"error=no session directory under {SSD_DATA}")
            return 2
        paths = [s]

    bursts = collect(paths)
    if not bursts:
        print(f"error=no burst npz in {' '.join(paths)}")
        return 2

    by_host = OrderedDict()
    for host, n, f in bursts:
        by_host.setdefault(host, []).append((n, f))

    all_ok = True
    summaries = []
    for host, items in by_host.items():
        total = len(items)
        if args.last > 0:
            items = items[-args.last:]
        counts = {"ok": 0, "warmup": 0, "short": 0, "long": 0, "bad": 0}
        short_list = []
        prev_dir = None
        for n, f in items:
            first_in_session = os.path.dirname(f) != prev_dir and n == 1
            prev_dir = os.path.dirname(f)
            try:
                if time.time() - os.path.getmtime(f) < RECENT_S:
                    raise IOError("modified just now (still being written?)")
                k = frame_count(f)
            except Exception as e:
                counts["bad"] += 1
                all_ok = False
                print(f"burst host={host};burst={n:04d};frames=-1;"
                      f"maxgap_ms=-1;status=bad;err={e};file={f}")
                continue
            if k == args.expect:
                status = "ok"
            elif k < args.expect and first_in_session:
                status = "warmup"
            elif k < args.expect:
                status = "short"
                short_list.append(f"{n:04d}:{k}")
            else:
                status = "long"
                short_list.append(f"{n:04d}:{k}")
            counts[status] += 1
            if status not in ("ok", "warmup"):
                all_ok = False
            gap = -1 if args.no_gaps else max_gap_ms(f)
            print(f"burst host={host};burst={n:04d};frames={k};"
                  f"maxgap_ms={gap};status={status};file={f}")
        summaries.append(
            f"host host={host};total={total};bursts={len(items)};"
            f"ok={counts['ok']};"
            f"short={counts['short']};long={counts['long']};"
            f"warmup={counts['warmup']};bad={counts['bad']};"
            f"expect={args.expect};short_list={','.join(short_list)}")

    for s in summaries:
        print(s)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
